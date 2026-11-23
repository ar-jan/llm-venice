"""Service errors. Local file failures remain OSError; invalid inputs use ValidationError.

Failed saves may attach ``partial_path: Path`` naming retained bytes. If
cancellation arrives after the file was already published, the error instead
carries ``published_path: Path`` naming the complete file. Neither attribute
changes the exception's type, message, or original cause.
For cancellation, inspect the exception where the save is awaited.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
from typing import Any

import llm


def _restore(cls: type["VeniceError"], args: tuple[Any, ...]) -> "VeniceError":
    """Recreate a pickled or copied error; its attributes are restored afterwards."""
    error = cls.__new__(cls)
    error.args = args
    return error


class VeniceError(Exception):
    """Base class for Venice library failures. Errors can be pickled and copied."""

    def __reduce__(self) -> tuple[Any, ...]:
        # The default calls cls(*args), but subclass constructors take arguments
        # that args, the message, does not hold. Rebuild without __init__ instead.
        return (_restore, (type(self), self.args), self.__dict__)


class MissingKeyError(VeniceError, llm.NeedsKeyException):
    """No Venice key was supplied or available in llm's key store/environment."""


class VeniceConnectionError(VeniceError):
    """A transport failure, with the original HTTP exception as its cause."""


class VeniceResponseError(VeniceError):
    """A successful HTTP response did not contain the expected resource."""

    def __init__(self, action: str, detail: str) -> None:
        self.action = action
        super().__init__(f"{action} failed: {detail}")


class VeniceAPIError(VeniceError):
    """An unsuccessful HTTP status returned by Venice."""

    def __init__(
        self,
        action: str,
        status_code: int,
        reason: str | None,
        *,
        error_code: str | None = None,
        detail: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.action = action
        self.status_code = status_code
        self.reason = reason
        self.error_code = error_code
        self.detail = detail
        self.headers = dict(headers or {})
        message = f"{action} failed: Venice API returned {status_code} {reason or 'HTTP error'}"
        if error_code:
            message += f" ({error_code})"
        if detail:
            message += f" - {detail}"
        super().__init__(message)


class ContentViolationError(VeniceAPIError):
    """Venice or the model provider refused the request under a content policy (422).

    Raised for every 422 from image upscaling, and for image generation 422s that
    carry policy fields. Other 422s, such as an oversized style reference image,
    remain plain VeniceAPIError. ``suggested_prompt`` and ``recommended_model`` are
    optional alternatives. ``retryable`` says whether the same request may succeed
    on retry, and is None when Venice did not say. The message keeps VeniceAPIError's
    first line and adds one line per hint.
    """

    def __init__(
        self,
        action: str,
        status_code: int,
        reason: str | None,
        *,
        error_code: str | None = None,
        detail: str | None = None,
        headers: Mapping[str, str] | None = None,
        suggested_prompt: str | None = None,
        recommended_model: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.suggested_prompt = suggested_prompt
        self.recommended_model = recommended_model
        self.retryable = retryable
        super().__init__(
            action,
            status_code,
            reason,
            error_code=error_code,
            detail=detail,
            headers=headers,
        )
        self._summary = str(self)
        self.args = (self.describe(),)

    def describe(self, *, model_prefix: str = "") -> str:
        """The message, naming the recommended model with ``model_prefix`` before its ID.

        The llm adapters pass ``"venice/"``, since llm knows Venice models by that name.
        """
        lines = [self._summary]
        if self.suggested_prompt:
            lines.append(f"Suggested prompt: {self.suggested_prompt}")
        if self.recommended_model:
            lines.append(f"Recommended model: {model_prefix}{self.recommended_model}")
        if self.retryable is not None:
            lines.append(
                "Retrying may succeed."
                if self.retryable
                else "Retrying the same request will not succeed."
            )
        return "\n".join(lines)


class AuthenticationError(VeniceAPIError):
    """Venice rejected authentication or authorization (401 or 403)."""


class RateLimitError(VeniceAPIError):
    """Venice rate limited the request (429). No retry is performed automatically."""

    @property
    def retry_after(self) -> float | None:
        """Retry-After seconds (including HTTP dates), or None when unavailable."""
        value = next((v for k, v in self.headers.items() if k.lower() == "retry-after"), None)
        if value is None:
            return None
        try:
            seconds = float(value)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                seconds = (date - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                return None
        return max(0.0, seconds) if math.isfinite(seconds) else None
