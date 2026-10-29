"""Service errors. Local file failures remain OSError; invalid inputs use ValidationError.

Failed saves may attach ``partial_path: Path`` naming retained bytes. If
cancellation arrives after the file was already published, the error instead
carries ``published_path: Path`` naming the complete file. Neither attribute
changes the exception's type, message, or original cause.
For cancellation, inspect the exception where the save is awaited: Python 3.10
wraps it when another task awaits the cancelled task.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math

import llm


class VeniceError(Exception):
    """Base class for Venice library failures."""


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
