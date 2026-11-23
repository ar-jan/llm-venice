"""Shared HTTP preparation and error mapping; transport lives in client.py."""

from functools import lru_cache
from importlib.metadata import version
from typing import Any, NoReturn, cast

import httpx2
import llm

from .errors import (
    AuthenticationError,
    ContentViolationError,
    MissingKeyError,
    RateLimitError,
    VeniceAPIError,
)

DEFAULT_TIMEOUT = httpx2.Timeout(30.0, read=120.0)


def resolve_key(key: str | None) -> str:
    resolved = llm.get_key(key, "venice", "LLM_VENICE_KEY")
    if not resolved:
        raise MissingKeyError("No key found for Venice")
    return resolved


@lru_cache(maxsize=1)
def _user_agent() -> str:
    return f"llm-venice/{version('llm-venice')} llm/{version('llm')}"


def request_headers(key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {key}",
        "Accept-Encoding": "gzip",
        "User-Agent": _user_agent(),
    }


# ImageContentViolationError's fields beside `error`. Venice sends retryable with
# every content-policy rejection and with no other 422.
_POLICY_FIELDS = ("suggested_prompt", "recommended_model", "retryable")


def _text(obj: dict[str, Any], key: str) -> str | None:
    value = obj.get(key)
    return value if isinstance(value, str) and value else None


def raise_api_error(
    action: str, error: httpx2.HTTPStatusError, *, policy_422: bool = False
) -> NoReturn:
    """Map an error status; ``policy_422`` marks endpoints whose 422s are all refusals."""
    response = error.response
    code: str | None = None
    detail: str | None = None
    try:
        payload: Any = response.json()
    except ValueError:
        payload = None
    obj = cast(dict[str, Any], payload) if isinstance(payload, dict) else {}
    if obj:
        value = obj.get("code")
        code = str(value) if value is not None else None
        value = obj.get("message") or obj.get("error")
        detail = str(value) if value is not None else None
    if not detail:
        detail = response.text.strip() or None
    if response.status_code == 422 and (
        policy_422 or any(obj.get(key) is not None for key in _POLICY_FIELDS)
    ):
        retryable = obj.get("retryable")
        raise ContentViolationError(
            action,
            response.status_code,
            response.reason_phrase,
            error_code=code,
            detail=detail,
            headers=response.headers,
            suggested_prompt=_text(obj, "suggested_prompt"),
            recommended_model=_text(obj, "recommended_model"),
            retryable=retryable if isinstance(retryable, bool) else None,
        ) from error
    cls = (
        AuthenticationError
        if response.status_code in {401, 403}
        else RateLimitError
        if response.status_code == 429
        else VeniceAPIError
    )
    raise cls(
        action,
        response.status_code,
        response.reason_phrase,
        error_code=code,
        detail=detail,
        headers=response.headers,
    ) from error


def check_status(response: httpx2.Response, action: str, *, policy_422: bool = False) -> None:
    try:
        response.raise_for_status()
    except httpx2.HTTPStatusError as exc:
        raise_api_error(action, exc, policy_422=policy_422)
