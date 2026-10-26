"""Shared HTTP preparation and error mapping; transport lives in client.py."""

from importlib.metadata import version
from typing import Any, NoReturn, cast

import httpx2
import llm

from .errors import AuthenticationError, MissingKeyError, RateLimitError, VeniceAPIError

DEFAULT_TIMEOUT = httpx2.Timeout(30.0, read=120.0)


def resolve_key(key: str | None) -> str:
    resolved = llm.get_key(key, "venice", "LLM_VENICE_KEY")
    if not resolved:
        raise MissingKeyError("No key found for Venice")
    return resolved


def request_headers(key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {key}",
        "Accept-Encoding": "gzip",
        "User-Agent": f"llm-venice/{version('llm-venice')} llm/{version('llm')}",
    }


def raise_api_error(action: str, error: httpx2.HTTPStatusError) -> NoReturn:
    response = error.response
    code: str | None = None
    detail: str | None = None
    try:
        payload: Any = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        obj = cast(dict[str, Any], payload)
        value = obj.get("code")
        code = str(value) if value is not None else None
        value = obj.get("message") or obj.get("error")
        detail = str(value) if value is not None else None
    if not detail:
        detail = response.text.strip() or None
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


def check_status(response: httpx2.Response, action: str) -> None:
    try:
        response.raise_for_status()
    except httpx2.HTTPStatusError as exc:
        raise_api_error(action, exc)
