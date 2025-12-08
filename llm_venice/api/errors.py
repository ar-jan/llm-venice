"""Shared error handling utilities for Venice API responses."""

from typing import Optional, Tuple

import httpx


class VeniceAPIError(RuntimeError):
    """Normalized error for Venice API failures."""

    def __init__(
        self,
        action: str,
        status_code,
        reason: Optional[str],
        *,
        error_code: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        self.action = action
        self.status_code = status_code
        self.reason = reason
        self.error_code = error_code
        self.detail = detail

        message = _format_error_message(
            action=action,
            status_code=status_code,
            reason=reason,
            error_code=error_code,
            detail=detail,
        )
        super().__init__(message)


def _format_error_message(
    *,
    action: str,
    status_code,
    reason: Optional[str],
    error_code: Optional[str],
    detail: Optional[str],
) -> str:
    """Build a readable error message for API failures."""
    reason_text = reason or "HTTP error"
    message = f"{action} failed: Venice API returned {status_code} {reason_text}"

    if error_code:
        message = f"{message} ({error_code})"

    if detail:
        message = f"{message} - {detail}"

    return message


def _extract_error_parts(response: httpx.Response) -> Tuple[Optional[str], Optional[str]]:
    """Extract a useful error code/detail tuple from an httpx.Response."""
    try:
        payload = response.json()
    except ValueError:
        payload = None

    error_code = None
    detail = None

    if isinstance(payload, dict):
        error_code = payload.get("code")
        detail = payload.get("message") or payload.get("error")
    elif payload is not None:
        detail = str(payload)

    if not detail:
        text = (response.text or "").strip()
        detail = text or None

    return error_code, detail


def raise_api_error(action: str, error: httpx.HTTPStatusError):
    """Normalize HTTPStatusError into a VeniceAPIError for callers."""
    response = error.response
    status = response.status_code if response is not None else "unknown status"
    reason = response.reason_phrase if response is not None else "HTTP error"

    error_code = None
    detail = None
    if response is not None:
        error_code, detail = _extract_error_parts(response)

    raise VeniceAPIError(
        action=action,
        status_code=status,
        reason=reason,
        error_code=error_code,
        detail=detail,
    ) from error
