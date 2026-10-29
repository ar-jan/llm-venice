"""Binary-safe logging for owned clients, without consuming streaming bodies."""

import json
import os
import sys
from typing import Any, cast

import httpx2

_SECRET_KEYS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "apikey",
    "api_key",
    "token",
    "secret",
}

# Long strings (mostly base64 image data) are shortened so logs stay readable.
_MAX_STRING = 256
_STRING_PREVIEW = 64


def enabled() -> bool:
    value = os.environ.get(
        "LLM_VENICE_SHOW_RESPONSES", os.environ.get("LLM_OPENAI_SHOW_RESPONSES", "")
    )
    return value.strip().lower() not in {"", "0", "false", "off"}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): "[redacted]" if str(k).lower() in _SECRET_KEYS else redact(v)
            for k, v in cast(dict[str, Any], value).items()
        }
    if isinstance(value, list):
        return [redact(item) for item in cast(list[Any], value)]
    return value


def shorten(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_STRING:
        return f"{value[:_STRING_PREVIEW]}...[{len(value)} chars]"
    if isinstance(value, dict):
        return {k: shorten(v) for k, v in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        return [shorten(item) for item in cast(list[Any], value)]
    return value


def log_request(request: httpx2.Request) -> None:
    if enabled():
        print(f"Venice request: {request.method} {request.url}", file=sys.stderr)
        print(json.dumps(redact(dict(request.headers))), file=sys.stderr)


def log_response(response: httpx2.Response) -> None:
    if enabled():
        print(f"Venice response: {response.status_code}", file=sys.stderr)
        print(json.dumps(redact(dict(response.headers))), file=sys.stderr)


def log_buffered(response: httpx2.Response) -> None:
    if not enabled():
        return
    if "json" in response.headers.get("content-type", ""):
        try:
            body: Any = response.json()
        except (ValueError, UnicodeError):
            print("[invalid JSON body]", file=sys.stderr)
        else:
            print(json.dumps(shorten(redact(body))), file=sys.stderr)
    else:
        print(f"[{len(response.content)} bytes]", file=sys.stderr)


async def async_log_request(request: httpx2.Request) -> None:
    log_request(request)


async def async_log_response(response: httpx2.Response) -> None:
    log_response(response)
