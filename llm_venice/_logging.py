"""Debug logging of Venice HTTP traffic, switched on by environment variables.

``LLM_VENICE_SHOW_RESPONSES`` controls every request the plugin makes: chat,
media and admin. When it is unset or blank, ``LLM_OPENAI_SHOW_RESPONSES`` is used
instead. ``0``, ``false``, ``off`` and ``no`` disable logging; any other non-blank
value enables it. The setting is read on every request.

Logs go to stderr, redact credentials, summarise binary bodies and never consume
a stream: streamed bodies are either skipped or echoed as the caller reads them.
"""

import codecs
import json
import os
import sys
import textwrap
from collections.abc import AsyncGenerator, Iterator
from typing import Any, cast

import httpx2
import openai

VENICE_ENV = "LLM_VENICE_SHOW_RESPONSES"
OPENAI_ENV = "LLM_OPENAI_SHOW_RESPONSES"
_OFF = {"0", "false", "off", "no"}

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

# Long unbroken strings (base64 media, data URLs) are shortened so logs stay
# readable; prose such as prompts contains whitespace and is kept whole.
_MAX_STRING = 256
_STRING_PREVIEW = 64


def enabled() -> bool:
    """Return whether logging is on; a blank Venice value falls back like an unset one."""
    for name in (VENICE_ENV, OPENAI_ENV):
        value = os.environ.get(name, "").strip().lower()
        if value:
            return value not in _OFF
    return False


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): "[redacted]" if str(k).lower() in _SECRET_KEYS else redact(v)
            for k, v in cast(dict[str, Any], value).items()
        }
    if isinstance(value, list):
        return [redact(item) for item in cast(list[Any], value)]
    return value


def _has_space(value: str) -> bool:
    return any(char.isspace() for char in value)


def shorten(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_STRING and not _has_space(value):
        return f"{value[:_STRING_PREVIEW]}...[{len(value)} chars]"
    if isinstance(value, dict):
        return {k: shorten(v) for k, v in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        return [shorten(item) for item in cast(list[Any], value)]
    return value


def _echo(text: str = "") -> None:
    print(text, file=sys.stderr)


def _echo_headers(headers: httpx2.Headers) -> None:
    _echo("  Headers:")
    for key, value in headers.multi_items():
        if key.lower() in _SECRET_KEYS:
            value = "[redacted]"
        _echo(f"    {key}: {value}")


def _echo_body(text: str) -> None:
    _echo("  Body:")
    _echo(textwrap.indent(text, "    "))


def _echo_content(content: bytes, content_type: str) -> None:
    """Pretty-print JSON, show short text and summarise anything else."""
    if not content:
        return
    if "json" in content_type:
        try:
            body: Any = json.loads(content)
        except (ValueError, UnicodeError):
            _echo_body(f"[invalid JSON, {len(content)} bytes]")
        else:
            _echo_body(json.dumps(shorten(redact(body)), indent=2))
    elif content_type.startswith("text/"):
        _echo_body(content.decode(errors="replace"))
    else:
        _echo_body(f"[{content_type or 'unknown type'}, {len(content)} bytes]")


def log_request(request: httpx2.Request) -> None:
    if not enabled():
        return
    _echo(f"Venice request: {request.method} {request.url}")
    _echo_headers(request.headers)
    content_type = request.headers.get("content-type", "")
    try:
        content = request.content
    except httpx2.RequestNotRead:
        # Multipart uploads are streamed; reading them here would consume the body.
        length = request.headers.get("content-length", "unknown")
        _echo_body(f"[{content_type.split(';')[0] or 'streamed'}, {length} bytes]")
    else:
        _echo_content(content, content_type)


def log_response(response: httpx2.Response) -> None:
    if enabled():
        _echo(f"Venice response: {response.status_code} {response.reason_phrase}".rstrip())
        _echo_headers(response.headers)


def log_buffered(response: httpx2.Response) -> None:
    """Log a body that has already been read."""
    if enabled():
        _echo_content(response.content, response.headers.get("content-type", ""))


async def async_log_request(request: httpx2.Request) -> None:
    log_request(request)


async def async_log_response(response: httpx2.Response) -> None:
    log_response(response)


class _BodyEcho:
    """Echo decoded text chunks as they are read, starting with a Body: label."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._started = False
        self._ends_with_newline = True

    def write(self, chunk: bytes, final: bool = False) -> None:
        text = self._decoder.decode(chunk, final=final)
        if not text:
            return
        if not self._started:
            self._started = True
            _echo("  Body:")
        sys.stderr.write(text)
        sys.stderr.flush()
        self._ends_with_newline = text.endswith("\n")

    def close(self) -> None:
        self.write(b"", final=True)
        if self._started and not self._ends_with_newline:
            _echo()


def _is_text(response: httpx2.Response) -> bool:
    content_type = response.headers.get("content-type", "")
    return "json" in content_type or content_type.startswith("text/")


class _EchoResponse(httpx2.Response):
    """Response whose decoded body is echoed while the caller reads it."""

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        if not (enabled() and _is_text(self)):
            yield from super().iter_bytes(chunk_size)
            return
        echo = _BodyEcho()
        try:
            for chunk in super().iter_bytes(chunk_size):
                echo.write(chunk)
                yield chunk
        finally:
            echo.close()

    async def aiter_bytes(self, chunk_size: int | None = None) -> AsyncGenerator[bytes, None]:
        if not (enabled() and _is_text(self)):
            async for chunk in super().aiter_bytes(chunk_size):
                yield chunk
            return
        echo = _BodyEcho()
        try:
            async for chunk in super().aiter_bytes(chunk_size):
                echo.write(chunk)
                yield chunk
        finally:
            echo.close()


def _echo_response(response: httpx2.Response) -> httpx2.Response:
    return _EchoResponse(
        status_code=response.status_code,
        headers=response.headers,
        stream=response.stream,
        extensions=response.extensions,
    )


class _EchoTransport(httpx2.BaseTransport):
    def __init__(self, transport: httpx2.BaseTransport) -> None:
        self._transport = transport

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        return _echo_response(self._transport.handle_request(request))

    def close(self) -> None:
        self._transport.close()


class _AsyncEchoTransport(httpx2.AsyncBaseTransport):
    def __init__(self, transport: httpx2.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return _echo_response(await self._transport.handle_async_request(request))

    async def aclose(self) -> None:
        await self._transport.aclose()


def chat_http_client() -> httpx2.Client:
    """HTTP client for the OpenAI SDK that logs requests and echoes response bodies."""
    return openai.DefaultHttpxClient(
        transport=_EchoTransport(httpx2.HTTPTransport()),
        event_hooks={"request": [log_request], "response": [log_response]},
    )


def async_chat_http_client() -> httpx2.AsyncClient:
    """Async counterpart of :func:`chat_http_client`."""
    return openai.DefaultAsyncHttpxClient(
        transport=_AsyncEchoTransport(httpx2.AsyncHTTPTransport()),
        event_hooks={"request": [async_log_request], "response": [async_log_response]},
    )
