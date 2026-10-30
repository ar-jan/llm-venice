"""Debug logging of Venice HTTP traffic, switched on by environment variables.

``LLM_VENICE_SHOW_RESPONSES`` controls every request the plugin makes: chat,
media and admin. When it is unset or blank, ``LLM_OPENAI_SHOW_RESPONSES`` is used
instead. ``0``, ``false``, ``off`` and ``no`` disable logging; any other non-blank
value enables it. The setting is read on every request.

Logs go to stderr, redact credentials, summarise binary bodies and never consume
a stream: streamed bodies are either skipped or echoed as the caller reads them.
Logging never changes the transport, so proxy settings from the environment apply.
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

_SECRET_HEADERS = {"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"}
# Body fields holding credentials. Kept narrow on purpose: generic names such as
# "token" or "secret" also appear in logprobs and tool schemas, which must stay visible.
_SECRET_FIELDS = {"apikey", "api_key"}

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
            str(k): "[redacted]" if str(k).lower() in _SECRET_FIELDS else redact(v)
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
        if key.lower() in _SECRET_HEADERS:
            value = "[redacted]"
        _echo(f"    {key}: {value}")


def _echo_body(text: str) -> None:
    _echo("  Body:")
    _echo(textwrap.indent(text, "    "))


def _dumps(value: Any, indent: int | None = None) -> str:
    return json.dumps(shorten(redact(value)), indent=indent, ensure_ascii=False)


def _sse_line(line: str) -> str:
    """Redact the JSON payload of an SSE data line; return other lines unchanged."""
    line = line.rstrip("\r")
    if not line.startswith("data:"):
        return line
    try:
        payload: Any = json.loads(line[5:])
    except ValueError:
        return line
    return f"data: {_dumps(payload)}"


def _echo_content(content: bytes, content_type: str) -> None:
    """Pretty-print JSON, redact SSE events, show text and summarise anything else."""
    if not content:
        return
    if "text/event-stream" in content_type:
        text = content.decode(errors="replace")
        _echo_body("\n".join(_sse_line(line) for line in text.split("\n")).rstrip("\n"))
    elif "json" in content_type:
        try:
            body: Any = json.loads(content)
        except (ValueError, UnicodeError):
            _echo_body(f"[invalid JSON, {len(content)} bytes]")
        else:
            _echo_body(_dumps(body, indent=2))
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
    """Log a body while the caller reads it, without holding back any bytes.

    JSON is collected and logged once complete, so it can be redacted and
    pretty-printed. SSE is redacted line by line as events arrive. Other text is
    echoed as it is decoded.
    """

    def __init__(self, content_type: str) -> None:
        self._content_type = content_type
        self._sse = "text/event-stream" in content_type
        self._json = not self._sse and "json" in content_type
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._collected = bytearray()
        self._pending = ""
        self._started = False
        self._ends_with_newline = True

    def _emit(self, text: str) -> None:
        if not text:
            return
        if not self._started:
            self._started = True
            _echo("  Body:")
        sys.stderr.write(text)
        sys.stderr.flush()
        self._ends_with_newline = text.endswith("\n")

    def _emit_lines(self, lines: list[str]) -> None:
        self._emit("".join(f"    {line}\n" if line else "\n" for line in map(_sse_line, lines)))

    def write(self, chunk: bytes) -> None:
        if self._json:
            self._collected += chunk
            return
        text = self._decoder.decode(chunk)
        if self._sse:
            *lines, self._pending = (self._pending + text).split("\n")
            self._emit_lines(lines)
        else:
            self._emit(text)

    def close(self) -> None:
        if self._json:
            _echo_content(bytes(self._collected), self._content_type)
            return
        text = self._decoder.decode(b"", final=True)
        if self._sse:
            rest = self._pending + text
            self._pending = ""
            if rest:
                self._emit_lines(rest.split("\n"))
        else:
            self._emit(text)
        if self._started and not self._ends_with_newline:
            _echo()


def _is_text(content_type: str) -> bool:
    return "json" in content_type or content_type.startswith("text/")


class _EchoResponse(httpx2.Response):
    """Response whose decoded body is logged while the caller reads it."""

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        echo = _BodyEcho(self.headers.get("content-type", ""))
        try:
            for chunk in super().iter_bytes(chunk_size):
                echo.write(chunk)
                yield chunk
        finally:
            echo.close()

    async def aiter_bytes(self, chunk_size: int | None = None) -> AsyncGenerator[bytes, None]:
        echo = _BodyEcho(self.headers.get("content-type", ""))
        try:
            async for chunk in super().aiter_bytes(chunk_size):
                echo.write(chunk)
                yield chunk
        finally:
            echo.close()


def log_chat_response(response: httpx2.Response) -> None:
    """Log headers, then log the body now if it is read, or as the caller reads it.

    Echoing hooks into the response rather than replacing the transport, which
    would disable httpx's proxy routing from HTTP(S)_PROXY, ALL_PROXY and NO_PROXY.
    """
    if not enabled():
        return
    log_response(response)
    content_type = response.headers.get("content-type", "")
    try:
        content = response.content
    except httpx2.ResponseNotRead:
        if _is_text(content_type):
            # httpx calls response hooks before reading the body. Response has no
            # __slots__, and the subclass only overrides the byte iterators.
            response.__class__ = _EchoResponse
    else:
        _echo_content(content, content_type)


async def async_log_chat_response(response: httpx2.Response) -> None:
    log_chat_response(response)


def chat_http_client() -> httpx2.Client:
    """HTTP client for the OpenAI SDK that logs requests and response bodies."""
    return openai.DefaultHttpxClient(
        event_hooks={"request": [log_request], "response": [log_chat_response]},
    )


def async_chat_http_client() -> httpx2.AsyncClient:
    """Async counterpart of :func:`chat_http_client`."""
    return openai.DefaultAsyncHttpxClient(
        event_hooks={"request": [async_log_request], "response": [async_log_chat_response]},
    )
