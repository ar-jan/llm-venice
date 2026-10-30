"""Reusable sync and async clients for Venice's non-chat endpoints."""

from collections.abc import AsyncGenerator, Generator
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from datetime import datetime
import os
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast

import httpx2
from typing_extensions import Self

from . import _endpoints as ep
from . import _logging
from ._files import async_file_call
from ._http import DEFAULT_TIMEOUT, check_status, request_headers, resolve_key
from .constants import VENICE_API_BASE
from .errors import VeniceConnectionError, VeniceResponseError
from .params import ImageParams, SpeechParams
from .results import (
    AsyncSpeechStream,
    GeneratedImage,
    ImageResult,
    SpeechResult,
    SpeechStream,
)
from .types import ApiKey, Character, CreatedApiKey, ModelInfo, RateLimitLogEntry, RateLimits

ImageInput = str | os.PathLike[str] | bytes | BinaryIO


def _image_file(image: ImageInput) -> tuple[str, bytes] | tuple[str, bytes, str]:
    """Build the multipart file, labelling recognized PNG/JPEG/WebP bytes."""
    if isinstance(image, (str, os.PathLike)):
        path = Path(image)
        filename, content = path.name, path.read_bytes()
    elif isinstance(image, bytes):
        filename, content = "image", image
    else:
        # Caller owns the file, including its current cursor and lifetime.
        filename, content = "image", image.read()
    try:
        format = ep.media_format(None, content, None, image=True)
    except VeniceResponseError:
        # Unrecognized bytes: send as-is and let the API decide.
        return filename, content
    if filename == "image":
        filename = f"image.{'jpg' if format == 'jpeg' else format}"
    return filename, content, f"image/{format}"


class _Configuration:
    def _configure(self, key: str | None, base_url: str, timeout: float | httpx2.Timeout) -> None:
        self._key = key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("Venice client is closed")

    def _prepare(self, path: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        self._ensure_open()
        if self._key is None:
            self._key = resolve_key(None)
        elif not self._key:
            self._key = resolve_key(self._key)
        headers = request_headers(self._key)
        if "json" in kwargs:
            headers["Content-Type"] = "application/json"
        return f"{self._base_url}/{path.lstrip('/')}", {
            **kwargs,
            "headers": headers,
            "timeout": self._timeout,
            "auth": None,
        }


class VeniceClient(_Configuration):
    """Venice API client. Injected HTTP clients are borrowed and never closed.

    Keys resolve on the first request. Close is idempotent; closed wrappers reject
    new requests. No generation method performs catalog lookups or retries.
    """

    def __init__(
        self,
        key: str | None = None,
        *,
        base_url: str = VENICE_API_BASE,
        timeout: float | httpx2.Timeout = DEFAULT_TIMEOUT,
        http_client: httpx2.Client | None = None,
    ) -> None:
        self._configure(key, base_url, timeout)
        self._owned = http_client is None
        self._http = (
            http_client
            if http_client is not None
            else httpx2.Client(
                event_hooks={
                    "request": [_logging.log_request],
                    "response": [_logging.log_response],
                }
            )
        )

    def close(self) -> None:
        """Close this wrapper and its owned transport; leave injected transports open."""
        if not self._closed:
            self._closed = True
            if self._owned:
                self._http.close()

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _send(self, method: str, path: str, action: str, **kwargs: Any) -> httpx2.Response:
        url, options = self._prepare(path, kwargs)
        try:
            response = self._http.request(method, url, **options)
            if self._owned:
                _logging.log_buffered(response)
            check_status(response, action)
            return response
        except httpx2.RequestError as exc:
            raise VeniceConnectionError(f"{action} failed: {exc}") from exc

    def _json_operation(self, name: str, **inputs: Any) -> dict[str, Any]:
        method, path, action, kwargs = ep.json_operation(name, inputs)
        return ep.parse_envelope(self._send(method, path, action, **kwargs), name, action)

    def list_models(self, *, type: str = "all") -> list[ModelInfo]:
        """Fetch catalog resources, skipping malformed entries; empty filtered results are valid."""
        return cast(list[ModelInfo], self._json_operation("models", type=type)["data"])

    def list_characters(
        self,
        *,
        web_enabled: bool | None = None,
        adult: bool | None = None,
    ) -> list[Character]:
        """List characters with optional boolean filters, unwrapping the envelope."""
        return cast(
            list[Character],
            self._json_operation(
                "characters",
                web_enabled=web_enabled,
                adult=adult,
            )["data"],
        )

    def get_rate_limits(self) -> RateLimits:
        """Return this key's access, balance and rate-limit resource."""
        return cast(RateLimits, self._json_operation("rate_limits")["data"])

    def get_rate_limits_log(self) -> list[RateLimitLogEntry]:
        """Return recent rate-limit events with wire-format timestamps."""
        return cast(list[RateLimitLogEntry], self._json_operation("rate_limits_log")["data"])

    def list_api_keys(self) -> list[ApiKey]:
        """List account keys (requires an ADMIN key)."""
        return cast(list[ApiKey], self._json_operation("keys")["data"])

    def create_api_key(
        self,
        *,
        key_type: Literal["ADMIN", "INFERENCE"],
        description: str = "",
        expires_at: datetime | None = None,
        limit_usd: float | None = None,
        limit_vcu: float | None = None,
    ) -> CreatedApiKey:
        """Create a key once. Expiry must be timezone-aware and is sent in UTC."""
        return cast(
            CreatedApiKey,
            self._json_operation(
                "create_key",
                key_type=key_type,
                description=description,
                expires_at=expires_at,
                limit_usd=limit_usd,
                limit_vcu=limit_vcu,
            )["data"],
        )

    def delete_api_key(self, key_id: str) -> None:
        """Delete a key by its resource ID."""
        self._json_operation("delete_key", id=key_id)

    def generate_image(
        self,
        model: str,
        prompt: str,
        params: ImageParams | None = None,
    ) -> ImageResult:
        """Generate bytes and moderation metadata, without saving or fetching models."""
        body = ep.image_body(model, prompt, params)
        response = self._send("POST", "image/generate", "Generating image", json=body)
        return ep.parse_image(response, model, body.get("format"))

    def upscale_image(
        self,
        image: ImageInput,
        *,
        scale: float = 2,
        enhance: bool = False,
        enhance_creativity: float | None = None,
        enhance_prompt: str | None = None,
        replication: float | None = None,
    ) -> GeneratedImage:
        """Upscale a path, bytes or borrowed file read from its current position."""
        self._ensure_open()
        data = ep.UpscaleInput.model_validate(
            {
                "scale": scale,
                "enhance": enhance,
                "enhance_creativity": enhance_creativity,
                "enhance_prompt": enhance_prompt,
                "replication": replication,
            }
        ).model_dump(by_alias=True, exclude_none=True)
        file = _image_file(image)
        return ep.parse_upscale(
            self._send(
                "POST",
                "image/upscale",
                "Upscaling image",
                data=data,
                files={"image": file},
            )
        )

    def generate_speech(
        self,
        model: str,
        text: str,
        params: SpeechParams | None = None,
    ) -> SpeechResult:
        """Buffer speech without saving. An omitted format uses the server default."""
        body = ep.speech_body(model, text, params, streaming=False)
        response = self._send("POST", "audio/speech", "Generating speech", json=body)
        return ep.parse_speech(response, model, body.get("response_format"))

    @contextmanager
    def stream_speech(
        self,
        model: str,
        text: str,
        params: SpeechParams | None = None,
    ) -> Generator[SpeechStream, None, None]:
        """Open single-use speech. Keep consumption inside this context and client lifetime."""
        body = ep.speech_body(model, text, params, streaming=True)
        url, options = self._prepare("audio/speech", {"json": body})
        with ExitStack() as stack:
            # Map transport errors only while opening; body errors are mapped by
            # SpeechStream, and errors from the caller's block must pass through.
            try:
                response = stack.enter_context(self._http.stream("POST", url, **options))
                if not response.is_success:
                    response.read()
                    if self._owned:
                        _logging.log_buffered(response)
                    check_status(response, "Generating speech")
            except httpx2.RequestError as exc:
                raise VeniceConnectionError(f"Generating speech failed: {exc}") from exc
            format = ep.media_format(
                response.headers.get("content-type"),
                b"",
                body.get("response_format"),
                image=False,
            )
            stream = SpeechStream(response, format)
            stack.callback(stream.close)
            yield stream


class AsyncVeniceClient(_Configuration):
    """Venice API client. Injected HTTP clients are borrowed and never closed.

    Keys resolve on the first request. Close is idempotent; closed wrappers reject
    new requests. No generation method performs catalog lookups or retries.
    """

    def __init__(
        self,
        key: str | None = None,
        *,
        base_url: str = VENICE_API_BASE,
        timeout: float | httpx2.Timeout = DEFAULT_TIMEOUT,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._configure(key, base_url, timeout)
        self._owned = http_client is None
        self._http = (
            http_client
            if http_client is not None
            else httpx2.AsyncClient(
                event_hooks={
                    "request": [_logging.async_log_request],
                    "response": [_logging.async_log_response],
                }
            )
        )

    async def aclose(self) -> None:
        """Close this wrapper and its owned transport; leave injected transports open."""
        if not self._closed:
            self._closed = True
            if self._owned:
                await self._http.aclose()

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _send(self, method: str, path: str, action: str, **kwargs: Any) -> httpx2.Response:
        url, options = self._prepare(path, kwargs)
        try:
            response = await self._http.request(method, url, **options)
            if self._owned:
                _logging.log_buffered(response)
            check_status(response, action)
            return response
        except httpx2.RequestError as exc:
            raise VeniceConnectionError(f"{action} failed: {exc}") from exc

    async def _json_operation(self, name: str, **inputs: Any) -> dict[str, Any]:
        method, path, action, kwargs = ep.json_operation(name, inputs)
        return ep.parse_envelope(await self._send(method, path, action, **kwargs), name, action)

    async def list_models(self, *, type: str = "all") -> list[ModelInfo]:
        """Fetch catalog resources, skipping malformed entries; empty filtered results are valid."""
        return cast(list[ModelInfo], (await self._json_operation("models", type=type))["data"])

    async def list_characters(
        self,
        *,
        web_enabled: bool | None = None,
        adult: bool | None = None,
    ) -> list[Character]:
        """List characters with optional boolean filters, unwrapping the envelope."""
        return cast(
            list[Character],
            (
                await self._json_operation(
                    "characters",
                    web_enabled=web_enabled,
                    adult=adult,
                )
            )["data"],
        )

    async def get_rate_limits(self) -> RateLimits:
        """Return this key's access, balance and rate-limit resource."""
        return cast(RateLimits, (await self._json_operation("rate_limits"))["data"])

    async def get_rate_limits_log(self) -> list[RateLimitLogEntry]:
        """Return recent rate-limit events with wire-format timestamps."""
        return cast(
            list[RateLimitLogEntry], (await self._json_operation("rate_limits_log"))["data"]
        )

    async def list_api_keys(self) -> list[ApiKey]:
        """List account keys (requires an ADMIN key)."""
        return cast(list[ApiKey], (await self._json_operation("keys"))["data"])

    async def create_api_key(
        self,
        *,
        key_type: Literal["ADMIN", "INFERENCE"],
        description: str = "",
        expires_at: datetime | None = None,
        limit_usd: float | None = None,
        limit_vcu: float | None = None,
    ) -> CreatedApiKey:
        """Create a key once. Expiry must be timezone-aware and is sent in UTC."""
        return cast(
            CreatedApiKey,
            (
                await self._json_operation(
                    "create_key",
                    key_type=key_type,
                    description=description,
                    expires_at=expires_at,
                    limit_usd=limit_usd,
                    limit_vcu=limit_vcu,
                )
            )["data"],
        )

    async def delete_api_key(self, key_id: str) -> None:
        """Delete a key by its resource ID."""
        await self._json_operation("delete_key", id=key_id)

    async def generate_image(
        self,
        model: str,
        prompt: str,
        params: ImageParams | None = None,
    ) -> ImageResult:
        """Generate bytes and moderation metadata, without saving or fetching models."""
        body = ep.image_body(model, prompt, params)
        response = await self._send("POST", "image/generate", "Generating image", json=body)
        return ep.parse_image(response, model, body.get("format"))

    async def upscale_image(
        self,
        image: ImageInput,
        *,
        scale: float = 2,
        enhance: bool = False,
        enhance_creativity: float | None = None,
        enhance_prompt: str | None = None,
        replication: float | None = None,
    ) -> GeneratedImage:
        """Upscale a path, bytes or borrowed file read from its current position."""
        self._ensure_open()
        data = ep.UpscaleInput.model_validate(
            {
                "scale": scale,
                "enhance": enhance,
                "enhance_creativity": enhance_creativity,
                "enhance_prompt": enhance_prompt,
                "replication": replication,
            }
        ).model_dump(by_alias=True, exclude_none=True)
        file = await async_file_call(_image_file, image)
        return ep.parse_upscale(
            await self._send(
                "POST",
                "image/upscale",
                "Upscaling image",
                data=data,
                files={"image": file},
            )
        )

    async def generate_speech(
        self,
        model: str,
        text: str,
        params: SpeechParams | None = None,
    ) -> SpeechResult:
        """Buffer speech without saving. An omitted format uses the server default."""
        body = ep.speech_body(model, text, params, streaming=False)
        response = await self._send("POST", "audio/speech", "Generating speech", json=body)
        return ep.parse_speech(response, model, body.get("response_format"))

    @asynccontextmanager
    async def stream_speech(
        self,
        model: str,
        text: str,
        params: SpeechParams | None = None,
    ) -> AsyncGenerator[AsyncSpeechStream, None]:
        """Open single-use speech. Keep consumption inside this context and client lifetime."""
        body = ep.speech_body(model, text, params, streaming=True)
        url, options = self._prepare("audio/speech", {"json": body})
        async with AsyncExitStack() as stack:
            # Map transport errors only while opening; body errors are mapped by
            # AsyncSpeechStream, and errors from the caller's block must pass through.
            try:
                response = await stack.enter_async_context(
                    self._http.stream("POST", url, **options)
                )
                if not response.is_success:
                    await response.aread()
                    if self._owned:
                        _logging.log_buffered(response)
                    check_status(response, "Generating speech")
            except httpx2.RequestError as exc:
                raise VeniceConnectionError(f"Generating speech failed: {exc}") from exc
            format = ep.media_format(
                response.headers.get("content-type"),
                b"",
                body.get("response_format"),
                image=False,
            )
            stream = AsyncSpeechStream(response, format)
            stack.push_async_callback(stream.aclose)
            yield stream
