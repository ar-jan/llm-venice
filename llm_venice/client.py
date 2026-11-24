"""Reusable sync and async clients for Venice's non-chat endpoints."""

from collections.abc import AsyncGenerator, Generator, Sequence
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from datetime import datetime
import os
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast, overload

import httpx2
from typing_extensions import Self, Unpack

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
    ImageEditResult,
    ImageResult,
    SpeechResult,
    SpeechStream,
)
from .types import (
    ApiKey,
    ApiKeyChanges,
    ApiKeyLimitPeriod,
    ApiKeyModelPrivacy,
    AugmentScrapeResponse,
    AugmentSearchProvider,
    AugmentTextFormat,
    AugmentTextParserResponse,
    CharacterSortBy,
    CharacterSortOrder,
    AugmentSearchResponse,
    Character,
    CharacterReviewPage,
    CreatedApiKey,
    EmbeddingEncodingFormat,
    EmbeddingResponse,
    ImageEditOptions,
    ModelInfo,
    RateLimitLogEntry,
    RateLimits,
)

FileInput = str | os.PathLike[str] | bytes | BinaryIO
ImageInput = FileInput


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


def _document_file(file: FileInput, filename: str | None) -> tuple[str, bytes, str]:
    """Build the multipart file, named so that Venice can tell the document's format."""
    if isinstance(file, (str, os.PathLike)):
        path = Path(file)
        name, content = path.name, path.read_bytes()
    elif isinstance(file, bytes):
        name, content = "document", file
    else:
        # Caller owns the file, including its current cursor and lifetime.
        attribute: object = getattr(file, "name", None)
        name = Path(attribute).name if isinstance(attribute, str) and attribute else "document"
        # A text-mode file would return str despite the annotation.
        read = cast(object, file.read())
        if not isinstance(read, bytes):
            raise TypeError("file must be opened in binary mode")
        content = read
    return ep.document_part(filename or name, content)


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

    def _send(
        self, method: str, path: str, action: str, *, policy_422: bool = False, **kwargs: Any
    ) -> httpx2.Response:
        url, options = self._prepare(path, kwargs)
        try:
            response = self._http.request(method, url, **options)
            if self._owned:
                _logging.log_buffered(response)
            check_status(response, action, policy_422=policy_422)
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
        pro: bool | None = None,
        categories: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        model_ids: Sequence[str] | None = None,
        search: str | None = None,
        sort_by: CharacterSortBy | None = None,
        sort_order: CharacterSortOrder | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Character]:
        """List one page of characters (Venice default 50, max 100 per page).

        Unset filters are not sent. ``categories``, ``tags`` and ``model_ids`` take
        up to 20 values each. Page with ``offset``; a page shorter than ``limit`` is
        the last one.
        """
        return cast(
            list[Character],
            self._json_operation(
                "characters",
                web_enabled=web_enabled,
                adult=adult,
                pro=pro,
                categories=categories,
                tags=tags,
                model_ids=model_ids,
                search=search,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset,
            )["data"],
        )

    def get_character(self, slug: str) -> Character:
        """Return one public character by its slug."""
        return cast(Character, self._json_operation("character", slug=slug)["data"])

    def list_character_reviews(
        self, slug: str, *, page: int | None = None, page_size: int | None = None
    ) -> CharacterReviewPage:
        """Return one page of a character's public reviews (Venice default 20, max 100).

        Pages start at 1. The result holds the reviews in ``data``, the page totals in
        ``pagination`` and the rating summary in ``summary``.
        """
        return cast(
            CharacterReviewPage,
            self._json_operation("character_reviews", slug=slug, page=page, page_size=page_size),
        )

    def list_image_styles(self) -> list[str]:
        """Return the style names that ``ImageParams.style_preset`` accepts."""
        return cast(list[str], self._json_operation("image_styles")["data"])

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
        limit_diem: float | None = None,
        limit_vcu: float | None = None,
        limit_period: ApiKeyLimitPeriod | None = None,
        model_privacy: ApiKeyModelPrivacy | None = None,
        ip_allowlist: Sequence[str] | None = None,
    ) -> CreatedApiKey:
        """Create a key and return its one-time secret; unset options are not sent.

        Expiry must be timezone-aware and is sent in UTC. ``limit_vcu`` is deprecated
        by Venice in favour of ``limit_diem``. An empty ``ip_allowlist`` allows any IP.
        """
        return cast(
            CreatedApiKey,
            self._json_operation(
                "create_key",
                key_type=key_type,
                description=description,
                expires_at=expires_at,
                limit_usd=limit_usd,
                limit_diem=limit_diem,
                limit_vcu=limit_vcu,
                limit_period=limit_period,
                model_privacy=model_privacy,
                ip_allowlist=ip_allowlist,
            )["data"],
        )

    def get_api_key(self, key_id: str) -> ApiKey:
        """Return one key's details by its resource ID (requires an ADMIN key)."""
        return cast(ApiKey, self._json_operation("get_key", id=key_id)["data"])

    def update_api_key(self, key_id: str, **changes: Unpack[ApiKeyChanges]) -> ApiKey:
        """Change a key's settings and return the key; settings left out are kept.

        None removes the expiry or a limit, and an empty ``ip_allowlist`` allows any
        IP address. Requires an ADMIN key.
        """
        inputs: dict[str, Any] = {**changes, "id": key_id}
        return cast(ApiKey, self._json_operation("update_key", **inputs)["data"])

    def delete_api_key(self, key_id: str) -> None:
        """Delete a key by its resource ID."""
        self._json_operation("delete_key", id=key_id)

    def augment_search(
        self,
        query: str,
        *,
        limit: int | None = None,
        provider: AugmentSearchProvider | None = None,
    ) -> AugmentSearchResponse:
        """Search the web without a model. Omitted options use Venice's defaults."""
        return cast(
            AugmentSearchResponse,
            self._json_operation("augment_search", query=query, limit=limit, provider=provider),
        )

    def augment_scrape(self, url: str) -> AugmentScrapeResponse:
        """Fetch a web page through Venice and return its content as markdown.

        Pages Venice cannot fetch fail with a VeniceAPIError: 400 for sites that block
        automated access, 502 when the page could not be scraped.
        """
        return cast(AugmentScrapeResponse, self._json_operation("augment_scrape", url=url))

    @overload
    def augment_parse_text(
        self,
        file: FileInput,
        *,
        filename: str | None = None,
        response_format: Literal["json"] = "json",
    ) -> AugmentTextParserResponse: ...

    @overload
    def augment_parse_text(
        self, file: FileInput, *, filename: str | None = None, response_format: Literal["text"]
    ) -> str: ...

    @overload
    def augment_parse_text(
        self,
        file: FileInput,
        *,
        filename: str | None = None,
        response_format: AugmentTextFormat = "json",
    ) -> AugmentTextParserResponse | str: ...

    def augment_parse_text(
        self,
        file: FileInput,
        *,
        filename: str | None = None,
        response_format: AugmentTextFormat = "json",
    ) -> AugmentTextParserResponse | str:
        """Extract the text of a document: a path, bytes or borrowed binary file.

        Venice parses PDF, DOCX, PPTX, XLSX, EPUB, plain text, Markdown, CSV, JSON and
        most source code, up to 25 MB, and keeps no copy. It tells the format from the
        filename's extension: a path or named file supplies one, and ``filename``
        overrides it. Unnamed bytes are labelled PDF or plain text from their content,
        so pass ``filename`` for other formats. ``response_format="json"`` (the
        default) returns the text with its token count; ``"text"`` returns a string.
        """
        self._ensure_open()
        options = ep.TextParserInput(filename=filename, response_format=response_format)
        part = _document_file(file, options.filename)
        response = self._send(
            "POST",
            "augment/text-parser",
            ep.TEXT_PARSER_ACTION,
            data={"response_format": options.response_format},
            files={"file": part},
        )
        return ep.parse_text_parser(response, options.response_format)

    def create_embeddings(
        self,
        model: str,
        input: str | Sequence[str],
        *,
        dimensions: int | None = None,
        encoding_format: EmbeddingEncodingFormat | None = None,
    ) -> EmbeddingResponse:
        """Embed a string, or each string of a list, and return the embeddings in input order.

        A list takes up to 2048 strings; token-ID arrays are rejected. Omitted options
        use Venice's defaults: the model's native size and lists of floats.
        ``dimensions`` shortens the vectors on models that support it.
        """
        inputs = ep.EmbeddingInput.model_validate(
            {
                "model": model,
                "input": input,
                "dimensions": dimensions,
                "encoding_format": encoding_format,
            }
        )
        response = self._send("POST", "embeddings", ep.EMBEDDINGS_ACTION, json=inputs.body())
        return ep.parse_embeddings(response, inputs.count())

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
        creativity: float | None = None,
    ) -> GeneratedImage:
        """Upscale a path, bytes or borrowed file read from its current position."""
        self._ensure_open()
        data = ep.UpscaleInput.model_validate(
            {"scale": scale, "creativity": creativity}
        ).model_dump(by_alias=True, exclude_none=True)
        file = _image_file(image)
        return ep.parse_upscale(
            self._send(
                "POST",
                "image/upscale",
                "Upscaling image",
                # Every upscale 422 is a moderation refusal; most carry only `error`.
                policy_422=True,
                data=data,
                files={"image": file},
            )
        )

    def remove_image_background(
        self, image: ImageInput, *, anon_user_id: str | None = None
    ) -> GeneratedImage:
        """Return the image as a PNG with its background removed (made transparent).

        ``image`` is a path, bytes or borrowed binary file, uploaded as a file, or a
        str starting with http:// or https://, which Venice fetches. Pass a Path for
        a local file whose name starts that way.
        """
        self._ensure_open()
        url = image if ep.is_image_url(image) else None
        body = ep.BackgroundRemoveInput(image_url=url, anon_user_id=anon_user_id)
        fields = body.model_dump(exclude_none=True)
        # A URL goes as JSON; a local image as a multipart upload.
        request: dict[str, Any] = (
            {"json": fields} if url else {"data": fields, "files": {"image": _image_file(image)}}
        )
        response = self._send("POST", "image/background-remove", ep.BACKGROUND_ACTION, **request)
        return ep.parse_binary_image(response, ep.BACKGROUND_ACTION)

    def edit_image(
        self,
        image: ImageInput,
        prompt: str,
        *,
        model: str | None = None,
        **options: Unpack[ImageEditOptions],
    ) -> ImageEditResult:
        """Edit an image as ``prompt`` directs, returning the image and response facts.

        ``image`` is a path, bytes or borrowed binary file, uploaded as a file, or a
        str starting with http:// or https://, which Venice fetches. ``model`` is an
        edit model ID; omitted, Venice uses its default edit model. Options left out
        use Venice's defaults (see ``ImageEditOptions``). No catalog checks are made.
        """
        self._ensure_open()
        url = image if ep.is_image_url(image) else None
        inputs = ep.SingleImageEditInput.model_validate(
            {**options, "prompt": prompt, "model": model, "image_url": url}
        )
        # A URL goes as JSON; a local image as a multipart upload.
        request: dict[str, Any] = (
            {"json": inputs.body()}
            if url
            else {"data": ep.form_fields(inputs.body()), "files": {"image": _image_file(image)}}
        )
        response = self._send("POST", "image/edit", ep.IMAGE_EDIT_ACTION, **request)
        return ep.parse_image_edit(response, ep.IMAGE_EDIT_ACTION)

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
                content_type = response.headers.get("content-type")
                # Buffer error bodies, including a 200 that carries something other than audio.
                if not response.is_success or not ep.is_speech_content_type(content_type):
                    response.read()
                    if self._owned:
                        _logging.log_buffered(response)
                    check_status(response, "Generating speech")
                    ep.check_speech_content_type(content_type, response.content)
            except httpx2.RequestError as exc:
                raise VeniceConnectionError(f"Generating speech failed: {exc}") from exc
            format = ep.media_format(
                content_type,
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

    async def _send(
        self, method: str, path: str, action: str, *, policy_422: bool = False, **kwargs: Any
    ) -> httpx2.Response:
        url, options = self._prepare(path, kwargs)
        try:
            response = await self._http.request(method, url, **options)
            if self._owned:
                _logging.log_buffered(response)
            check_status(response, action, policy_422=policy_422)
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
        pro: bool | None = None,
        categories: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        model_ids: Sequence[str] | None = None,
        search: str | None = None,
        sort_by: CharacterSortBy | None = None,
        sort_order: CharacterSortOrder | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Character]:
        """List one page of characters (Venice default 50, max 100 per page).

        Unset filters are not sent. ``categories``, ``tags`` and ``model_ids`` take
        up to 20 values each. Page with ``offset``; a page shorter than ``limit`` is
        the last one.
        """
        return cast(
            list[Character],
            (
                await self._json_operation(
                    "characters",
                    web_enabled=web_enabled,
                    adult=adult,
                    pro=pro,
                    categories=categories,
                    tags=tags,
                    model_ids=model_ids,
                    search=search,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    limit=limit,
                    offset=offset,
                )
            )["data"],
        )

    async def get_character(self, slug: str) -> Character:
        """Return one public character by its slug."""
        return cast(Character, (await self._json_operation("character", slug=slug))["data"])

    async def list_character_reviews(
        self, slug: str, *, page: int | None = None, page_size: int | None = None
    ) -> CharacterReviewPage:
        """Return one page of a character's public reviews (Venice default 20, max 100).

        Pages start at 1. The result holds the reviews in ``data``, the page totals in
        ``pagination`` and the rating summary in ``summary``.
        """
        return cast(
            CharacterReviewPage,
            await self._json_operation(
                "character_reviews", slug=slug, page=page, page_size=page_size
            ),
        )

    async def list_image_styles(self) -> list[str]:
        """Return the style names that ``ImageParams.style_preset`` accepts."""
        return cast(list[str], (await self._json_operation("image_styles"))["data"])

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
        limit_diem: float | None = None,
        limit_vcu: float | None = None,
        limit_period: ApiKeyLimitPeriod | None = None,
        model_privacy: ApiKeyModelPrivacy | None = None,
        ip_allowlist: Sequence[str] | None = None,
    ) -> CreatedApiKey:
        """Create a key and return its one-time secret; unset options are not sent.

        Expiry must be timezone-aware and is sent in UTC. ``limit_vcu`` is deprecated
        by Venice in favour of ``limit_diem``. An empty ``ip_allowlist`` allows any IP.
        """
        return cast(
            CreatedApiKey,
            (
                await self._json_operation(
                    "create_key",
                    key_type=key_type,
                    description=description,
                    expires_at=expires_at,
                    limit_usd=limit_usd,
                    limit_diem=limit_diem,
                    limit_vcu=limit_vcu,
                    limit_period=limit_period,
                    model_privacy=model_privacy,
                    ip_allowlist=ip_allowlist,
                )
            )["data"],
        )

    async def get_api_key(self, key_id: str) -> ApiKey:
        """Return one key's details by its resource ID (requires an ADMIN key)."""
        return cast(ApiKey, (await self._json_operation("get_key", id=key_id))["data"])

    async def update_api_key(self, key_id: str, **changes: Unpack[ApiKeyChanges]) -> ApiKey:
        """Change a key's settings and return the key; settings left out are kept.

        None removes the expiry or a limit, and an empty ``ip_allowlist`` allows any
        IP address. Requires an ADMIN key.
        """
        inputs: dict[str, Any] = {**changes, "id": key_id}
        return cast(ApiKey, (await self._json_operation("update_key", **inputs))["data"])

    async def delete_api_key(self, key_id: str) -> None:
        """Delete a key by its resource ID."""
        await self._json_operation("delete_key", id=key_id)

    async def augment_search(
        self,
        query: str,
        *,
        limit: int | None = None,
        provider: AugmentSearchProvider | None = None,
    ) -> AugmentSearchResponse:
        """Search the web without a model. Omitted options use Venice's defaults."""
        return cast(
            AugmentSearchResponse,
            await self._json_operation(
                "augment_search", query=query, limit=limit, provider=provider
            ),
        )

    async def augment_scrape(self, url: str) -> AugmentScrapeResponse:
        """Fetch a web page through Venice and return its content as markdown.

        Pages Venice cannot fetch fail with a VeniceAPIError: 400 for sites that block
        automated access, 502 when the page could not be scraped.
        """
        return cast(AugmentScrapeResponse, await self._json_operation("augment_scrape", url=url))

    @overload
    async def augment_parse_text(
        self,
        file: FileInput,
        *,
        filename: str | None = None,
        response_format: Literal["json"] = "json",
    ) -> AugmentTextParserResponse: ...

    @overload
    async def augment_parse_text(
        self, file: FileInput, *, filename: str | None = None, response_format: Literal["text"]
    ) -> str: ...

    @overload
    async def augment_parse_text(
        self,
        file: FileInput,
        *,
        filename: str | None = None,
        response_format: AugmentTextFormat = "json",
    ) -> AugmentTextParserResponse | str: ...

    async def augment_parse_text(
        self,
        file: FileInput,
        *,
        filename: str | None = None,
        response_format: AugmentTextFormat = "json",
    ) -> AugmentTextParserResponse | str:
        """Extract the text of a document: a path, bytes or borrowed binary file.

        Venice parses PDF, DOCX, PPTX, XLSX, EPUB, plain text, Markdown, CSV, JSON and
        most source code, up to 25 MB, and keeps no copy. It tells the format from the
        filename's extension: a path or named file supplies one, and ``filename``
        overrides it. Unnamed bytes are labelled PDF or plain text from their content,
        so pass ``filename`` for other formats. ``response_format="json"`` (the
        default) returns the text with its token count; ``"text"`` returns a string.
        """
        self._ensure_open()
        options = ep.TextParserInput(filename=filename, response_format=response_format)
        part = await async_file_call(_document_file, file, options.filename)
        response = await self._send(
            "POST",
            "augment/text-parser",
            ep.TEXT_PARSER_ACTION,
            data={"response_format": options.response_format},
            files={"file": part},
        )
        return ep.parse_text_parser(response, options.response_format)

    async def create_embeddings(
        self,
        model: str,
        input: str | Sequence[str],
        *,
        dimensions: int | None = None,
        encoding_format: EmbeddingEncodingFormat | None = None,
    ) -> EmbeddingResponse:
        """Embed a string, or each string of a list, and return the embeddings in input order.

        A list takes up to 2048 strings; token-ID arrays are rejected. Omitted options
        use Venice's defaults: the model's native size and lists of floats.
        ``dimensions`` shortens the vectors on models that support it.
        """
        inputs = ep.EmbeddingInput.model_validate(
            {
                "model": model,
                "input": input,
                "dimensions": dimensions,
                "encoding_format": encoding_format,
            }
        )
        response = await self._send("POST", "embeddings", ep.EMBEDDINGS_ACTION, json=inputs.body())
        return ep.parse_embeddings(response, inputs.count())

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
        creativity: float | None = None,
    ) -> GeneratedImage:
        """Upscale a path, bytes or borrowed file read from its current position."""
        self._ensure_open()
        data = ep.UpscaleInput.model_validate(
            {"scale": scale, "creativity": creativity}
        ).model_dump(by_alias=True, exclude_none=True)
        file = await async_file_call(_image_file, image)
        return ep.parse_upscale(
            await self._send(
                "POST",
                "image/upscale",
                "Upscaling image",
                # Every upscale 422 is a moderation refusal; most carry only `error`.
                policy_422=True,
                data=data,
                files={"image": file},
            )
        )

    async def remove_image_background(
        self, image: ImageInput, *, anon_user_id: str | None = None
    ) -> GeneratedImage:
        """Return the image as a PNG with its background removed (made transparent).

        ``image`` is a path, bytes or borrowed binary file, uploaded as a file, or a
        str starting with http:// or https://, which Venice fetches. Pass a Path for
        a local file whose name starts that way.
        """
        self._ensure_open()
        url = image if ep.is_image_url(image) else None
        body = ep.BackgroundRemoveInput(image_url=url, anon_user_id=anon_user_id)
        fields = body.model_dump(exclude_none=True)
        # A URL goes as JSON; a local image as a multipart upload.
        request: dict[str, Any] = {"json": fields}
        if not url:
            file = await async_file_call(_image_file, image)
            request = {"data": fields, "files": {"image": file}}
        response = await self._send(
            "POST", "image/background-remove", ep.BACKGROUND_ACTION, **request
        )
        return ep.parse_binary_image(response, ep.BACKGROUND_ACTION)

    async def edit_image(
        self,
        image: ImageInput,
        prompt: str,
        *,
        model: str | None = None,
        **options: Unpack[ImageEditOptions],
    ) -> ImageEditResult:
        """Edit an image as ``prompt`` directs, returning the image and response facts.

        ``image`` is a path, bytes or borrowed binary file, uploaded as a file, or a
        str starting with http:// or https://, which Venice fetches. ``model`` is an
        edit model ID; omitted, Venice uses its default edit model. Options left out
        use Venice's defaults (see ``ImageEditOptions``). No catalog checks are made.
        """
        self._ensure_open()
        url = image if ep.is_image_url(image) else None
        inputs = ep.SingleImageEditInput.model_validate(
            {**options, "prompt": prompt, "model": model, "image_url": url}
        )
        # A URL goes as JSON; a local image as a multipart upload.
        request: dict[str, Any] = {"json": inputs.body()}
        if not url:
            file = await async_file_call(_image_file, image)
            request = {"data": ep.form_fields(inputs.body()), "files": {"image": file}}
        response = await self._send("POST", "image/edit", ep.IMAGE_EDIT_ACTION, **request)
        return ep.parse_image_edit(response, ep.IMAGE_EDIT_ACTION)

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
                content_type = response.headers.get("content-type")
                # Buffer error bodies, including a 200 that carries something other than audio.
                if not response.is_success or not ep.is_speech_content_type(content_type):
                    await response.aread()
                    if self._owned:
                        _logging.log_buffered(response)
                    check_status(response, "Generating speech")
                    ep.check_speech_content_type(content_type, response.content)
            except httpx2.RequestError as exc:
                raise VeniceConnectionError(f"Generating speech failed: {exc}") from exc
            format = ep.media_format(
                content_type,
                b"",
                body.get("response_format"),
                image=False,
            )
            stream = AsyncSpeechStream(response, format)
            stack.push_async_callback(stream.aclose)
            yield stream
