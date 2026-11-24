"""Pure request builders and response parsers shared by both clients."""

import base64
import binascii
import ipaddress
import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated, Any, Literal, TypeVar, cast
from urllib.parse import quote, unquote, urlsplit

import httpx2
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Self, get_type_hints

from .errors import VeniceResponseError
from .params import ImageParams, SpeechParams, copy_params, parameter_body
from .results import NO_AUDIO, GeneratedImage, ImageResult, ModelDeprecation, SpeechResult
from .types import (
    AugmentSearchProvider,
    ApiKeyLimitPeriod,
    ApiKeyModelPrivacy,
    CharacterSortBy,
    CharacterSortOrder,
    EmbeddingEncodingFormat,
    EmbeddingResponse,
    ModelInfo,
)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str = Field(min_length=1)


class _ImageInput(_Input):
    prompt: str = Field(min_length=1, max_length=7500)


class _SpeechInput(_Input):
    input: str = Field(min_length=1, max_length=4096)


class UpscaleInput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, allow_inf_nan=False)
    scale: float = Field(default=2.0, ge=2, le=4)
    # The endpoint clamps to 0-0.02; reject instead so values are not silently changed.
    creativity: float | None = Field(default=None, ge=0, le=0.02)


def _ip_or_subnet(value: str) -> str:
    """Check the syntax only; Venice decides which addresses are globally reachable."""
    try:
        ipaddress.ip_network(value, strict=False)
    except ValueError:
        raise ValueError(f"{value!r} is not an IP address or CIDR subnet") from None
    return value


IpAllowlistEntry = Annotated[str, Field(max_length=64), AfterValidator(_ip_or_subnet)]
KeyLimit = Annotated[float, Field(ge=0, le=9_999_999_999)]


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _KeySettings(BaseModel):
    """Settings shared by key creation and update; None leaves a field out."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    expires_at: datetime | None = None
    limit_usd: KeyLimit | None = None
    limit_diem: KeyLimit | None = None
    # Deprecated by Venice in favour of Diem; still accepted.
    limit_vcu: KeyLimit | None = None
    limit_period: ApiKeyLimitPeriod | None = None
    model_privacy: ApiKeyModelPrivacy | None = None
    ip_allowlist: list[IpAllowlistEntry] | None = Field(default=None, max_length=100)

    @field_validator("expires_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("expires_at must be timezone-aware")
        return value

    def _limits(self, names: set[str]) -> dict[str, float | None]:
        return {
            wire: getattr(self, name)
            for name, wire in (("limit_usd", "usd"), ("limit_diem", "diem"), ("limit_vcu", "vcu"))
            if name in names
        }

    def _settings(self, names: set[str]) -> dict[str, Any]:
        """The wire fields for the settings in ``names``, with None sent as null."""
        body: dict[str, Any] = {}
        if "expires_at" in names:
            body["expiresAt"] = _utc(self.expires_at) if self.expires_at else None
        limits = self._limits(names)
        if limits:
            body["consumptionLimit"] = limits
        for name, wire in (
            ("limit_period", "limitPeriod"),
            ("model_privacy", "modelPrivacy"),
            ("ip_allowlist", "ipAllowlist"),
        ):
            if name in names:
                body[wire] = getattr(self, name)
        return body


class KeyInput(_KeySettings):
    key_type: Literal["ADMIN", "INFERENCE"]
    description: str = Field(default="", max_length=64)

    def body(self) -> dict[str, Any]:
        """Venice requires the type and description; unset optional fields are omitted."""
        given = {name for name in _KeySettings.model_fields if getattr(self, name) is not None}
        return {
            "apiKeyType": self.key_type,
            "description": self.description,
            **self._settings(given),
        }


# Venice has no null for these, so None cannot clear them in an update.
_NOT_CLEARABLE = ("description", "limit_period", "model_privacy", "ip_allowlist")


class KeyUpdateInput(_KeySettings):
    """Only the settings given are sent; None clears the expiry or a limit."""

    id: str = Field(min_length=1)
    description: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _check_changes(self) -> Self:
        if not self.model_fields_set - {"id"}:
            raise ValueError("no API key settings to update")
        for name in _NOT_CLEARABLE:
            if name in self.model_fields_set and getattr(self, name) is None:
                hint = " (an empty list allows any IP)" if name == "ip_allowlist" else ""
                raise ValueError(f"{name} cannot be None; leave it out to keep it{hint}")
        return self

    def body(self) -> dict[str, Any]:
        changed = self.model_fields_set - {"id"}
        body: dict[str, Any] = {"id": self.id}
        if "description" in changed:
            body["description"] = self.description
        return {**body, **self._settings(changed)}


class KeyIdInput(BaseModel):
    id: str = Field(min_length=1)


class AugmentSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    query: str = Field(min_length=1, max_length=400)
    limit: int | None = Field(default=None, ge=1, le=20)
    provider: AugmentSearchProvider | None = None

    def body(self) -> dict[str, Any]:
        """Omit unset options so Venice applies its defaults (10 results, Brave)."""
        body: dict[str, Any] = {"query": self.query}
        if self.limit is not None:
            body["limit"] = self.limit
        if self.provider is not None:
            body["search_provider"] = self.provider
        return body


EMBEDDINGS_ACTION = "Creating embeddings"
EMBEDDING_INPUT_LIMIT = 2048

_EmbeddingText = Annotated[str, Field(strict=True)]


def _is_token_ids(value: Any) -> bool:
    """Whether ``value`` looks like token IDs (ints, or lists of ints) rather than text."""
    if not isinstance(value, (list, tuple)) or not value:
        return False
    items = cast(list[Any] | tuple[Any, ...], value)
    return all(type(item) is int or isinstance(item, (list, tuple)) for item in items)


class EmbeddingInput(_Input):
    # Venice embeds a nonempty string, or each string of a list in one request.
    input: (
        Annotated[str, Field(strict=True, min_length=1)]
        | Annotated[list[_EmbeddingText], Field(min_length=1, max_length=EMBEDDING_INPUT_LIMIT)]
    )
    dimensions: int | None = Field(default=None, ge=1, strict=True)
    encoding_format: EmbeddingEncodingFormat | None = None

    @field_validator("input", mode="before")
    @classmethod
    def _text_only(cls, value: Any) -> Any:
        # Venice answers token arrays with a 400; say so instead of a union error.
        if _is_token_ids(value):
            raise ValueError("token arrays are not supported; pass a string or a list of strings")
        return value

    def body(self) -> dict[str, Any]:
        """Omit unset options so Venice applies its defaults (native size, floats)."""
        body: dict[str, Any] = {"model": self.model, "input": self.input}
        if self.dimensions is not None:
            body["dimensions"] = self.dimensions
        if self.encoding_format is not None:
            body["encoding_format"] = self.encoding_format
        return body

    def count(self) -> int:
        return 1 if isinstance(self.input, str) else len(self.input)


def parse_embeddings(response: httpx2.Response, count: int) -> EmbeddingResponse:
    """Check for exactly one embedding per input, and return them in input order."""
    action = EMBEDDINGS_ACTION
    obj = resource(json_object(response, action), EmbeddingResponse, action)
    data = obj["data"]
    if len(data) != count:
        raise VeniceResponseError(
            action, f"expected {count} embedding(s), one per input, but received {len(data)}"
        )
    if sorted(item["index"] for item in data) != list(range(count)):
        raise VeniceResponseError(action, "embedding indexes do not match the inputs")
    # Venice lists them in input order already; sorting makes that a guarantee.
    return {**obj, "data": sorted(data, key=lambda item: item["index"])}


def _web_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError("url must be an absolute http or https URL")
    return value


class AugmentScrapeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Sent as given; Venice fetches the page, so only the form is checked here.
    url: Annotated[str, Field(strict=True), AfterValidator(_web_url)]


CHARACTER_PAGE_LIMIT = 100


_Category = Annotated[str, Field(min_length=1, max_length=100)]
_CharacterModelId = Annotated[str, Field(min_length=1, max_length=200)]


class CharacterFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    web_enabled: bool | None = None
    adult: bool | None = None
    pro: bool | None = None
    # Each list is sent as a repeated query parameter, matching any listed value.
    categories: list[_Category] | None = Field(default=None, max_length=20)
    tags: list[_Category] | None = Field(default=None, max_length=20)
    model_ids: list[_CharacterModelId] | None = Field(default=None, max_length=20)
    search: str | None = Field(default=None, min_length=1, max_length=200)
    sort_by: CharacterSortBy | None = None
    sort_order: CharacterSortOrder | None = None
    # Venice pages with limit (default 50) and offset, and reports no total.
    limit: int | None = Field(default=None, ge=1, le=CHARACTER_PAGE_LIMIT, strict=True)
    offset: int | None = Field(default=None, ge=0, strict=True)

    def query(self) -> dict[str, str | list[str]]:
        flags = {"isWebEnabled": self.web_enabled, "isAdult": self.adult, "isPro": self.pro}
        query: dict[str, str | list[str]] = {
            key: str(value).lower() for key, value in flags.items() if value is not None
        }
        for key, values in (
            ("categories", self.categories),
            ("tags", self.tags),
            ("modelId", self.model_ids),
        ):
            if values:
                query[key] = list(values)
        for key, value in (
            ("search", self.search),
            ("sortBy", self.sort_by),
            ("sortOrder", self.sort_order),
        ):
            if value is not None:
                query[key] = value
        if self.limit is not None:
            query["limit"] = str(self.limit)
        if self.offset is not None:
            query["offset"] = str(self.offset)
        return query


class CharacterSlugInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str = Field(min_length=1)

    def path(self) -> str:
        # Quote the slug so it stays one path segment and cannot reach another endpoint.
        return f"characters/{quote(self.slug, safe='')}"


class CharacterReviewsInput(CharacterSlugInput):
    # Venice numbers pages from 1, with 20 reviews per page by default.
    page: int | None = Field(default=None, ge=1, strict=True)
    page_size: int | None = Field(default=None, ge=1, le=100, strict=True)

    def query(self) -> dict[str, str]:
        query: dict[str, str] = {}
        if self.page is not None:
            query["page"] = str(self.page)
        if self.page_size is not None:
            query["pageSize"] = str(self.page_size)
        return query


def image_body(model: str, prompt: str, params: ImageParams | None) -> dict[str, Any]:
    inputs = _ImageInput(model=model, prompt=prompt)
    return {
        **inputs.model_dump(),
        **parameter_body(copy_params(ImageParams, params)),
        "return_binary": False,
    }


def speech_body(
    model: str,
    text: str,
    params: SpeechParams | None,
    *,
    streaming: bool,
) -> dict[str, Any]:
    inputs = _SpeechInput(model=model, input=text)
    return {
        **inputs.model_dump(),
        **parameter_body(copy_params(SpeechParams, params)),
        "streaming": streaming,
    }


def json_object(response: httpx2.Response, action: str) -> dict[str, Any]:
    try:
        payload: Any = response.json()
    except (ValueError, UnicodeError) as exc:
        raise VeniceResponseError(action, "invalid JSON") from exc
    if not isinstance(payload, dict):
        raise VeniceResponseError(action, "expected a JSON object")
    return cast(dict[str, Any], payload)


_T = TypeVar("_T")


@lru_cache(maxsize=None)
def _required_validators(kind: type[Any]) -> tuple[tuple[str, TypeAdapter[Any]], ...]:
    hints = get_type_hints(kind)
    required: frozenset[str] = getattr(kind, "__required_keys__")
    return tuple((key, TypeAdapter(hints[key])) for key in sorted(required))


def resource(value: Any, kind: type[_T], action: str) -> _T:
    """Check the object and its required keys; optional fields pass through unchecked.

    Optional metadata drifts (e.g. a field becoming null) without breaking the
    resource. Unfamiliar JSON and wire-format timestamps are preserved.
    """
    if not isinstance(value, dict):
        raise VeniceResponseError(action, f"invalid {kind.__name__}: expected an object")
    obj = cast(dict[str, Any], value)
    for key, validator in _required_validators(kind):
        try:
            validator.validate_python(obj[key], strict=True)
        except (KeyError, ValidationError) as exc:
            raise VeniceResponseError(action, f"invalid {kind.__name__}: {key}") from exc
    return cast(_T, obj)


def model_resource(value: Any, action: str) -> ModelInfo:
    """Validate structure; optional catalog constraints are normalized by adapters."""
    if not isinstance(value, dict):
        raise VeniceResponseError(action, "expected model object")
    obj = cast(dict[str, Any], value)
    if any(not isinstance(obj.get(key), str) or not obj[key] for key in ("id", "type")):
        raise VeniceResponseError(action, "model requires nonempty id and type")
    if "model_spec" in obj and not isinstance(obj["model_spec"], dict):
        raise VeniceResponseError(action, "model_spec must be an object")
    return cast(ModelInfo, obj)


def list_data(envelope: dict[str, Any], action: str) -> list[Any]:
    value: Any = envelope.get("data")
    if not isinstance(value, list):
        raise VeniceResponseError(action, "missing data array")
    return cast(list[Any], value)


_IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpeg", "image/webp": "webp"}
_AUDIO_TYPES = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/opus": "opus",
    "audio/ogg": "opus",
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/pcm": "pcm",
    "audio/l16": "pcm",
}


def _media_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def media_format(
    content_type: str | None,
    data: bytes,
    fallback: str | None,
    *,
    image: bool,
) -> str:
    types = _IMAGE_TYPES if image else _AUDIO_TYPES
    media_type = _media_type(content_type)
    if media_type in types:
        return types[media_type]
    if image:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if data.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "webp"
    else:
        # ADTS (AAC) and MPEG audio frames share the sync bits. ADTS always has
        # layer 00, which MPEG audio reserves, so check it first.
        if len(data) > 1 and data[0] == 0xFF and data[1] & 0xF6 == 0xF0:
            return "aac"
        if data.startswith(b"ID3") or (
            len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0 and data[1] & 0x06
        ):
            return "mp3"
        if data.startswith(b"fLaC"):
            return "flac"
        if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
            return "wav"
        if data.startswith(b"OggS") and b"OpusHead" in data[:64]:
            return "opus"
    if fallback in types.values():
        return cast(str, fallback)
    raise VeniceResponseError("Reading media", "unknown output format")


def _header_bool(response: httpx2.Response, name: str) -> bool:
    value = response.headers.get(name, "false").lower()
    if value not in {"true", "false"}:
        raise VeniceResponseError("Generating image", f"invalid {name} header")
    return value == "true"


# Advisory headers never fail a generation that has already been charged.
def _header_text(response: httpx2.Response, name: str) -> str | None:
    return response.headers.get(name, "").strip() or None


def _deprecation(response: httpx2.Response) -> ModelDeprecation | None:
    deprecation = ModelDeprecation(
        (_header_text(response, "x-venice-deprecated") or "").lower() == "true",
        _header_text(response, "x-venice-model-deprecation-warning"),
        _header_text(response, "x-venice-model-deprecation-date"),
        _header_text(response, "x-venice-deprecated-replacement"),
    )
    return None if deprecation == ModelDeprecation() else deprecation


def parse_image(response: httpx2.Response, model: str, requested: str | None) -> ImageResult:
    action = "Generating image"
    violation = _header_bool(response, "x-venice-is-content-violation")
    blurred = _header_bool(response, "x-venice-is-blurred")
    header_count = response.headers.get("x-venice-content-violation-count", "0")
    try:
        count = int(header_count)
        if count < 0:
            raise ValueError("negative count")
    except ValueError as exc:
        raise VeniceResponseError(action, "invalid omitted-variant count") from exc
    try:
        obj = json_object(response, action)
    except VeniceResponseError:
        if not violation and not count:
            raise
        obj = {}
    if "content_policy_violation_count" in obj:
        value: Any = obj["content_policy_violation_count"]
        if type(value) is not int or value < 0:
            raise VeniceResponseError(action, "invalid omitted-variant count")
        count = value
    encoded: Any = obj.get("images", [])
    if not isinstance(encoded, list) or (not encoded and not violation and not count):
        raise VeniceResponseError(action, "response did not include any images")
    images: list[GeneratedImage] = []
    metadata_format = obj.get("format")
    fallback = metadata_format if isinstance(metadata_format, str) else requested
    for item in cast(list[Any], encoded):
        try:
            if not isinstance(item, str):
                raise ValueError("image must be base64 text")
            data = base64.b64decode(item, validate=True)
            if not data:
                raise ValueError("empty image")
        except (ValueError, binascii.Error) as exc:
            raise VeniceResponseError(action, "invalid base64 image") from exc
        format = media_format(response.headers.get("content-type"), data, fallback, image=True)
        images.append(GeneratedImage(data, format, f"image/{format}"))
    timing: Any = obj.get("timing")
    if timing is not None and not isinstance(timing, dict):
        raise VeniceResponseError(action, "timing must be an object")
    # Venice URL-encodes the rewritten prompt to fit it in a header.
    enhanced = _header_text(response, "x-venice-enhanced-prompt")
    return ImageResult(
        model,
        tuple(images),
        violation,
        count,
        blurred,
        cast(dict[str, Any] | None, timing),
        {key: value for key, value in obj.items() if key != "images"},
        enhanced_prompt=unquote(enhanced) if enhanced else None,
        deprecation=_deprecation(response),
    )


def parse_upscale(response: httpx2.Response) -> GeneratedImage:
    content_type = response.headers.get("content-type")
    if not response.content:
        raise VeniceResponseError("Upscaling image", "response did not include an image")
    return GeneratedImage(
        response.content,
        media_format(content_type, response.content, None, image=True),
        content_type,
    )


SPEECH_ACTION = "Generating speech"


# Speech without an audio type is identified by its bytes, so accept these too.
_GENERIC_BINARY_TYPES = {"", "application/octet-stream", "binary/octet-stream", "application/ogg"}


def is_speech_content_type(content_type: str | None) -> bool:
    """Whether a speech response may carry audio: audio/*, generic binary, or no type."""
    media_type = _media_type(content_type)
    return media_type.startswith("audio/") or media_type in _GENERIC_BINARY_TYPES


def _is_text_media(media_type: str) -> bool:
    return (
        media_type.startswith("text/")
        or media_type in {"application/json", "application/xml"}
        or media_type.endswith(("+json", "+xml"))
    )


def check_speech_content_type(content_type: str | None, body: bytes) -> None:
    """Reject a successful response that is not audio, citing a text body's message."""
    if is_speech_content_type(content_type):
        return
    media_type = _media_type(content_type)
    detail = f"expected audio but received {media_type}"
    if _is_text_media(media_type):
        message: Any = None
        try:
            payload: Any = json.loads(body)
            if isinstance(payload, dict):
                obj = cast(dict[str, Any], payload)
                message = obj.get("message") or obj.get("error")
        except (ValueError, UnicodeError):
            pass
        if message is None:
            message = body.decode("utf-8", "replace").strip()
        if message:
            detail += f": {str(message)[:200]}"
    raise VeniceResponseError(SPEECH_ACTION, detail)


def parse_speech(response: httpx2.Response, model: str, requested: str | None) -> SpeechResult:
    content_type = response.headers.get("content-type")
    check_speech_content_type(content_type, response.content)
    if not response.content:
        raise VeniceResponseError(SPEECH_ACTION, NO_AUDIO)
    return SpeechResult(
        model,
        response.content,
        content_type,
        media_format(content_type, response.content, requested, image=False),
    )


# The CLI and public methods share these operations; only public methods unwrap data.
def json_operation(name: str, inputs: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    if name == "models":
        type_filter = TypeAdapter(str).validate_python(inputs.get("type", "all"), strict=True)
        return "GET", "models", "Fetching model list", {"params": {"type": type_filter}}
    if name == "characters":
        return (
            "GET",
            "characters",
            "Listing characters",
            {"params": CharacterFilters.model_validate(inputs).query()},
        )
    if name == "character":
        path = CharacterSlugInput.model_validate(inputs).path()
        return "GET", path, "Retrieving character", {}
    if name == "character_reviews":
        reviews = CharacterReviewsInput.model_validate(inputs)
        return (
            "GET",
            f"{reviews.path()}/reviews",
            "Listing character reviews",
            {"params": reviews.query()},
        )
    if name == "keys":
        return "GET", "api_keys", "Listing API keys", {}
    if name == "rate_limits":
        return "GET", "api_keys/rate_limits", "Retrieving rate limits", {}
    if name == "rate_limits_log":
        return "GET", "api_keys/rate_limits/log", "Retrieving rate limit logs", {}
    if name == "create_key":
        return (
            "POST",
            "api_keys",
            "Creating API key",
            {"json": KeyInput.model_validate(inputs).body()},
        )
    if name == "delete_key":
        return (
            "DELETE",
            "api_keys",
            "Deleting API key",
            {"params": KeyIdInput.model_validate(inputs).model_dump()},
        )
    if name == "get_key":
        # Quote the ID so it stays one path segment and cannot reach another endpoint.
        key_id = quote(KeyIdInput.model_validate(inputs).id, safe="")
        return "GET", f"api_keys/{key_id}", "Retrieving API key", {}
    if name == "update_key":
        return (
            "PATCH",
            "api_keys",
            "Updating API key",
            {"json": KeyUpdateInput.model_validate(inputs).body()},
        )
    if name == "augment_search":
        return (
            "POST",
            "augment/search",
            "Searching the web",
            {"json": AugmentSearchInput.model_validate(inputs).body()},
        )
    if name == "augment_scrape":
        return (
            "POST",
            "augment/scrape",
            "Scraping web page",
            {"json": AugmentScrapeInput.model_validate(inputs).model_dump()},
        )
    raise ValueError(f"Unknown JSON operation: {name}")


def parse_envelope(response: httpx2.Response, name: str, action: str) -> dict[str, Any]:
    from .types import (
        ApiKey,
        AugmentScrapeResponse,
        AugmentSearchResult,
        Character,
        CharacterReview,
        CreatedApiKey,
        RateLimitLogEntry,
        RateLimits,
    )

    if name == "delete_key" and response.status_code == 204:
        return {}
    obj = json_object(response, action)
    if name == "models":
        # Catalog semantics: skip malformed entries, reject only a wholly invalid list.
        items = list_data(obj, action)
        valid: list[ModelInfo] = []
        for item in items:
            try:
                valid.append(model_resource(item, action))
            except VeniceResponseError:
                continue
        if items and not valid:
            raise VeniceResponseError(action, "no usable models returned")
        obj = {**obj, "data": valid}
    elif name == "characters":
        for item in list_data(obj, action):
            resource(item, Character, action)
    elif name == "character":
        resource(obj.get("data"), Character, action)
    elif name == "character_reviews":
        for item in list_data(obj, action):
            resource(item, CharacterReview, action)
    elif name == "keys":
        for item in list_data(obj, action):
            resource(item, ApiKey, action)
    elif name == "rate_limits_log":
        for item in list_data(obj, action):
            resource(item, RateLimitLogEntry, action)
    elif name == "rate_limits":
        resource(obj.get("data"), RateLimits, action)
    elif name == "create_key":
        resource(obj.get("data"), CreatedApiKey, action)
    elif name in ("get_key", "update_key"):
        # An update returns the key without its usage fields, which are optional.
        resource(obj.get("data"), ApiKey, action)
    elif name == "delete_key":
        # A 200 must confirm the revocation; anything else may have left the key live.
        if obj.get("success") is not True:
            raise VeniceResponseError(action, "Venice did not confirm the deletion")
    elif name == "augment_search":
        # Not an envelope. Check items one by one so that drift in their optional
        # fields (such as a null date) does not fail the whole search.
        if not isinstance(obj.get("query"), str):
            raise VeniceResponseError(action, "invalid AugmentSearchResponse: query")
        results: Any = obj.get("results")
        if not isinstance(results, list):
            raise VeniceResponseError(action, "invalid AugmentSearchResponse: results")
        for item in cast(list[Any], results):
            resource(item, AugmentSearchResult, action)
    elif name == "augment_scrape":
        resource(obj, AugmentScrapeResponse, action)
    return obj
