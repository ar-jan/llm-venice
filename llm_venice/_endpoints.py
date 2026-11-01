"""Pure request builders and response parsers shared by both clients."""

import base64
import binascii
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Literal, TypeVar, cast

import httpx2
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from typing_extensions import get_type_hints

from .errors import VeniceResponseError
from .params import ImageParams, SpeechParams, copy_params, parameter_body
from .results import GeneratedImage, ImageResult, SpeechResult
from .types import AugmentSearchProvider, ModelInfo


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

    @field_validator("scale")
    @classmethod
    def _scale(cls, value: float) -> float:
        if value not in {2, 4}:
            raise ValueError("scale must be 2 or 4")
        return value

    enhance: bool = False
    enhance_creativity: float | None = Field(default=None, ge=0, le=1, alias="enhanceCreativity")
    enhance_prompt: str | None = Field(default=None, alias="enhancePrompt")
    replication: float | None = Field(default=None, ge=0, le=1)


class KeyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    key_type: Literal["ADMIN", "INFERENCE"]
    description: str = Field(default="", max_length=64)
    expires_at: datetime | None = None
    limit_usd: float | None = Field(default=None, ge=0, le=9_999_999_999)
    limit_vcu: float | None = Field(default=None, ge=0, le=9_999_999_999)

    @field_validator("expires_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("expires_at must be timezone-aware")
        return value

    def body(self) -> dict[str, Any]:
        return {
            "apiKeyType": self.key_type,
            "description": self.description,
            "expiresAt": self.expires_at.astimezone(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
            if self.expires_at
            else "",
            "consumptionLimit": {"usd": self.limit_usd, "vcu": self.limit_vcu},
        }


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


class CharacterFilters(BaseModel):
    web_enabled: bool | None = None
    adult: bool | None = None

    def query(self) -> dict[str, str]:
        return {
            key: str(value).lower()
            for key, value in {
                "isWebEnabled": self.web_enabled,
                "isAdult": self.adult,
            }.items()
            if value is not None
        }


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


def media_format(
    content_type: str | None,
    data: bytes,
    fallback: str | None,
    *,
    image: bool,
) -> str:
    types = _IMAGE_TYPES if image else _AUDIO_TYPES
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
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
        if data.startswith(b"ID3") or (len(data) > 1 and data[0] == 255 and data[1] & 0xE0 == 0xE0):
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
    return ImageResult(
        model,
        tuple(images),
        violation,
        count,
        blurred,
        cast(dict[str, Any] | None, timing),
        {key: value for key, value in obj.items() if key != "images"},
    )


def parse_upscale(response: httpx2.Response) -> GeneratedImage:
    content_type = response.headers.get("content-type")
    return GeneratedImage(
        response.content,
        media_format(content_type, response.content, None, image=True),
        content_type,
    )


def parse_speech(response: httpx2.Response, model: str, requested: str | None) -> SpeechResult:
    content_type = response.headers.get("content-type")
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
    if name == "augment_search":
        return (
            "POST",
            "augment/search",
            "Searching the web",
            {"json": AugmentSearchInput.model_validate(inputs).body()},
        )
    raise ValueError(f"Unknown JSON operation: {name}")


def parse_envelope(response: httpx2.Response, name: str, action: str) -> dict[str, Any]:
    from .types import (
        ApiKey,
        AugmentSearchResult,
        Character,
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
    return obj
