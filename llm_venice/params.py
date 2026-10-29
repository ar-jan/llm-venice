"""Endpoint parameters. None omits a field and lets Venice choose its default."""

from typing import Any, Literal, TypeVar
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


_RESERVED = {
    "model",
    "input",
    "return_binary",
    "streaming",
    "output_dir",
    "output_filename",
    "overwrite_files",
    "output_path",
    "output_paths",
    "progress",
    "stdout",
}


class _VeniceParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, allow_inf_nan=False)
    extra_body: dict[str, Any] | None = Field(
        default=None, description="Additional API fields; cannot override declared parameters."
    )

    @model_validator(mode="after")
    def _check_extra_body(self) -> Self:
        reserved = _RESERVED | set(type(self).model_fields)
        reserved.update(field.alias for field in type(self).model_fields.values() if field.alias)
        reserved.update(
            field.serialization_alias
            for field in type(self).model_fields.values()
            if field.serialization_alias
        )
        if isinstance(self, ImageParams):
            reserved.add("prompt")
        collisions = reserved.intersection(self.extra_body or {})
        if collisions:
            raise ValueError(
                f"extra_body conflicts with reserved fields: {', '.join(sorted(collisions))}"
            )
        return self


class ImageParams(_VeniceParams):
    """Image generation parameters, with API defaults rather than plugin defaults."""

    negative_prompt: str | None = Field(default=None, max_length=7500)
    style_preset: str | None = None
    width: int | None = Field(default=None, gt=0, le=1280)
    height: int | None = Field(default=None, gt=0, le=1280)
    aspect_ratio: str | None = None
    resolution: str | None = None
    steps: int | None = None
    quality: Literal["low", "medium", "high"] | None = None
    transparent_background: bool | None = None
    cfg_scale: float | None = Field(default=None, gt=0, le=20)
    seed: int | None = Field(default=None, ge=-999_999_999, le=999_999_999)
    lora_strength: int | None = Field(default=None, ge=0, le=100)
    safe_mode: bool | None = None
    hide_watermark: bool | None = None
    variants: int | None = Field(default=None, ge=1, le=4)
    image_format: Literal["png", "jpeg", "webp"] | None = Field(
        default=None,
        validation_alias=AliasChoices("image_format", "format"),
        serialization_alias="format",
    )
    embed_exif_metadata: bool | None = None
    enable_web_search: bool | None = None

    @model_validator(mode="after")
    def _transparent_format(self) -> Self:
        if self.transparent_background and self.image_format == "jpeg":
            raise ValueError("transparent_background requires png or webp format")
        return self


class SpeechParams(_VeniceParams):
    """Speech parameters. Voice IDs are open strings, including cloned handles."""

    voice: str | None = Field(default=None, max_length=512)
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)


_P = TypeVar("_P", bound=_VeniceParams)


def copy_params(cls: type[_P], options: BaseModel | None) -> _P:
    """Revalidate attributes, preserving extras while dropping adapter-only fields."""
    return cls.model_validate(
        {name: getattr(options, name) for name in cls.model_fields} if options is not None else {}
    )


def parameter_body(params: _VeniceParams) -> dict[str, Any]:
    """Serialize independently from copying; explicit nulls in extras survive."""
    body = params.model_dump(by_alias=True, exclude_none=True, exclude={"extra_body"})
    body.update(params.extra_body or {})
    return body
