"""Venice image generation model implementation."""

import asyncio
import base64
import os
import pathlib
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal, Optional, Union

import httpx2
import llm
from llm.utils import logging_client
from pydantic import ConfigDict, Field, model_validator

from llm_venice.constants import (
    ENDPOINT_IMAGE_GENERATE,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_IMAGE_HIDE_WATERMARK,
    DEFAULT_IMAGE_SAFE_MODE,
)
from llm_venice.utils import (
    generate_timestamp_filename,
    get_unique_filepath,
    validate_output_directory,
)
from llm_venice.api.client import get_auth_headers_with_content_type
from llm_venice.api.errors import VeniceAPIError, raise_api_error
from llm_venice.notices import VeniceNotice, render_notices


class VeniceImageOptions(llm.Options):
    """Options for Venice image generation models."""

    model_config = ConfigDict(populate_by_name=True)

    negative_prompt: Optional[str] = Field(
        description="Negative prompt to guide image generation away from certain features",
        default=None,
    )
    style_preset: Optional[str] = Field(
        description="Style preset to use for generation", default=None
    )
    height: Optional[int] = Field(
        description="Height of generated image", default=None, ge=64, le=1280
    )
    width: Optional[int] = Field(
        description="Width of generated image", default=None, ge=64, le=1280
    )
    aspect_ratio: Optional[str] = Field(
        description="Aspect ratio to use for generation", default=None
    )
    resolution: Optional[str] = Field(
        description="Resolution preset to use for generation", default=None
    )
    steps: Optional[int] = Field(description="Number of inference steps", default=None, ge=7, le=50)
    cfg_scale: Optional[float] = Field(
        description="CFG scale for generation", default=None, gt=0, le=20.0
    )
    seed: Optional[int] = Field(
        description="Random seed for reproducible generation",
        default=None,
        ge=-999999999,
        le=999999999,
    )
    lora_strength: Optional[int] = Field(
        description="LoRA adapter strength percentage", default=None, ge=0, le=100
    )
    safe_mode: Optional[bool] = Field(
        description="Enable safety filters", default=DEFAULT_IMAGE_SAFE_MODE
    )
    hide_watermark: Optional[bool] = Field(
        description="Hide watermark in generated image", default=DEFAULT_IMAGE_HIDE_WATERMARK
    )
    return_binary: Optional[bool] = Field(
        description="Return raw binary instead of base64", default=False
    )
    variants: Optional[int] = Field(
        description="Number of images to generate (1-4). Only supported when return_binary is false.",
        default=None,
        ge=1,
        le=4,
    )
    image_format: Optional[Literal["png", "jpeg", "webp"]] = Field(
        description="The image format to return",
        default=DEFAULT_IMAGE_FORMAT,
        alias="format",
    )
    embed_exif_metadata: Optional[bool] = Field(
        description="Embed prompt generation information in the image's EXIF metadata",
        default=False,
    )
    enable_web_search: Optional[bool] = Field(
        description="Enable web search for image generation on supported models",
        default=None,
    )
    output_dir: Optional[Union[pathlib.Path, str]] = Field(
        description="Directory to save generated images",
        default=None,
    )
    output_filename: Optional[str] = Field(
        description="Custom filename for saved image", default=None
    )
    overwrite_files: Optional[bool] = Field(
        description="Option to overwrite existing output files", default=False
    )

    @model_validator(mode="after")
    def validate_variants_with_return_binary(self):
        """Venice only supports multi-image responses in JSON mode."""
        if self.return_binary and self.variants is not None:
            raise ValueError("variants is only supported when return_binary is false")
        return self


@dataclass
class ImageGenerationResult:
    image_bytes_list: list[bytes] = dataclass_field(default_factory=list)
    output_paths: list[pathlib.Path] = dataclass_field(default_factory=list)
    response_json: Optional[dict] = None
    content_violation: bool = False
    is_blurred: bool = False
    notices: list[VeniceNotice] = dataclass_field(default_factory=list)


def _is_true_response_header(headers: httpx2.Headers, header_name: str) -> bool:
    """Return True when a Venice boolean response header is explicitly enabled."""
    return headers.get(header_name, "").lower() == "true"


def append_blurred_notice(notices: list[VeniceNotice], *, is_blurred: bool) -> None:
    """Record a warning when Venice returns a blurred image."""
    if not is_blurred:
        return
    notices.append(
        VeniceNotice(
            level="warning",
            message="generated image was blurred because Safe Venice filtered adult material",
        )
    )


def _decode_base64_images(data: dict[str, Any]) -> list[bytes]:
    """Decode the Venice image array from a JSON response."""
    images = data.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Response did not include any images")

    decoded_images = []
    for image_data in images:
        try:
            decoded_images.append(base64.b64decode(image_data))
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image data: {e}") from e
    return decoded_images


def _filename_with_index(filename: str, index: int) -> str:
    """Append a 1-based index before the filename extension."""
    path = pathlib.Path(filename)
    return f"{path.stem}_{index}{path.suffix}"


def _resolve_output_paths(
    *,
    directory: pathlib.Path,
    output_filename: str,
    overwrite_files: bool,
    image_count: int,
) -> list[pathlib.Path]:
    """Resolve one or more image output paths while preserving current overwrite behavior."""
    if image_count < 1:
        raise ValueError("No output paths available to save image")

    if image_count == 1:
        return [get_unique_filepath(directory, output_filename, overwrite_files)]

    return [
        get_unique_filepath(
            directory,
            _filename_with_index(output_filename, index + 1),
            overwrite_files,
        )
        for index in range(image_count)
    ]


def normalize_image_options_for_model(
    *,
    model_name: str,
    options_dict: dict[str, Any],
    image_constraints: Optional[dict[str, Any]] = None,
) -> list[VeniceNotice]:
    """Drop unsupported options and validate supported values when constraints are available."""
    if not image_constraints:
        return []

    dropped_options = []

    aspect_ratio = options_dict.get("aspect_ratio")
    supported_aspect_ratios = image_constraints.get("aspectRatios")
    if aspect_ratio is not None:
        if not supported_aspect_ratios:
            options_dict.pop("aspect_ratio", None)
            dropped_options.append("aspect_ratio")
        elif supported_aspect_ratios is not None:
            if aspect_ratio not in supported_aspect_ratios:
                allowed = ", ".join(supported_aspect_ratios)
                raise ValueError(
                    f"Invalid aspect_ratio '{aspect_ratio}' for model '{model_name}'. "
                    f"Supported values: {allowed}"
                )

    resolution = options_dict.get("resolution")
    supported_resolutions = image_constraints.get("resolutions")
    if resolution is not None:
        if not supported_resolutions:
            options_dict.pop("resolution", None)
            dropped_options.append("resolution")
        elif supported_resolutions is not None:
            if resolution not in supported_resolutions:
                allowed = ", ".join(supported_resolutions)
                raise ValueError(
                    f"Invalid resolution '{resolution}' for model '{model_name}'. "
                    f"Supported values: {allowed}"
                )

    if not dropped_options:
        return []

    dropped = ", ".join(dropped_options)
    return [
        VeniceNotice(
            level="info",
            message=f"dropped unsupported options for model '{model_name}': {dropped}",
        )
    ]


def generate_image_result(
    *,
    prompt: str,
    options: llm.Options,
    model_id: str,
    model_name: str,
    api_key: str,
    supports_web_search: bool = False,
    image_constraints: Optional[dict[str, Any]] = None,
) -> ImageGenerationResult:
    """
    Generate an image via the Venice API without writing to disk.

    Returns the generated image bytes, resolved output paths, and any response metadata.
    """
    options_dict = options.model_dump(by_alias=True)
    output_dir = options_dict.pop("output_dir", None)
    output_filename = options_dict.pop("output_filename", None)
    overwrite_files = options_dict.pop("overwrite_files", False)
    return_binary = options_dict.get("return_binary", False)
    image_format = options_dict.get("format")
    variants = options_dict.get("variants")
    web_search_requested = options_dict.get("enable_web_search")

    if web_search_requested is True and not supports_web_search:
        raise llm.ModelError(f"Model {model_id} does not support web search")
    if return_binary and variants is not None:
        raise ValueError("variants is only supported when return_binary is false")

    resolved_output_dir = validate_output_directory(output_dir)
    notices = normalize_image_options_for_model(
        model_name=model_name,
        options_dict=options_dict,
        image_constraints=image_constraints,
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        **{k: v for k, v in options_dict.items() if v is not None},
    }

    headers = get_auth_headers_with_content_type(api_key)

    # Logging client option like LLM_OPENAI_SHOW_RESPONSES
    if os.environ.get("LLM_VENICE_SHOW_RESPONSES"):
        with logging_client() as client:
            r = client.post(ENDPOINT_IMAGE_GENERATE, headers=headers, json=payload, timeout=120)
            try:
                r.raise_for_status()
            except httpx2.HTTPStatusError as exc:
                raise_api_error("Generating image", exc)

            content_violation = _is_true_response_header(r.headers, "x-venice-is-content-violation")
            is_blurred = _is_true_response_header(r.headers, "x-venice-is-blurred")
            append_blurred_notice(notices, is_blurred=is_blurred)

            if content_violation:
                return ImageGenerationResult(
                    content_violation=True,
                    is_blurred=is_blurred,
                    notices=notices,
                )

            response_json = None
            if return_binary:
                image_bytes_list = [r.content]
                if is_blurred:
                    response_json = {"is_blurred": True}
            else:
                data = r.json()
                response_json = {
                    "request": data["request"],
                    "timing": data["timing"],
                    "is_blurred": is_blurred,
                }
                image_bytes_list = _decode_base64_images(data)
    else:
        r = httpx2.post(ENDPOINT_IMAGE_GENERATE, headers=headers, json=payload, timeout=120)

        try:
            r.raise_for_status()
        except httpx2.HTTPStatusError as exc:
            raise_api_error("Generating image", exc)

        content_violation = _is_true_response_header(r.headers, "x-venice-is-content-violation")
        is_blurred = _is_true_response_header(r.headers, "x-venice-is-blurred")
        append_blurred_notice(notices, is_blurred=is_blurred)

        if content_violation:
            return ImageGenerationResult(
                content_violation=True,
                is_blurred=is_blurred,
                notices=notices,
            )

        response_json = None
        if return_binary:
            image_bytes_list = [r.content]
            if is_blurred:
                response_json = {"is_blurred": True}
        else:
            data = r.json()
            response_json = {
                "request": data["request"],
                "timing": data["timing"],
                "is_blurred": is_blurred,
            }
            image_bytes_list = _decode_base64_images(data)

    target_dir = resolved_output_dir or (llm.user_dir() / "images")

    if not output_filename:
        extension = image_format or DEFAULT_IMAGE_FORMAT
        output_filename = generate_timestamp_filename("venice", model_name, extension)

    output_filepaths = _resolve_output_paths(
        directory=target_dir,
        output_filename=output_filename,
        overwrite_files=overwrite_files,
        image_count=len(image_bytes_list),
    )

    return ImageGenerationResult(
        image_bytes_list=image_bytes_list,
        output_paths=output_filepaths,
        response_json=response_json,
        content_violation=False,
        is_blurred=is_blurred,
        notices=notices,
    )


def save_image_result(result: ImageGenerationResult) -> list[pathlib.Path]:
    """Persist an ImageGenerationResult to disk."""
    if not result.output_paths:
        raise ValueError("No output path available to save image")
    if not result.image_bytes_list:
        raise ValueError("No image bytes available to save")
    if len(result.output_paths) != len(result.image_bytes_list):
        raise ValueError("Image result paths do not match the number of generated images")

    for output_path, image_bytes in zip(result.output_paths, result.image_bytes_list):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
    return result.output_paths


def render_notices_for_output(notices: list[VeniceNotice]) -> str:
    """Render notices for llm model output, separated from following text."""
    rendered = render_notices(notices)
    if not rendered:
        return ""
    return "\n".join(rendered) + "\n"


def render_saved_paths_for_output(output_paths: list[pathlib.Path]) -> str:
    """Render one or more saved image paths for llm model output."""
    return "\n".join(f"Image saved to {saved_path}" for saved_path in output_paths)


class VeniceImage(llm.KeyModel):
    """Venice AI image generation model."""

    can_stream = False
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    supports_web_search = False

    def __init__(
        self, model_id, model_name=None, image_constraints=None, supports_web_search=False
    ):
        self.model_id = f"venice/{model_id}"
        self.model_name = model_id
        self.image_constraints = image_constraints
        self.supports_web_search = supports_web_search

    def __str__(self):
        return f"Venice Image: {self.model_id}"

    class Options(VeniceImageOptions):  # type: ignore[override]
        pass

    def execute(self, prompt, stream, response, conversation=None, key=None):
        """Execute image generation request."""
        try:
            api_key = self.get_key(key)
            if api_key is None:
                raise llm.NeedsKeyException("No key found for Venice")
            try:
                options_dict = prompt.options.model_dump(by_alias=True)
                validate_output_directory(
                    options_dict.get("output_dir"),
                    create_if_missing=True,
                )
                result = generate_image_result(
                    prompt=prompt.prompt,
                    options=prompt.options,
                    model_id=self.model_id,
                    model_name=self.model_name,
                    api_key=api_key,
                    supports_web_search=self.supports_web_search,
                    image_constraints=self.image_constraints,
                )
            except ValueError as exc:
                raise llm.ModelError(str(exc)) from exc

            if result.content_violation:
                rendered_notices = render_notices_for_output(result.notices)
                if rendered_notices:
                    yield rendered_notices
                yield "Response marked as content violation; no image was returned."
                return

            if result.response_json is not None:
                response.response_json = result.response_json

            try:
                save_image_result(result)
                rendered_notices = render_notices_for_output(result.notices)
                if rendered_notices:
                    yield rendered_notices
                yield render_saved_paths_for_output(result.output_paths)
            except (OSError, ValueError) as exc:
                raise llm.ModelError(f"Failed to write image file: {exc}") from exc
        except VeniceAPIError as exc:
            raise llm.ModelError(str(exc)) from exc


class AsyncVeniceImage(llm.AsyncKeyModel):
    """Asynchronous Venice AI image generation model."""

    can_stream = False
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    supports_web_search = False

    def __init__(
        self, model_id, model_name=None, image_constraints=None, supports_web_search=False
    ):
        self.model_id = f"venice/{model_id}"
        self.model_name = model_id
        self.image_constraints = image_constraints
        self.supports_web_search = supports_web_search

    def __str__(self):
        return f"Venice Image: {self.model_id}"

    class Options(VeniceImageOptions):  # type: ignore[override]
        pass

    async def execute(self, prompt, stream, response, conversation=None, key=None):
        """Execute image generation request asynchronously."""
        try:
            api_key = self.get_key(key)
            if api_key is None:
                raise llm.NeedsKeyException("No key found for Venice")
            try:
                options_dict = prompt.options.model_dump(by_alias=True)
                validate_output_directory(
                    options_dict.get("output_dir"),
                    create_if_missing=True,
                )
                result = await asyncio.to_thread(
                    generate_image_result,
                    prompt=prompt.prompt,
                    options=prompt.options,
                    model_id=self.model_id,
                    model_name=self.model_name,
                    api_key=api_key,
                    supports_web_search=self.supports_web_search,
                    image_constraints=self.image_constraints,
                )
            except ValueError as exc:
                raise llm.ModelError(str(exc)) from exc

            if result.content_violation:
                rendered_notices = render_notices_for_output(result.notices)
                if rendered_notices:
                    yield rendered_notices
                yield "Response marked as content violation; no image was returned."
                return

            if result.response_json is not None:
                response.response_json = result.response_json

            try:
                await asyncio.to_thread(save_image_result, result)
                rendered_notices = render_notices_for_output(result.notices)
                if rendered_notices:
                    yield rendered_notices
                yield render_saved_paths_for_output(result.output_paths)
            except (OSError, ValueError) as exc:
                raise llm.ModelError(f"Failed to write image file: {exc}") from exc
        except VeniceAPIError as exc:
            raise llm.ModelError(str(exc)) from exc
