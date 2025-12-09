"""Venice image generation model implementation."""

import base64
import os
import pathlib
from dataclasses import dataclass
from typing import Literal, Optional, Union

import httpx
import llm
from llm.utils import logging_client
from pydantic import ConfigDict, Field

from llm_venice.constants import (
    ENDPOINT_IMAGE_GENERATE,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGE_HIDE_WATERMARK,
    DEFAULT_IMAGE_SAFE_MODE,
)
from llm_venice.utils import (
    generate_timestamp_filename,
    get_unique_filepath,
    validate_output_directory,
)
from llm_venice.api.client import get_auth_headers_with_content_type


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
        description="Height of generated image", default=DEFAULT_IMAGE_SIZE, ge=64, le=1280
    )
    width: Optional[int] = Field(
        description="Width of generated image", default=DEFAULT_IMAGE_SIZE, ge=64, le=1280
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
    image_format: Optional[Literal["png", "webp"]] = Field(
        description="The image format to return",
        default=DEFAULT_IMAGE_FORMAT,
        alias="format",
    )
    embed_exif_metadata: Optional[bool] = Field(
        description="Embed prompt generation information in the image's EXIF metadata",
        default=False,
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


@dataclass
class ImageGenerationResult:
    image_bytes: Optional[bytes]
    output_path: Optional[pathlib.Path]
    response_json: Optional[dict] = None
    content_violation: bool = False


def generate_image_result(
    *,
    prompt: str,
    options: llm.Options,
    model_name: str,
    api_key: str,
) -> ImageGenerationResult:
    """
    Generate an image via the Venice API without writing to disk.

    Returns the image bytes, the resolved output path, and any response metadata.
    """
    options_dict = options.model_dump(by_alias=True)
    output_dir = options_dict.pop("output_dir", None)
    output_filename = options_dict.pop("output_filename", None)
    overwrite_files = options_dict.pop("overwrite_files", False)
    return_binary = options_dict.get("return_binary", False)
    image_format = options_dict.get("format")

    resolved_output_dir = validate_output_directory(output_dir)

    payload = {
        "model": model_name,
        "prompt": prompt,
        **{k: v for k, v in options_dict.items() if v is not None},
    }

    headers = get_auth_headers_with_content_type(api_key)

    # Logging client option like LLM_OPENAI_SHOW_RESPONSES
    if os.environ.get("LLM_VENICE_SHOW_RESPONSES"):
        client = logging_client()
        r = client.post(ENDPOINT_IMAGE_GENERATE, headers=headers, json=payload, timeout=120)
    else:
        r = httpx.post(ENDPOINT_IMAGE_GENERATE, headers=headers, json=payload, timeout=120)

    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ValueError(f"API request failed: {e.response.text}")

    if r.headers.get("x-venice-is-content-violation") == "true":
        return ImageGenerationResult(image_bytes=None, output_path=None, content_violation=True)

    response_json = None
    if return_binary:
        image_bytes = r.content
    else:
        data = r.json()
        response_json = {
            "request": data["request"],
            "timing": data["timing"],
        }
        image_data = data["images"][0]
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image data: {e}")

    target_dir = resolved_output_dir or (llm.user_dir() / "images")

    if not output_filename:
        extension = image_format or DEFAULT_IMAGE_FORMAT
        output_filename = generate_timestamp_filename("venice", model_name, extension)

    output_filepath = get_unique_filepath(target_dir, output_filename, overwrite_files)

    return ImageGenerationResult(
        image_bytes=image_bytes,
        output_path=output_filepath,
        response_json=response_json,
        content_violation=False,
    )


def save_image_result(result: ImageGenerationResult) -> pathlib.Path:
    """Persist an ImageGenerationResult to disk."""
    if result.output_path is None:
        raise ValueError("No output path available to save image")
    result.output_path.parent.mkdir(parents=True, exist_ok=True)
    if result.image_bytes is None:
        raise ValueError("No image bytes available to save")
    result.output_path.write_bytes(result.image_bytes)
    return result.output_path


class VeniceImage(llm.KeyModel):
    """Venice AI image generation model."""

    can_stream = False
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"

    def __init__(self, model_id, model_name=None):
        self.model_id = f"venice/{model_id}"
        self.model_name = model_id

    def __str__(self):
        return f"Venice Image: {self.model_id}"

    class Options(VeniceImageOptions):  # type: ignore[override]
        pass

    def execute(self, prompt, stream, response, conversation=None, key=None):
        """Execute image generation request."""
        api_key = self.get_key(key)
        if api_key is None:
            raise llm.NeedsKeyException("No key found for Venice")
        result = generate_image_result(
            prompt=prompt.prompt,
            options=prompt.options,
            model_name=self.model_name,
            api_key=api_key,
        )

        if result.content_violation:
            yield "Response marked as content violation; no image was returned."
            return

        if result.response_json is not None:
            response.response_json = result.response_json

        try:
            saved_path = save_image_result(result)
            yield f"Image saved to {saved_path}"
        except Exception as e:
            raise ValueError(f"Failed to write image file: {e}")
