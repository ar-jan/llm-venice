"""Image upscaling functionality for Venice API."""

import datetime
import pathlib
from dataclasses import dataclass
from typing import Optional, Union

import httpx

from llm_venice.constants import ENDPOINT_IMAGE_UPSCALE
from llm_venice.api.client import get_auth_headers


@dataclass
class UpscaleResult:
    image_bytes: bytes
    output_path: pathlib.Path


def perform_image_upscale(
    image_path,
    scale,
    enhance: bool = False,
    enhance_creativity=None,
    enhance_prompt=None,
    replication=None,
    output_path: Optional[Union[pathlib.Path, str]] = None,
    overwrite: bool = False,
    *,
    key=None,
) -> UpscaleResult:
    """
    Upscale an image using Venice AI without writing to disk.

    Returns an UpscaleResult containing the image bytes and the resolved output path.
    """
    headers = get_auth_headers(key)

    with open(image_path, "rb") as img_file:
        image_data = img_file.read()

    # Create multipart form data
    files = {
        "image": (pathlib.Path(image_path).name, image_data),
    }

    data = {
        "scale": scale,
        "enhance": enhance,
        "enhanceCreativity": enhance_creativity,
        "enhancePrompt": enhance_prompt,
        "replication": replication,
    }
    # Remove None values from data in order to use API defaults
    data = {k: v for k, v in data.items() if v is not None}

    r = httpx.post(ENDPOINT_IMAGE_UPSCALE, headers=headers, files=files, data=data, timeout=120)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ValueError(f"API request failed: {e.response.text}")

    image_bytes = r.content

    # Handle output path logic
    input_path = pathlib.Path(image_path)
    # The upscaled image is always PNG
    default_filename = f"{input_path.stem}_upscaled.png"

    resolved_output_path = (
        pathlib.Path(output_path) if output_path else input_path.parent / default_filename
    )
    if resolved_output_path.is_dir():
        resolved_output_path = resolved_output_path / default_filename

    if resolved_output_path.exists() and not overwrite:
        stem = resolved_output_path.stem
        suffix = resolved_output_path.suffix
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{stem}_{timestamp}{suffix}"
        resolved_output_path = resolved_output_path.parent / new_filename

    return UpscaleResult(image_bytes=image_bytes, output_path=resolved_output_path)


def write_upscaled_image(result: UpscaleResult) -> pathlib.Path:
    """Persist an UpscaleResult to disk."""
    result.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.output_path.write_bytes(result.image_bytes)
    return result.output_path
