"""Image upscaling functionality for Venice API."""

import pathlib
from dataclasses import dataclass
from typing import Optional, Union

import httpx

from llm_venice.constants import ENDPOINT_IMAGE_UPSCALE
from llm_venice.api.client import get_auth_headers
from llm_venice.utils import get_unique_filepath


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

    input_path = pathlib.Path(image_path)
    default_filename = f"{input_path.stem}_upscaled.png"

    if output_path:
        candidate_path = pathlib.Path(output_path)
        if candidate_path.is_dir():
            target_dir = candidate_path
            filename = default_filename
        else:
            target_dir = candidate_path.parent
            filename = candidate_path.name
    else:
        target_dir = input_path.parent
        filename = default_filename

    resolved_output_path = get_unique_filepath(target_dir, filename, overwrite)

    return UpscaleResult(image_bytes=image_bytes, output_path=resolved_output_path)


def write_upscaled_image(result: UpscaleResult) -> pathlib.Path:
    """Persist an UpscaleResult to disk."""
    result.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.output_path.write_bytes(result.image_bytes)
    return result.output_path
