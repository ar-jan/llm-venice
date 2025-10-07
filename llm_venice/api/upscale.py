"""Image upscaling functionality for Venice API."""

import datetime
import pathlib

import click
import httpx

from llm_venice.constants import ENDPOINT_IMAGE_UPSCALE
from llm_venice.api.client import get_auth_headers


def image_upscale(
    image_path,
    scale,
    enhance=False,
    enhance_creativity=None,
    enhance_prompt=None,
    replication=None,
    output_path=None,
    overwrite=False,
):
    """
    Upscale an image using Venice AI.

    Args:
        image_path: Path to the input image
        scale: Scale factor (1-4)
        enhance: Whether to enhance the image
        enhance_creativity: Creativity level for enhancement (0.0-1.0)
        enhance_prompt: Text prompt for enhancement
        replication: Line preservation strength (0.0-1.0)
        output_path: Path for the output image (file or directory)
        overwrite: Whether to overwrite existing files

    Example usage:
        llm venice upscale image.jpg --scale 4
    """
    headers = get_auth_headers()

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

    if output_path is None:
        # No output path specified, save next to input
        output_path = input_path.parent / default_filename
    else:
        output_path = pathlib.Path(output_path)
        if output_path.is_dir():
            # If output_path is a directory, save there with default filename
            output_path = output_path / default_filename

    # Handle existing files by adding timestamp
    if output_path.exists() and not overwrite:
        stem = output_path.stem
        suffix = output_path.suffix
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{stem}_{timestamp}{suffix}"
        output_path = output_path.parent / new_filename

    try:
        output_path.write_bytes(image_bytes)
        click.echo(f"Upscaled image saved to {output_path}")
    except Exception as e:
        raise ValueError(f"Failed to write image file: {e}")
