"""Helpers shared by the image commands: input sources and output paths."""

from pathlib import Path
from urllib.parse import unquote, urlsplit

import click

from llm_venice._endpoints import is_image_url
from llm_venice.results import GeneratedImage
from llm_venice._files import publish_media_file, write_chunks


class ImageSource(click.ParamType):
    """An http(s) URL, kept as a str for Venice to fetch, or an existing local file."""

    name = "PATH|URL"

    def convert(self, value, param, ctx) -> str | Path:
        if is_image_url(value):
            return value
        path = Path(value)
        if not path.is_file():
            self.fail(f"File {str(value)!r} does not exist or is not a file.", param, ctx)
        # A Path, so a local name that starts like a URL is still read as a file.
        return path


def output_target(source: str | Path, suffix: str, image: GeneratedImage, output) -> Path:
    """Where to save: ``output`` (a file, or a directory for the default name), else
    ``<stem>_<suffix>.<ext>`` next to a local source or in the current directory for a URL.
    """
    extension = "jpg" if image.format == "jpeg" else image.format
    if isinstance(source, str):
        stem = Path(unquote(urlsplit(source).path)).stem or "image"
        default = Path(f"{stem}_{suffix}.{extension}")
    else:
        default = source.with_name(f"{source.stem}_{suffix}.{extension}")
    target = Path(output) if output else default
    if target.is_dir():
        target = target / default.name
    return target


def save_image(image: GeneratedImage, target: Path, *, overwrite: bool) -> Path:
    """Publish atomically; without overwrite, a taken name gets a timestamp instead."""
    return write_chunks([image.data], target, overwrite=overwrite, publish=publish_media_file).path
