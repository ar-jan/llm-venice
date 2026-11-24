"""Helpers shared by the image commands: inputs, catalog checks, notices and output."""

from pathlib import Path
from typing import Any, cast, get_args
from urllib.parse import unquote, urlsplit

import click

from llm_venice._endpoints import is_image_url
from llm_venice._files import publish_media_file, write_chunks
from llm_venice._notices import VeniceNotice
from llm_venice.catalog import load_catalog
from llm_venice.cli.output import clean_text
from llm_venice.results import GeneratedImage, ImageEditResult
from llm_venice.types import ImageOutputFormat


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


def _object(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def check_edit_request(
    model: str | None, prompt: str, options: dict[str, Any], *, images: int = 1
) -> None:
    """Check an edit against the model's cached catalog entry, when there is one.

    Without ``model`` Venice picks its default, so nothing is checked. Venice lists
    edit models as type ``inpaint``; ``llm venice refresh`` updates the cache.
    """
    if model is None:
        return
    entry = next((item for item in load_catalog() or [] if item["id"] == model), None)
    if entry is None:
        return
    spec = _object(entry.get("model_spec"))
    constraints = _object(spec.get("constraints"))
    capabilities = _object(spec.get("capabilities"))
    problems: list[str] = []
    limit = constraints.get("promptCharacterLimit")
    if type(limit) is int and len(prompt) > limit:
        problems.append(f"{model} accepts prompts of at most {limit} characters")
    for name, key in (
        ("aspect_ratio", "aspectRatios"),
        ("resolution", "resolutions"),
        ("quality", "qualities"),
    ):
        value = options.get(name)
        choices = constraints.get(key)
        if value is not None and isinstance(choices, list) and choices and value not in choices:
            allowed = ", ".join(str(choice) for choice in cast(list[Any], choices))
            problems.append(f"--{name.replace('_', '-')} for {model} must be one of: {allowed}")
    if (
        options.get("transparent_background")
        and constraints.get("supportsTransparentBackground") is not True
    ):
        problems.append(f"{model} does not support transparent backgrounds")
    # Swagger names capabilities.maxInputImages; the catalog has it under constraints.
    maximum = constraints.get("maxInputImages", capabilities.get("maxInputImages"))
    if type(maximum) is int and images > maximum:
        problems.append(f"{model} accepts at most {maximum} input images")
    if problems:
        raise click.UsageError("\n".join(problems))


def edit_notices(result: ImageEditResult, model: str | None) -> list[VeniceNotice]:
    """Warnings about a returned edit: blurring and model deprecation."""
    notices: list[VeniceNotice] = []
    if result.blurred:
        notices.append(
            VeniceNotice(
                "warning",
                "edited image was blurred because Safe Venice filtered adult material",
            )
        )
    deprecation = result.deprecation
    if deprecation is not None:
        name = result.model_id or model or "the edit model"
        if deprecation.deprecated:
            message = f"{name} is deprecated"
        else:
            message = f"{name} is scheduled for deprecation"
            if deprecation.date:
                message += f" on {deprecation.date}"
        if deprecation.warning:
            message += f": {deprecation.warning}"
        if deprecation.replacement:
            message += f"; use {deprecation.replacement} instead"
        notices.append(VeniceNotice("warning", clean_text(message, single_line=True)))
    return notices


def edit_options(function):
    """Options shared by `llm venice edit` and `llm venice multi-edit`."""
    for option in reversed(
        [
            click.option(
                "-m",
                "--model",
                help="Edit model ID; Venice's default edit model when omitted",
            ),
            click.option("--aspect-ratio", help="auto, or a ratio such as 16:9"),
            click.option("--resolution", help="Resolution tier, such as 1K, 2K or 4K"),
            click.option(
                "--format",
                "output_format",
                type=click.Choice(get_args(ImageOutputFormat)),
                help="Output format; Venice picks PNG at 1K and JPEG above when omitted",
            ),
            click.option(
                "--safe-mode/--no-safe-mode",
                default=False,
                show_default=True,
                help="Blur adult content (off by default, as for image models)",
            ),
            click.option(
                "--transparent-background",
                is_flag=True,
                default=None,
                help="Transparent background, on supporting models (PNG or WebP)",
            ),
            click.option(
                "--enhance-prompt",
                is_flag=True,
                default=None,
                help="Have Venice rewrite the prompt with more detail first (extra cost)",
            ),
            click.option(
                "--disable-prompt-optimization-thinking",
                is_flag=True,
                default=None,
                help="Skip the prompt-optimization thinking step on models that have one",
            ),
            click.option("--anon-user-id", help="ID of your end user, for attribution"),
            click.option(
                "--output-path",
                "-o",
                type=click.Path(file_okay=True, dir_okay=True, writable=True),
                help="Output path (file or directory)",
            ),
            click.option(
                "--overwrite", is_flag=True, default=False, help="Overwrite existing files"
            ),
        ]
    ):
        function = option(function)
    return function
