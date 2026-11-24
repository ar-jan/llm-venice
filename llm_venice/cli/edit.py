"""Image edit commands for Venice CLI."""

from typing import get_args

import click
from pydantic import ValidationError

from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceError
from llm_venice.models._common import normalize_model_id
from llm_venice.cli.errors import handle_cli_error
from llm_venice.cli.images import (
    ImageSource,
    check_edit_request,
    edit_notices,
    edit_options,
    output_target,
    save_image,
)
from llm_venice.cli.notices import emit_cli_notices
from llm_venice.cli.output import clean_text
from llm_venice.types import ImageQuality

_OPTION_NAMES = (
    "aspect_ratio",
    "resolution",
    "output_format",
    "safe_mode",
    "transparent_background",
    "enhance_prompt",
    "disable_prompt_optimization_thinking",
    "anon_user_id",
    "quality",
)


# Arguments and flags whose names differ from their fields.
_FLAGS = {"prompt": "PROMPT", "output_format": "--format"}


def _given(values: dict) -> dict:
    """The edit options that were set; the rest are left to Venice."""
    return {name: values[name] for name in _OPTION_NAMES if values.get(name) is not None}


def _usage_error(exc: ValidationError) -> click.UsageError:
    messages = []
    for error in exc.errors():
        if not error["loc"]:
            # A check across options, such as a transparent background in JPEG.
            messages.append(str(error.get("ctx", {}).get("error") or error["msg"]))
            continue
        field = str(error["loc"][0])
        flag = _FLAGS.get(field, f"--{field.replace('_', '-')}")
        messages.append(f"Invalid value for {flag}: {error['msg']}")
    return click.UsageError("\n".join(messages))


def _save(client_call, source, model, prompt, options, output_path, overwrite, images=1):
    model = normalize_model_id(model) if model else None
    check_edit_request(model, prompt, options, images=images)
    try:
        with VeniceClient() as client:
            result = client_call(client, model)
        emit_cli_notices(edit_notices(result, model))
        if result.content_violation:
            raise click.ClickException(
                "Venice marked the edited image as a content violation; it was not saved"
            )
        target = output_target(source, "edited", result.image, output_path)
        saved_path = save_image(result.image, target, overwrite=overwrite)
    except ValidationError as exc:
        raise _usage_error(exc) from exc
    except (VeniceError, ValueError, OSError) as e:
        handle_cli_error(e)
    if result.enhanced_prompt:
        click.echo(f"Enhanced prompt: {clean_text(result.enhanced_prompt)}", err=True)
    click.echo(f"Edited image saved to {saved_path}")


def create_edit_command():
    """
    Create the edit command.

    Returns:
        Click command for editing one image with a prompt
    """

    @click.command(name="edit")
    @click.argument("image", type=ImageSource())
    @click.argument("prompt", nargs=-1, required=True)
    @edit_options
    def edit(image, prompt, model, output_path, overwrite, **values):
        """Edit an image as the prompt directs.

        IMAGE is a local file or an http(s) URL, which Venice fetches. Multiple
        PROMPT words are joined with spaces. The result is saved as
        <name>_edited.<ext> next to a local image, or in the current directory for
        a URL. Each edit is billed.

        \b
        Example:
          llm venice edit photo.jpg "change the sky to sunrise" -m qwen-image-2-1-turbo-edit
        """
        text = " ".join(prompt)
        options = _given(values)
        _save(
            lambda client, model_id: client.edit_image(image, text, model=model_id, **options),
            image,
            model,
            text,
            options,
            output_path,
            overwrite,
        )

    return edit


def create_multi_edit_command():
    """
    Create the multi-edit command.

    Returns:
        Click command for editing layered images with one prompt
    """

    @click.command(name="multi-edit")
    @click.argument("images", nargs=-1, required=True, type=ImageSource())
    @click.option("-p", "--prompt", required=True, help="Directions for the edit")
    @click.option(
        "--quality",
        type=click.Choice(get_args(ImageQuality)),
        help="Output quality on models that offer it; higher can cost more",
    )
    @edit_options
    def multi_edit(images, prompt, quality, model, output_path, overwrite, **values):
        """Edit layered images with one prompt; the first IMAGE is the base.

        Each IMAGE is a local file or an http(s) URL, which Venice fetches. The rest
        are layers or masks; how many a model takes is maxInputImages in the catalog.
        The result is saved as <name>_edited.<ext> after the first image. Each edit is
        billed.

        \b
        Example:
          llm venice multi-edit room.jpg sofa.png -p "place the sofa in the room"
        """
        options = _given({**values, "quality": quality})
        _save(
            lambda client, model_id: client.multi_edit_image(
                images, prompt, model=model_id, **options
            ),
            images[0],
            model,
            prompt,
            options,
            output_path,
            overwrite,
            images=len(images),
        )

    return multi_edit
