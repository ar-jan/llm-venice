"""Upscale command for Venice CLI."""

import click
from pathlib import Path

from llm_venice.client import VeniceClient
from llm_venice._files import write_chunks
from llm_venice._files import publish_media_file
from llm_venice.errors import VeniceError
from llm_venice.cli.errors import handle_cli_error


def create_upscale_command():
    """
    Create the upscale command.

    Returns:
        Click command for image upscaling
    """

    @click.command(name="upscale")
    @click.argument("image_path", type=click.Path(exists=True, dir_okay=False, readable=True))
    @click.option(
        "--scale",
        # The endpoint accepts only 2 or 4; reject other values at parse time.
        type=click.Choice(["2", "4"]),
        default="2",
        show_default=True,
        help="Scale factor (2 or 4)",
    )
    @click.option(
        "--enhance",
        is_flag=True,
        default=False,
        help="Enhance the image using Venice's image engine",
    )
    @click.option(
        "--enhance-creativity",
        type=click.FloatRange(0.0, 1.0),
        default=None,
        show_default=True,
        help=("Higher values let the enhancement AI change the image more."),
    )
    @click.option(
        "--enhance-prompt",
        type=str,
        default=None,
        show_default=True,
        help="A short descriptive image style prompt to apply during enhancement",
    )
    @click.option(
        "--replication",
        type=click.FloatRange(0.0, 1.0),
        default=None,
        show_default=True,
        help=("How strongly lines and noise in the base image are preserved."),
    )
    @click.option(
        "--output-path",
        "-o",
        type=click.Path(file_okay=True, dir_okay=True, writable=True),
        help="Output path (file or directory)",
    )
    @click.option(
        "--overwrite",
        is_flag=True,
        default=False,
        help="Overwrite existing files",
    )
    def upscale(**kwargs):
        """Upscale an image using Venice API"""
        try:
            image_path = Path(kwargs.pop("image_path"))
            output = kwargs.pop("output_path")
            overwrite = kwargs.pop("overwrite")
            kwargs["scale"] = float(kwargs["scale"])
            with VeniceClient() as client:
                result = client.upscale_image(image_path, **kwargs)
            extension = "jpg" if result.format == "jpeg" else result.format
            default = image_path.with_name(f"{image_path.stem}_upscaled.{extension}")
            target = Path(output) if output else default
            if target.is_dir():
                target = target / default.name
            saved_path = write_chunks(
                [result.data], target, overwrite=overwrite, publish=publish_media_file
            ).path
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        click.echo(f"Upscaled image saved to {saved_path}")

    return upscale
