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
        type=click.FloatRange(2.0, 4.0),
        default=2.0,
        show_default=True,
        help="Scale factor, between 2 and 4",
    )
    @click.option(
        "--creativity",
        type=click.FloatRange(0.0, 0.02),
        default=None,
        help=(
            "Detail and texture the upscaler adds, between 0 and 0.02. "
            "Higher adds more; lower stays closer to the source. [API default: 0.01]"
        ),
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
