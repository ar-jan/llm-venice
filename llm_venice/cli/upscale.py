"""Upscale command for Venice CLI."""

import click
import httpx
import llm

from llm_venice.api.upscale import perform_image_upscale, write_upscaled_image
from llm_venice.api.errors import VeniceAPIError
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
        type=click.FloatRange(1, 4),
        default="2",
        help="Scale factor (between 1 and 4)",
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
            result = perform_image_upscale(**kwargs)
        except llm.NeedsKeyException as e:
            handle_cli_error(e)
        except VeniceAPIError as e:
            handle_cli_error(e)
        except ValueError as e:
            handle_cli_error(e)
        except httpx.RequestError as e:
            handle_cli_error(e)
        saved_path = write_upscaled_image(result)
        click.echo(f"Upscaled image saved to {saved_path}")

    return upscale
