"""Image styles command for Venice CLI."""

import json

import click

from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceError
from llm_venice.cli.errors import handle_cli_error
from llm_venice.cli.output import clean_text


def create_image_styles_command():
    """
    Create the image-styles command.

    Returns:
        Click command for listing image style presets
    """

    @click.command(name="image-styles")
    @click.option("--json", "as_json", is_flag=True, help="Output the JSON response")
    def image_styles(as_json):
        """List the style presets for image generation, one per line.

        Pass a style to an image model with -o style_preset, for example:

        \b
          llm -m venice/qwen-image-3 -o style_preset "Watercolor" "A windmill"
        """
        try:
            with VeniceClient() as client:
                response = client._json_operation("image_styles")
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        if as_json:
            click.echo(json.dumps(response, indent=2))
        else:
            for style in response["data"]:
                click.echo(clean_text(style, single_line=True))

    return image_styles
