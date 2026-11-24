"""Background removal command for Venice CLI."""

import click

from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceError
from llm_venice.cli.errors import handle_cli_error
from llm_venice.cli.images import ImageSource, output_target, save_image


def create_background_remove_command():
    """
    Create the background-remove command.

    Returns:
        Click command for removing an image's background
    """

    @click.command(name="background-remove")
    @click.argument("image", type=ImageSource())
    @click.option(
        "--output-path",
        "-o",
        type=click.Path(file_okay=True, dir_okay=True, writable=True),
        help="Output path (file or directory)",
    )
    @click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing files")
    @click.option("--anon-user-id", help="ID of your end user, for attribution to providers")
    def background_remove(image, output_path, overwrite, anon_user_id):
        """Remove an image's background, saving a PNG with transparency.

        IMAGE is a local file or an http(s) URL, which Venice fetches. The result is
        saved as <name>_nobg.png next to a local image, or in the current directory
        for a URL. Each request is billed.
        """
        try:
            with VeniceClient() as client:
                result = client.remove_image_background(image, anon_user_id=anon_user_id)
            target = output_target(image, "nobg", result, output_path)
            saved_path = save_image(result, target, overwrite=overwrite)
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        click.echo(f"Image saved to {saved_path}")

    return background_remove
