"""Characters command for Venice CLI."""

import click
import llm

from llm_venice.client import VeniceClient
from llm_venice.utils import write_json_atomic
from llm_venice.errors import VeniceError
from llm_venice.cli.errors import handle_cli_error


def create_characters_command():
    """
    Create the characters command.

    Returns:
        Click command for listing characters
    """

    @click.command(name="characters")
    @click.option(
        "--web-enabled",
        type=click.Choice(["true", "false"]),
        help="Filter by web-enabled status",
    )
    @click.option("--adult", type=click.Choice(["true", "false"]), help="Filter by adult category")
    def characters(web_enabled, adult):
        """List public characters."""
        try:
            with VeniceClient() as client:
                characters_data = client._json_operation(
                    "characters",
                    web_enabled=None if web_enabled is None else web_enabled == "true",
                    adult=None if adult is None else adult == "true",
                )
            path = llm.user_dir() / "venice_characters.json"
            write_json_atomic(path, characters_data)
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        characters_count = len(characters_data.get("data", []))
        click.echo(f"{characters_count} characters saved to {path}", err=True)

    return characters
