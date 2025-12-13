"""Characters command for Venice CLI."""

import click
import httpx

from llm_venice.api.characters import list_characters, persist_characters
from llm_venice.api.errors import VeniceAPIError
from llm_venice.cli.errors import handle_cli_error
from llm_venice.utils import get_venice_key


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
        key = get_venice_key(click_exceptions=True)
        try:
            characters_data = list_characters(key, web_enabled=web_enabled, adult=adult)
        except (VeniceAPIError, httpx.RequestError) as e:
            handle_cli_error(e)
        path = persist_characters(characters_data)
        characters_count = len(characters_data.get("data", []))
        click.echo(f"{characters_count} characters saved to {path}", err=True)

    return characters
