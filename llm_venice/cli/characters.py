"""Characters command for Venice CLI."""

import json

import click
import httpx
import llm

from llm_venice.constants import ENDPOINT_CHARACTERS
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
    @click.option(
        "--adult", type=click.Choice(["true", "false"]), help="Filter by adult category"
    )
    def characters(web_enabled, adult):
        """List public characters."""
        key = get_venice_key()
        headers = {"Authorization": f"Bearer {key}"}

        params = {
            k: v
            for k, v in {"isWebEnabled": web_enabled, "isAdult": adult}.items()
            if v
        }

        response = httpx.get(
            ENDPOINT_CHARACTERS,
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        characters_data = response.json()

        path = llm.user_dir() / "venice_characters.json"
        path.write_text(json.dumps(characters_data, indent=4))
        characters_count = len(characters_data.get("data", []))
        click.echo(f"{characters_count} characters saved to {path}", err=True)

    return characters
