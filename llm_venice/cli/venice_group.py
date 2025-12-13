"""Main venice CLI command group."""

import click
import httpx

from llm_venice.api.refresh import fetch_models, persist_models
from llm_venice.api.errors import VeniceAPIError
from llm_venice.cli.api_keys import create_api_keys_group
from llm_venice.cli.characters import create_characters_command
from llm_venice.cli.errors import handle_cli_error
from llm_venice.cli.upscale import create_upscale_command
from llm_venice.utils import get_venice_key


def create_venice_group():
    """
    Create the main venice command group with all subcommands.

    Returns:
        Click group for venice commands
    """

    @click.group(name="venice")
    def venice():
        """llm-venice plugin commands"""
        pass

    @venice.command(name="refresh")
    def refresh():
        """Refresh the list of models from the Venice API"""
        key = get_venice_key(click_exceptions=True)
        try:
            models = fetch_models(key)
        except (ValueError, VeniceAPIError, httpx.RequestError) as e:
            handle_cli_error(e)
        path = persist_models(models)
        click.echo(f"{len(models)} models saved to {path}", err=True)

    venice.add_command(create_api_keys_group())
    venice.add_command(create_characters_command())
    venice.add_command(create_upscale_command())

    return venice
