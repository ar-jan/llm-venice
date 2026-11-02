"""Main venice CLI command group."""

import click

from llm_venice.catalog import refresh_catalog, catalog_path
from llm_venice.errors import VeniceError
from llm_venice.cli.api_keys import create_api_keys_group
from llm_venice.cli.characters import create_characters_command
from llm_venice.cli.errors import handle_cli_error
from llm_venice.cli.search import create_search_command
from llm_venice.cli.tts import create_tts_command
from llm_venice.cli.upscale import create_upscale_command


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
        try:
            models = refresh_catalog()
        except (ValueError, VeniceError, OSError) as e:
            handle_cli_error(e)
        path = catalog_path()
        click.echo(f"{len(models)} models saved to {path}", err=True)

    venice.add_command(create_api_keys_group())
    venice.add_command(create_characters_command())
    venice.add_command(create_search_command())
    venice.add_command(create_tts_command())
    venice.add_command(create_upscale_command())

    return venice
