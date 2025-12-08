"""Main venice CLI command group."""

import click

from llm_venice.cli.api_keys import create_api_keys_group
from llm_venice.cli.characters import create_characters_command
from llm_venice.cli.upscale import create_upscale_command
from llm_venice.api.refresh import refresh_models


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
        refresh_models(use_click_exceptions=True)

    # Add subcommands
    venice.add_command(create_api_keys_group())
    venice.add_command(create_characters_command())
    venice.add_command(create_upscale_command())

    return venice
