"""Model trait and compatibility mapping commands for Venice CLI."""

import json

import click

from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceError
from llm_venice.cli.errors import handle_cli_error

_TYPE_HELP = "Model type, such as text, image or embedding (Venice's default: text)"


def _show(operation: str, model_type: str | None) -> None:
    try:
        with VeniceClient() as client:
            response = client._json_operation(operation, type=model_type)
    except (VeniceError, ValueError, OSError) as e:
        handle_cli_error(e)
    click.echo(json.dumps(response, indent=2))


def create_traits_command():
    """
    Create the traits command.

    Returns:
        Click command showing which model each trait selects
    """

    @click.command(name="traits")
    @click.option("-t", "--type", "model_type", help=_TYPE_HELP + "; all and code also work")
    def traits(model_type):
        """Show which model each trait, such as default or most_uncensored, selects."""
        _show("model_traits", model_type)

    return traits


def create_compatibility_mapping_command():
    """
    Create the compatibility-mapping command.

    Returns:
        Click command showing the Venice model behind other providers' model names
    """

    @click.command(name="compatibility-mapping")
    @click.option("-t", "--type", "model_type", help=_TYPE_HELP)
    def compatibility_mapping(model_type):
        """Show the Venice model that serves each other provider's model name."""
        _show("model_compatibility", model_type)

    return compatibility_mapping
