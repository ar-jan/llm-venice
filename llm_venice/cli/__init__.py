"""CLI modules for Venice plugin."""

from llm_venice.cli.venice_group import create_venice_group


def register_venice_commands(cli):
    """
    Register the venice command group.

    Args:
        cli: The LLM CLI application
    """
    cli.add_command(create_venice_group())
