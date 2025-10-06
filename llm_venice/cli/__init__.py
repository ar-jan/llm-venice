"""CLI modules for Venice plugin."""

from llm_venice.cli.venice_group import create_venice_group
from llm_venice.cli.command_hooks import install_command_hooks


def register_venice_commands(cli):
    """
    Single orchestration point for all CLI registrations.
    Maintains execution order and shared state.

    Args:
        cli: The LLM CLI application
    """
    # Step 1: Create and register venice command group
    venice_group = create_venice_group()
    cli.add_command(venice_group)

    # Step 2: Install prompt/chat hooks (must happen last)
    # This captures and modifies the original commands
    install_command_hooks(cli, venice_group)
