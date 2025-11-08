"""Command hooks to extend prompt and chat commands with Venice options."""

import click

from llm_venice.constants import VENICE_PARAMETERS_CLI
from llm_venice.cli.options import process_venice_options


def install_command_hooks(cli):
    """
    Captures and extends prompt/chat commands with Venice options.
    Must be called after all other CLI setup.

    Args:
        cli: The LLM CLI application
    """
    # Remove and store the original prompt and chat commands
    # in order to add them back with custom cli options
    original_prompt = cli.commands.pop("prompt")
    original_chat = cli.commands.pop("chat")

    # Create new prompt command
    @cli.command(name="prompt")
    @click.option(
        "--no-venice-system-prompt",
        is_flag=True,
        help="Disable Venice AI's default system prompt",
    )
    @click.option(
        "--web-search",
        type=click.Choice(["auto", "on", "off"]),
        help="Enable web search",
    )
    @click.option(
        "--character",
        help="Use a Venice AI public character (e.g. 'alan-watts')",
    )
    @click.option(
        "--strip-thinking-response",
        is_flag=True,
        help="Strip <think></think> blocks from the response (for reasoning models)",
    )
    @click.option(
        "--disable-thinking",
        is_flag=True,
        help="Disable thinking and strip <think></think> blocks (for reasoning models)",
    )
    @click.pass_context
    def new_prompt(
        ctx,
        no_venice_system_prompt,
        web_search,
        character,
        strip_thinking_response,
        disable_thinking,
        **kwargs
    ):
        """Execute a prompt"""
        kwargs = process_venice_options(
            {
                **kwargs,
                "no_venice_system_prompt": no_venice_system_prompt,
                "web_search": web_search,
                "character": character,
                "strip_thinking_response": strip_thinking_response,
                "disable_thinking": disable_thinking,
            }
        )
        return ctx.invoke(original_prompt, **kwargs)

    # Create new chat command
    @cli.command(name="chat")
    @click.option(
        "--no-venice-system-prompt",
        is_flag=True,
        help="Disable Venice AI's default system prompt",
    )
    @click.option(
        "--web-search",
        type=click.Choice(["auto", "on", "off"]),
        help="Enable web search",
    )
    @click.option(
        "--character",
        help="Use a Venice AI character (e.g. 'alan-watts')",
    )
    @click.option(
        "--strip-thinking-response",
        is_flag=True,
        help="Strip <think></think> blocks from the response (for reasoning models)",
    )
    @click.option(
        "--disable-thinking",
        is_flag=True,
        help="Disable thinking and strip <think></think> blocks (for reasoning models)",
    )
    @click.pass_context
    def new_chat(
        ctx,
        no_venice_system_prompt,
        web_search,
        character,
        strip_thinking_response,
        disable_thinking,
        **kwargs
    ):
        """Hold an ongoing chat with a model"""
        kwargs = process_venice_options(
            {
                **kwargs,
                "no_venice_system_prompt": no_venice_system_prompt,
                "web_search": web_search,
                "character": character,
                "strip_thinking_response": strip_thinking_response,
                "disable_thinking": disable_thinking,
            }
        )
        return ctx.invoke(original_chat, **kwargs)

    # Copy over all params from original commands
    for param in original_prompt.params:
        if param.name not in VENICE_PARAMETERS_CLI:
            new_prompt.params.append(param)

    for param in original_chat.params:
        if param.name not in VENICE_PARAMETERS_CLI:
            new_chat.params.append(param)
