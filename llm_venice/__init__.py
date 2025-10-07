"""LLM Venice plugin for Venice AI models."""

import llm

from llm_venice.cli import register_venice_commands
from llm_venice.models import register_venice_models

# Public API exports
from llm_venice.models.chat import VeniceChat, VeniceChatOptions
from llm_venice.models.image import VeniceImage, VeniceImageOptions
from llm_venice.api.upscale import image_upscale
from llm_venice.api.refresh import refresh_models


@llm.hookimpl
def register_commands(cli):
    """
    Register Venice CLI commands with the LLM CLI.

    Args:
        cli: The LLM CLI application
    """
    register_venice_commands(cli)


@llm.hookimpl
def register_models(register):
    """
    Register Venice models with the LLM plugin system.

    Args:
        register: The LLM model registration function
    """
    register_venice_models(register)


__all__ = [
    "register_commands",
    "register_models",
    "VeniceChat",
    "VeniceChatOptions",
    "VeniceImage",
    "VeniceImageOptions",
    "image_upscale",
    "refresh_models",
]
