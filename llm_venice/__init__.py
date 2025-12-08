"""LLM Venice plugin for Venice AI models."""

import llm

from llm_venice.cli import register_venice_commands
from llm_venice.models import register_venice_models

# Public API exports
from llm_venice.models.chat import VeniceChat, VeniceChatOptions
from llm_venice.models.image import (
    VeniceImage,
    VeniceImageOptions,
    generate_image_result,
    ImageGenerationResult,
    save_image_result,
)
from llm_venice.api.upscale import UpscaleResult, perform_image_upscale, write_upscaled_image
from llm_venice.api.refresh import fetch_models, persist_models
from llm_venice.api.characters import list_characters, persist_characters
from llm_venice.api.keys import (
    list_api_keys,
    get_rate_limits,
    get_rate_limits_log,
    create_api_key,
    delete_api_key,
)


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
    "ImageGenerationResult",
    "generate_image_result",
    "save_image_result",
    "UpscaleResult",
    "perform_image_upscale",
    "write_upscaled_image",
    "fetch_models",
    "persist_models",
    "list_characters",
    "persist_characters",
    "list_api_keys",
    "get_rate_limits",
    "get_rate_limits_log",
    "create_api_key",
    "delete_api_key",
]
