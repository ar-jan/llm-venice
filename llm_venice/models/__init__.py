"""Venice model implementations."""

import json

import click
import llm

from llm_venice.models.chat import VeniceChat
from llm_venice.models.image import VeniceImage
from llm_venice.constants import VENICE_API_BASE
from llm_venice.api.refresh import refresh_models


def register_venice_models(register):
    """
    Register all Venice models with the LLM plugin system.

    Args:
        register: The LLM registration function
    """
    venice_models_path = llm.user_dir() / "venice_models.json"
    models = None

    # Prefer cached models so it works without a stored key
    if venice_models_path.exists():
        models = json.loads(venice_models_path.read_text())
    else:
        # Only hit the API if we already have a key configured; otherwise warn
        key = llm.get_key("", "venice", "LLM_VENICE_KEY")
        if key:
            models = refresh_models()
        else:
            click.echo(
                "Skipped refreshing Venice models: run 'llm venice refresh' after configuring LLM_VENICE_KEY or providing a key.",
                err=True,
            )
            return

    for model in models:
        model_id = model["id"]
        capabilities = model.get("model_spec", {}).get("capabilities", {})

        if model.get("type") == "text":
            model_instance = VeniceChat(
                model_id=f"venice/{model_id}",
                model_name=model_id,
                api_base=VENICE_API_BASE,
                can_stream=True,
                vision=capabilities.get("supportsVision", False),
                supports_schema=capabilities.get("supportsResponseSchema", False),
                supports_tools=capabilities.get("supportsFunctionCalling", False),
            )
            # Venice-specific capabilities added as instance attributes
            model_instance.supports_web_search = capabilities.get("supportsWebSearch", False)
            register(model_instance)
        elif model.get("type") == "image":
            register(VeniceImage(model_id=model_id, model_name=model_id))


__all__ = ["VeniceChat", "VeniceImage", "register_venice_models"]
