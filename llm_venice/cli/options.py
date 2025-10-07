"""Venice-specific CLI option processing."""

import click
import llm


def process_venice_options(kwargs):
    """
    Helper to process venice-specific options and convert them to extra_body.

    Args:
        kwargs: Command arguments dictionary

    Returns:
        Modified kwargs with Venice options processed
    """
    no_venice_system_prompt = kwargs.pop("no_venice_system_prompt", False)
    web_search = kwargs.pop("web_search", False)
    character = kwargs.pop("character", None)
    strip_thinking_response = kwargs.pop("strip_thinking_response", False)
    disable_thinking = kwargs.pop("disable_thinking", False)
    options = list(kwargs.get("options", []))
    model_id = kwargs.get("model_id")

    if model_id and model_id.startswith("venice/"):
        model = llm.get_model(model_id)
        venice_params = {}

        if no_venice_system_prompt:
            venice_params["include_venice_system_prompt"] = False

        if web_search:
            if not getattr(model, "supports_web_search", False):
                raise click.ClickException(
                    f"Model {model_id} does not support web search"
                )
            venice_params["enable_web_search"] = web_search

        if character:
            venice_params["character_slug"] = character

        if strip_thinking_response:
            venice_params["strip_thinking_response"] = True

        if disable_thinking:
            venice_params["disable_thinking"] = True

        if venice_params:
            # If a Venice option is used, any `-o extra_body value` is overridden here.
            # TODO: Would prefer to remove the extra_body CLI option, but
            # the implementation is required for venice_parameters.
            options.append(("extra_body", {"venice_parameters": venice_params}))
            kwargs["options"] = options

    return kwargs
