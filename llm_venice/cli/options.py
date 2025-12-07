"""Venice-specific CLI option processing."""

import click
import llm


def process_venice_options(kwargs):
    """
    Helper to process Venice-specific CLI flags and convert them to typed
    VeniceChatOptions values. The Venice model class will package these into
    extra_body/venice_parameters during request build.

    Args:
        kwargs: Command arguments dictionary

    Returns:
        Modified kwargs with Venice options processed
    """
    no_venice_system_prompt = kwargs.pop("no_venice_system_prompt", False)
    web_search = kwargs.pop("web_search", False)
    web_citations = kwargs.pop("web_citations", False)
    include_search_results_in_stream = kwargs.pop("include_search_results_in_stream", False)
    character = kwargs.pop("character", None)
    strip_thinking_response = kwargs.pop("strip_thinking_response", False)
    disable_thinking = kwargs.pop("disable_thinking", False)
    options = list(kwargs.get("options", []))
    model_id = kwargs.get("model_id")

    if model_id and model_id.startswith("venice/"):
        model = llm.get_model(model_id)

        # Validate capability for web search early for a better UX
        if (web_search or web_citations or include_search_results_in_stream) and not getattr(
            model, "supports_web_search", False
        ):
            raise click.ClickException(f"Model {model_id} does not support web search")

        # Web citations/search result streaming require web search to be enabled
        # (if left unspecified, API default is "enable_web_search off")
        if (web_citations or include_search_results_in_stream) and web_search not in (
            "on",
            "auto",
        ):
            raise click.ClickException(
                "enable_web_search must be set to 'on' or 'auto' to use web citations or include search results in stream"
            )

        # Map CLI flags to typed VeniceChatOptions fields
        if no_venice_system_prompt:
            options.append(("include_venice_system_prompt", False))
        if web_search:
            options.append(("enable_web_search", web_search))
        if web_citations:
            options.append(("enable_web_citations", True))
        if include_search_results_in_stream:
            options.append(("include_search_results_in_stream", True))
        if character:
            options.append(("character_slug", character))
        if strip_thinking_response:
            options.append(("strip_thinking_response", True))
        if disable_thinking:
            options.append(("disable_thinking", True))

        if options:
            kwargs["options"] = options

    return kwargs
