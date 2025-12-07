"""Venice chat model implementation."""

import json
from typing import List, Optional, Union

import llm
from llm.default_plugins.openai_models import Chat
from llm_venice.constants import VENICE_PARAMETERS, NON_OPENAI_COMPATIBLE_PARAMS
from pydantic import Field, field_validator


class VeniceChatOptions(Chat.Options):
    """Options for Venice chat models."""

    # Non-standard generation parameters (top-level in extra_body)
    min_p: Optional[float] = Field(
        description=(
            "Sets a minimum probability threshold for token selection. "
            "Tokens with probabilities below this value are filtered out."
        ),
        ge=0,
        le=1,
        default=None,
    )
    top_k: Optional[int] = Field(
        description=(
            "The number of highest probability vocabulary tokens to keep for top-k-filtering."
        ),
        ge=0,
        default=None,
    )
    repetition_penalty: Optional[float] = Field(
        description=(
            "The parameter for repetition penalty. 1.0 means no penalty. "
            "Values > 1.0 discourage repetition."
        ),
        ge=0,
        default=None,
    )
    stop_token_ids: Optional[Union[List[int], str]] = Field(
        description=(
            "Array of token IDs where the API will stop generating further tokens. "
            "When provided via CLI, pass as JSON array string: '[151643, 151645]'"
        ),
        default=None,
    )

    # Venice-specific parameters (extra_body.venice_parameters)
    include_venice_system_prompt: Optional[bool] = Field(
        description=(
            "Whether to include Venice-supplied default system prompts alongside your own."
        ),
        default=None,
    )
    enable_web_search: Optional[str] = Field(
        description=("Enable web search: one of 'on', 'off', or 'auto'."),
        default=None,
    )
    enable_web_scraping: Optional[bool] = Field(
        description=("Enable scraping of URLs in the latest user message using."),
        default=None,
    )
    enable_web_citations: Optional[bool] = Field(
        description=(
            "When web search is enabled, request that the LLM cite its sources using ^index^ or ^i,j^ superscript format."
        ),
        default=None,
    )
    include_search_results_in_stream: Optional[bool] = Field(
        description=(
            "Experimental: Include search results in the stream as the first emitted chunk."
        ),
        default=None,
    )
    character_slug: Optional[str] = Field(
        description=("Public character slug to use (e.g. 'alan-watts')."),
        default=None,
    )
    strip_thinking_response: Optional[bool] = Field(
        description=("Strip <think></think> blocks from the response (reasoning models)."),
        default=None,
    )
    disable_thinking: Optional[bool] = Field(
        description=("Disable thinking and strip <think></think> blocks (reasoning models)."),
        default=None,
    )

    @field_validator("enable_web_search")
    def validate_enable_web_search(cls, value):
        if value is None:
            return None
        if value not in ("on", "off", "auto"):
            raise ValueError("enable_web_search must be one of 'on', 'off', or 'auto'")
        return value

    @field_validator("stop_token_ids")
    def validate_stop_token_ids(cls, stop_token_ids):
        """Validate and parse stop_token_ids parameter."""
        if stop_token_ids is None:
            return None

        if isinstance(stop_token_ids, str):
            try:
                parsed = json.loads(stop_token_ids)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in stop_token_ids string")

            if not isinstance(parsed, list):
                raise ValueError("stop_token_ids must be an array")

            # Validate all elements are integers
            if not all(isinstance(x, int) for x in parsed):
                raise ValueError("All elements in stop_token_ids must be integers")

            return parsed

        if isinstance(stop_token_ids, list):
            # Validate all elements are integers
            if not all(isinstance(x, int) for x in stop_token_ids):
                raise ValueError("All elements in stop_token_ids must be integers")
            return stop_token_ids

        raise ValueError("stop_token_ids must be a list or JSON array string")


class VeniceChat(Chat):
    """Venice AI chat model."""

    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    supports_web_search = False

    def __str__(self):
        return f"Venice Chat: {self.model_id}"

    class Options(VeniceChatOptions):
        pass

    def build_kwargs(self, prompt, stream):
        """Build kwargs for the API request, modifying JSON schema parameters."""
        kwargs = super().build_kwargs(prompt, stream)

        # Venice requires strict mode and no additional properties for JSON schema
        if "response_format" in kwargs and kwargs["response_format"].get("type") == "json_schema":
            kwargs["response_format"]["json_schema"]["strict"] = True
            kwargs["response_format"]["json_schema"]["schema"]["additionalProperties"] = False

        # Move non-openai-compatible generation parameters into extra_body
        non_openai_compatible_params = {}

        for param in NON_OPENAI_COMPATIBLE_PARAMS:
            if param in kwargs:
                non_openai_compatible_params[param] = kwargs.pop(param)

        venice_parameters = {}
        for key in VENICE_PARAMETERS:
            if key in kwargs and kwargs[key] is not None:
                venice_parameters[key] = kwargs.pop(key)

        web_search_requested = venice_parameters.get("enable_web_search")
        web_citations_requested = venice_parameters.get("enable_web_citations")
        include_results_requested = venice_parameters.get("include_search_results_in_stream")

        # Capability guard for web-search-related features
        if (
            web_search_requested or web_citations_requested or include_results_requested
        ) and not getattr(self, "supports_web_search", False):
            raise llm.ModelError(f"Model {self.model_id} does not support web search")

        if (web_citations_requested or include_results_requested) and web_search_requested not in (
            "on",
            "auto",
        ):
            raise llm.ModelError(
                "enable_web_search must be set to 'on' or 'auto' when using web citations or including search results in stream"
            )

        # Build extra_body
        if non_openai_compatible_params or venice_parameters:
            kwargs["extra_body"] = {}
            # Non-standard parameters go top-level inside extra_body
            if non_openai_compatible_params:
                kwargs["extra_body"].update(non_openai_compatible_params)
            # Venice-specific parameters go in extra_body.venice_parameters
            if venice_parameters:
                kwargs["extra_body"]["venice_parameters"] = venice_parameters

        return kwargs
