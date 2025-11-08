"""Venice chat model implementation."""

import json
from typing import List, Optional, Union

from llm.default_plugins.openai_models import Chat
from pydantic import Field, field_validator


class VeniceChatOptions(Chat.Options):
    """Options for Venice chat models."""

    extra_body: Optional[Union[dict, str]] = Field(
        description=(
            "Additional JSON properties to include in the request body. "
            "When provided via CLI, must be a valid JSON string."
        ),
        default=None,
    )
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
            "The number of highest probability vocabulary tokens to keep for "
            "top-k-filtering."
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

    @field_validator("extra_body")
    def validate_extra_body(cls, extra_body):
        """Validate and parse extra_body parameter."""
        if extra_body is None:
            return None

        if isinstance(extra_body, str):
            try:
                return json.loads(extra_body)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in extra_body string")

        if not isinstance(extra_body, dict):
            raise ValueError("extra_body must be a dictionary")

        return extra_body

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
        if (
            "response_format" in kwargs
            and kwargs["response_format"].get("type") == "json_schema"
        ):
            kwargs["response_format"]["json_schema"]["strict"] = True
            kwargs["response_format"]["json_schema"]["schema"][
                "additionalProperties"
            ] = False

        # Move Venice-specific parameters into extra_body
        # The OpenAI client doesn't accept these as top-level parameters
        venice_specific_params = ["min_p", "top_k", "repetition_penalty", "stop_token_ids"]
        params_to_move = {}

        for param in venice_specific_params:
            if param in kwargs:
                params_to_move[param] = kwargs.pop(param)

        # If we have parameters to move, merge them into extra_body
        if params_to_move:
            if "extra_body" not in kwargs:
                kwargs["extra_body"] = {}
            # Merge with existing extra_body, preserving existing fields
            kwargs["extra_body"].update(params_to_move)

        return kwargs
