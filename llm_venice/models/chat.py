"""Venice chat model implementation."""

import json
from typing import Optional, Union

from llm.default_plugins.openai_models import Chat
from pydantic import Field, field_validator

from llm_venice.constants import VENICE_KEY_NAME, VENICE_KEY_ENV_VAR


class VeniceChatOptions(Chat.Options):
    """Options for Venice chat models."""

    extra_body: Optional[Union[dict, str]] = Field(
        description=(
            "Additional JSON properties to include in the request body. "
            "When provided via CLI, must be a valid JSON string."
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


class VeniceChat(Chat):
    """Venice AI chat model."""

    needs_key = VENICE_KEY_NAME
    key_env_var = VENICE_KEY_ENV_VAR
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

        return kwargs
