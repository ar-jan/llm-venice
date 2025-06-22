"""Tests for VeniceChat model functionality"""

import pytest
from pydantic import ValidationError
from llm import Prompt
from llm_venice import VeniceChat, VeniceChatOptions


def test_venice_chat_options_extra_body_validation():
    """Test that extra_body validation works correctly for both dict and JSON string inputs."""
    # Valid dictionary
    options = VeniceChatOptions(extra_body={"venice_parameters": {"test": "value"}})
    assert options.extra_body == {"venice_parameters": {"test": "value"}}

    # Valid JSON string
    options = VeniceChatOptions(extra_body='{"venice_parameters": {"test": "value"}}')
    assert options.extra_body == {"venice_parameters": {"test": "value"}}

    # Invalid JSON string
    with pytest.raises(ValueError, match="Invalid JSON"):
        VeniceChatOptions(extra_body='{"invalid json')

    # Invalid type - Pydantic raises ValidationError for type mismatches
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(extra_body=["not", "a", "dict"])

    # Error should mention both dict and string type expectations
    error_str = str(exc_info.value).lower()
    assert "dict" in error_str
    assert "string" in error_str


def test_venice_chat_build_kwargs_json_schema():
    """Test that build_kwargs modifies JSON schema responses correctly.

    When a prompt has a schema, the parent class creates a response_format
    with type='json_schema'. VeniceChat modifies this to add
    strict=True and additionalProperties=False.
    """
    chat = VeniceChat(
        model_id="venice/test-model",
        model_name="test-model",
        api_base="https://api.venice.ai/api/v1",
    )

    # Create a schema for json_schema response format
    test_schema = {"type": "object", "properties": {"test": {"type": "string"}}}

    # Create a prompt instance with a schema
    # This will make the parent's build_kwargs add response_format
    prompt = Prompt(
        prompt="Generate a test object",
        model=chat,
        schema=test_schema,
    )

    kwargs = chat.build_kwargs(prompt, stream=False)

    # Verify the parent class created the response_format
    assert "response_format" in kwargs
    assert kwargs["response_format"]["type"] == "json_schema"

    # Verify VeniceChat modifications
    json_schema = kwargs["response_format"]["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["schema"]["additionalProperties"] is False

    # Verify the original schema content is preserved
    assert json_schema["schema"]["type"] == "object"
    assert "test" in json_schema["schema"]["properties"]
