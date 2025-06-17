"""Tests for VeniceChat model functionality"""

import pytest
from pydantic import ValidationError
from llm_venice import VeniceChatOptions


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
