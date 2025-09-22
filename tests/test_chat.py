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


def test_cli_venice_parameters_registration(
    cli_runner, monkeypatch, mock_venice_api_key
):
    """Test that venice parameter options are registered."""
    from llm import cli as llm_cli

    # Verify Venice parameters are present in the help text
    result = cli_runner.invoke(llm_cli.cli, ["prompt", "--help"])
    assert result.exit_code == 0
    assert "--no-venice-system-prompt" in result.output
    assert "--web-search" in result.output
    assert "--character" in result.output
    assert "--strip-thinking-response" in result.output
    assert "--disable-thinking" in result.output

    # Verify Venice parameters are present in the help text
    result = cli_runner.invoke(llm_cli.cli, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--no-venice-system-prompt" in result.output
    assert "--web-search" in result.output
    assert "--character" in result.output
    assert "--strip-thinking-response" in result.output
    assert "--disable-thinking" in result.output


def test_venice_parameters_validation():
    """Test validation of thinking parameter values."""
    # Test JSON string handling
    options = VeniceChatOptions(
        extra_body='{"venice_parameters": {"disable_thinking": true}}'
    )
    assert options.extra_body["venice_parameters"]["disable_thinking"] is True

    # Test invalid JSON string
    with pytest.raises(ValueError, match="Invalid JSON"):
        VeniceChatOptions(extra_body='{"venice_parameters": {"invalid": json}}')


def test_cli_thinking_parameters(cli_runner, monkeypatch):
    """Test that CLI properly accepts thinking parameters."""
    from llm import cli as llm_cli
    from unittest.mock import patch, MagicMock

    monkeypatch.setenv("LLM_VENICE_KEY", "test-venice-key")
    mock_response = MagicMock()
    mock_response.text = lambda: "Mock response"
    mock_response.usage = lambda: (10, 5, 15)
    with patch.object(VeniceChat, "prompt", return_value=mock_response):
        # CLI accepts --strip-thinking-response
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/qwen3-4b",
                "--strip-thinking-response",
                "--no-log",
                "Test prompt 1",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"
        # CLI accepts --disable-thinking
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/qwen3-4b",
                "--disable-thinking",
                "--no-log",
                "Test prompt 2",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"
        # CLI accepts both parameters
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/qwen3-4b",
                "--strip-thinking-response",
                "--disable-thinking",
                "--no-log",
                "Test prompt 3",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"


def test_thinking_parameters_build_kwargs():
    """Test that thinking parameters are processed correctly in build_kwargs."""
    chat = VeniceChat(
        model_id="venice/qwen3-235b",
        model_name="qwen3-235b",
        api_base="https://api.venice.ai/api/v1",
    )

    # Test single parameter: strip_thinking_response
    options = VeniceChatOptions(
        extra_body={"venice_parameters": {"strip_thinking_response": True}}
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert "extra_body" in kwargs, "extra_body should be present in kwargs"
    assert "venice_parameters" in kwargs["extra_body"], (
        "venice_parameters should be in extra_body"
    )
    assert (
        kwargs["extra_body"]["venice_parameters"]["strip_thinking_response"] is True
    ), "strip_thinking_response should be True"

    # Test with streaming enabled
    kwargs_stream = chat.build_kwargs(prompt, stream=True)
    assert "extra_body" in kwargs_stream, "extra_body should be present when streaming"
    assert (
        kwargs_stream["extra_body"]["venice_parameters"]["strip_thinking_response"]
        is True
    ), "strip_thinking_response should be preserved when streaming"

    # Test single parameter: disable_thinking
    options = VeniceChatOptions(
        extra_body={"venice_parameters": {"disable_thinking": True}}
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert kwargs["extra_body"]["venice_parameters"]["disable_thinking"] is True, (
        "disable_thinking should be True"
    )

    # Test both parameters together
    options = VeniceChatOptions(
        extra_body={
            "venice_parameters": {
                "strip_thinking_response": True,
                "disable_thinking": False,
            }
        }
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    venice_params = kwargs["extra_body"]["venice_parameters"]
    assert venice_params["strip_thinking_response"] is True, (
        "strip_thinking_response should be True when combined"
    )
    assert venice_params["disable_thinking"] is False, (
        "disable_thinking should be False when explicitly set"
    )

    # Test preservation of other extra_body fields
    options = VeniceChatOptions(
        extra_body={
            "custom_field": "preserved",
            "venice_parameters": {"strip_thinking_response": True},
        }
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert kwargs["extra_body"]["custom_field"] == "preserved", (
        "Other extra_body fields should be preserved"
    )
    assert (
        kwargs["extra_body"]["venice_parameters"]["strip_thinking_response"] is True
    ), "venice_parameters should coexist with other fields"

    # Test empty venice_parameters
    options = VeniceChatOptions(extra_body={"venice_parameters": {}})
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert "venice_parameters" in kwargs["extra_body"], (
        "Empty venice_parameters should still be included"
    )
    assert kwargs["extra_body"]["venice_parameters"] == {}, (
        "Empty venice_parameters should remain empty"
    )

    # Test without extra_body
    prompt = Prompt(prompt="Test", model=chat)
    kwargs = chat.build_kwargs(prompt, stream=False)

    # Should not raise an error and should return a dict (may be empty)
    assert isinstance(kwargs, dict), (
        "build_kwargs should return a dict even without extra_body"
    )
    # When no options are provided, kwargs may be empty
    assert "extra_body" not in kwargs or "venice_parameters" not in kwargs.get(
        "extra_body", {}
    ), "venice_parameters should not be added when not specified in options"


def test_venice_parameters_edge_cases():
    """Test edge cases and validation for venice_parameters."""
    chat = VeniceChat(
        model_id="venice/qwen3-4b",
        model_name="qwen3-4b",
        api_base="https://api.venice.ai/api/v1",
    )

    # Test with None extra_body
    options = VeniceChatOptions(extra_body=None)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)
    # Should not raise an error and should not have extra_body key
    assert "extra_body" not in kwargs, (
        "extra_body key should not exist when set to None"
    )

    # Test with nested structure preservation
    options = VeniceChatOptions(
        extra_body={
            "venice_parameters": {
                "strip_thinking_response": True,
            },
            "other": {"structure": "preserved"},
        }
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert kwargs["extra_body"]["other"]["structure"] == "preserved", (
        "Other structures should be preserved"
    )
