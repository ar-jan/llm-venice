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

    # Invalid JSON string - Pydantic wraps ValueError into ValidationError
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(extra_body='{"invalid json')
    assert "Invalid JSON" in str(exc_info.value)

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

    # Test invalid JSON string - Pydantic wraps ValueError into ValidationError
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(extra_body='{"venice_parameters": {"invalid": json}}')
    assert "Invalid JSON" in str(exc_info.value)


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


def test_new_parameters_validation():
    """Test validation of new parameters: min_p, top_k, repetition_penalty, stop_token_ids."""
    # Test valid min_p values
    options = VeniceChatOptions(min_p=0.05)
    assert options.min_p == 0.05

    options = VeniceChatOptions(min_p=0.0)
    assert options.min_p == 0.0

    options = VeniceChatOptions(min_p=1.0)
    assert options.min_p == 1.0

    # Test invalid min_p values
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(min_p=-0.1)
    assert "greater than or equal to 0" in str(exc_info.value).lower()

    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(min_p=1.1)
    assert "less than or equal to 1" in str(exc_info.value).lower()

    # Test valid top_k values
    options = VeniceChatOptions(top_k=40)
    assert options.top_k == 40

    options = VeniceChatOptions(top_k=0)
    assert options.top_k == 0

    # Test invalid top_k values
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(top_k=-1)
    assert "greater than or equal to 0" in str(exc_info.value).lower()

    # Test valid repetition_penalty values
    options = VeniceChatOptions(repetition_penalty=1.2)
    assert options.repetition_penalty == 1.2

    options = VeniceChatOptions(repetition_penalty=1.0)
    assert options.repetition_penalty == 1.0

    options = VeniceChatOptions(repetition_penalty=0.0)
    assert options.repetition_penalty == 0.0

    # Test invalid repetition_penalty values
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(repetition_penalty=-0.1)
    assert "greater than or equal to 0" in str(exc_info.value).lower()

    # Test valid stop_token_ids values (list)
    options = VeniceChatOptions(stop_token_ids=[151643, 151645])
    assert options.stop_token_ids == [151643, 151645]

    options = VeniceChatOptions(stop_token_ids=[])
    assert options.stop_token_ids == []

    # Test valid stop_token_ids values (JSON string)
    options = VeniceChatOptions(stop_token_ids="[151643, 151645]")
    assert options.stop_token_ids == [151643, 151645]

    options = VeniceChatOptions(stop_token_ids="[]")
    assert options.stop_token_ids == []

    # Test invalid stop_token_ids values (invalid JSON)
    # Pydantic validates type before our validator runs,
    # and wraps ValueError into ValidationError
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(stop_token_ids="[not valid json")
    assert "Invalid JSON" in str(exc_info.value)

    # Test invalid stop_token_ids values (not an array)
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(stop_token_ids='{"not": "array"}')
    assert "must be an array" in str(exc_info.value)

    # Test invalid stop_token_ids values (not integers in list)
    with pytest.raises(ValidationError):
        VeniceChatOptions(stop_token_ids=[1.5, 2.5])

    # Test invalid stop_token_ids values (non-integers in JSON string)
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(stop_token_ids="[1.5, 2.5]")
    assert "must be integers" in str(exc_info.value)


def test_new_parameters_build_kwargs():
    """Test that new parameters are moved into extra_body in build_kwargs output."""
    chat = VeniceChat(
        model_id="venice/test-model",
        model_name="test-model",
        api_base="https://api.venice.ai/api/v1",
    )

    # Test min_p parameter is in extra_body, not top-level
    options = VeniceChatOptions(min_p=0.05)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)
    assert "min_p" not in kwargs, "min_p should not be at top level"
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["min_p"] == 0.05

    # Test top_k parameter
    options = VeniceChatOptions(top_k=40)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)
    assert "top_k" not in kwargs, "top_k should not be at top level"
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["top_k"] == 40

    # Test repetition_penalty parameter
    options = VeniceChatOptions(repetition_penalty=1.2)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)
    assert "repetition_penalty" not in kwargs, (
        "repetition_penalty should not be at top level"
    )
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["repetition_penalty"] == 1.2

    # Test stop_token_ids parameter
    options = VeniceChatOptions(stop_token_ids=[151643, 151645])
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)
    assert "stop_token_ids" not in kwargs, "stop_token_ids should not be at top level"
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["stop_token_ids"] == [151643, 151645]

    # Test multiple parameters together
    options = VeniceChatOptions(
        min_p=0.05,
        top_k=40,
        repetition_penalty=1.2,
        stop_token_ids=[151643, 151645],
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["min_p"] == 0.05
    assert kwargs["extra_body"]["top_k"] == 40
    assert kwargs["extra_body"]["repetition_penalty"] == 1.2
    assert kwargs["extra_body"]["stop_token_ids"] == [151643, 151645]

    # Test that None values are not included in extra_body
    options = VeniceChatOptions(min_p=None)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)
    assert "min_p" not in kwargs
    assert "extra_body" not in kwargs or "min_p" not in kwargs.get("extra_body", {})

    # Test combination with existing parameters (temperature, max_tokens)
    # These should stay at top level
    options = VeniceChatOptions(
        min_p=0.05,
        top_k=40,
        temperature=0.7,
        max_tokens=100,
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)
    assert kwargs["extra_body"]["min_p"] == 0.05
    assert kwargs["extra_body"]["top_k"] == 40
    assert kwargs["temperature"] == 0.7, "temperature should stay at top level"
    assert kwargs["max_tokens"] == 100, "max_tokens should stay at top level"


def test_new_parameters_with_streaming():
    """Test that new parameters work correctly with streaming enabled."""
    chat = VeniceChat(
        model_id="venice/test-model",
        model_name="test-model",
        api_base="https://api.venice.ai/api/v1",
    )

    options = VeniceChatOptions(
        min_p=0.05,
        top_k=40,
        repetition_penalty=1.2,
        stop_token_ids=[151643, 151645],
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=True)

    # Verify all parameters are in extra_body when streaming
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["min_p"] == 0.05
    assert kwargs["extra_body"]["top_k"] == 40
    assert kwargs["extra_body"]["repetition_penalty"] == 1.2
    assert kwargs["extra_body"]["stop_token_ids"] == [151643, 151645]
    assert "stream_options" in kwargs
    assert kwargs["stream_options"]["include_usage"] is True


def test_new_parameters_defaults():
    """Test that new parameters default to None when not specified."""
    options = VeniceChatOptions()
    assert options.min_p is None
    assert options.top_k is None
    assert options.repetition_penalty is None
    assert options.stop_token_ids is None


def test_new_parameters_merge_with_extra_body():
    """Test that new parameters merge correctly with existing extra_body."""
    chat = VeniceChat(
        model_id="venice/test-model",
        model_name="test-model",
        api_base="https://api.venice.ai/api/v1",
    )

    # Test merging with venice_parameters
    options = VeniceChatOptions(
        min_p=0.05,
        top_k=40,
        extra_body={"venice_parameters": {"strip_thinking_response": True}},
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert "extra_body" in kwargs
    # Venice parameters should be preserved
    assert kwargs["extra_body"]["venice_parameters"]["strip_thinking_response"] is True
    # New parameters should be added
    assert kwargs["extra_body"]["min_p"] == 0.05
    assert kwargs["extra_body"]["top_k"] == 40

    # Test merging with other custom fields
    options = VeniceChatOptions(
        repetition_penalty=1.2,
        extra_body={"custom_field": "value", "another_field": 123},
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert kwargs["extra_body"]["custom_field"] == "value"
    assert kwargs["extra_body"]["another_field"] == 123
    assert kwargs["extra_body"]["repetition_penalty"] == 1.2

    # Test that all new parameters work together with venice_parameters
    options = VeniceChatOptions(
        min_p=0.05,
        top_k=40,
        repetition_penalty=1.2,
        stop_token_ids=[151643, 151645],
        extra_body={
            "venice_parameters": {
                "strip_thinking_response": True,
                "disable_thinking": False,
            },
            "custom_field": "preserved",
        },
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    # All venice_parameters should be preserved
    assert kwargs["extra_body"]["venice_parameters"]["strip_thinking_response"] is True
    assert kwargs["extra_body"]["venice_parameters"]["disable_thinking"] is False
    # Custom field should be preserved
    assert kwargs["extra_body"]["custom_field"] == "preserved"
    # New parameters should be added
    assert kwargs["extra_body"]["min_p"] == 0.05
    assert kwargs["extra_body"]["top_k"] == 40
    assert kwargs["extra_body"]["repetition_penalty"] == 1.2
    assert kwargs["extra_body"]["stop_token_ids"] == [151643, 151645]


def test_new_parameters_no_extra_body_pollution():
    """Test that new parameters don't pollute extra_body when not specified."""
    chat = VeniceChat(
        model_id="venice/test-model",
        model_name="test-model",
        api_base="https://api.venice.ai/api/v1",
    )

    # Test with only venice_parameters, no new parameters
    options = VeniceChatOptions(
        extra_body={"venice_parameters": {"strip_thinking_response": True}},
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["venice_parameters"]["strip_thinking_response"] is True
    # New parameters should not appear
    assert "min_p" not in kwargs["extra_body"]
    assert "top_k" not in kwargs["extra_body"]
    assert "repetition_penalty" not in kwargs["extra_body"]
    assert "stop_token_ids" not in kwargs["extra_body"]

    # Test with no options at all
    prompt = Prompt(prompt="Test", model=chat)
    kwargs = chat.build_kwargs(prompt, stream=False)

    # extra_body should not exist if there are no parameters to include
    assert "extra_body" not in kwargs


def test_new_parameters_cli_usage(cli_runner, monkeypatch):
    """Test that new parameters work via CLI and don't cause runtime errors."""
    from unittest.mock import patch, MagicMock

    monkeypatch.setenv("LLM_VENICE_KEY", "test-venice-key")

    # Mock the prompt method to capture what kwargs it receives
    mock_response = MagicMock()
    mock_response.text = lambda: "Mock response"
    mock_response.usage = lambda: (10, 5, 15)

    with patch.object(VeniceChat, "prompt", return_value=mock_response):
        from llm import cli as llm_cli

        # Test min_p parameter
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/venice-uncensored",
                "-o",
                "min_p",
                "0.05",
                "--no-log",
                "Test prompt",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"

        # Test top_k parameter
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/venice-uncensored",
                "-o",
                "top_k",
                "40",
                "--no-log",
                "Test prompt",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"

        # Test repetition_penalty parameter
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/venice-uncensored",
                "-o",
                "repetition_penalty",
                "1.2",
                "--no-log",
                "Test prompt",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"

        # Test stop_token_ids parameter with JSON string
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/venice-uncensored",
                "-o",
                "stop_token_ids",
                "[151643, 151645]",
                "--no-log",
                "Test prompt",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"

        # Test multiple parameters together
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/venice-uncensored",
                "-o",
                "min_p",
                "0.05",
                "-o",
                "top_k",
                "40",
                "-o",
                "repetition_penalty",
                "1.2",
                "-o",
                "stop_token_ids",
                "[151643, 151645]",
                "--no-log",
                "Test prompt",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"


def test_new_parameters_request_shape_client_call(monkeypatch):
    """Spy on the OpenAI client call to ensure request shape is correct.

    Verifies that the four Venice-specific parameters do not appear at the
    top-level of the API call and are instead included inside extra_body.
    Also checks that unrelated options (e.g. temperature) remain top-level
    and that stream_options is present when streaming.
    """
    from llm import Prompt
    from llm_venice import VeniceChat, VeniceChatOptions

    captured_kwargs = {}

    class FakeCompletions:
        def create(self, **kwargs):
            # Capture all keyword arguments passed to the API call
            captured_kwargs.update(kwargs)
            # Return an empty iterable for streaming branch
            return []

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    # Patch get_client to return our fake client
    monkeypatch.setattr(VeniceChat, "get_client", lambda self, key: FakeClient())

    chat = VeniceChat(
        model_id="venice/test-model",
        model_name="test-model",
        api_base="https://api.venice.ai/api/v1",
    )

    options = VeniceChatOptions(
        min_p=0.05,
        top_k=40,
        repetition_penalty=1.2,
        stop_token_ids=[151643, 151645],
        temperature=0.7,
        max_tokens=25,
    )
    prompt_obj = Prompt(prompt="Test request shape", model=chat, options=options)

    # Minimal stub response object for execute(); streaming branch will assign response_json
    class StubResponse:
        pass

    # Execute with streaming to take the simpler code path
    list(chat.execute(prompt_obj, stream=True, response=StubResponse()))

    # Ensure our fake client was called with expected shape
    assert "extra_body" in captured_kwargs, "extra_body must be included"
    assert captured_kwargs["extra_body"]["min_p"] == 0.05
    assert captured_kwargs["extra_body"]["top_k"] == 40
    assert captured_kwargs["extra_body"]["repetition_penalty"] == 1.2
    assert captured_kwargs["extra_body"]["stop_token_ids"] == [151643, 151645]

    # These should NOT appear at the top-level
    assert "min_p" not in captured_kwargs
    assert "top_k" not in captured_kwargs
    assert "repetition_penalty" not in captured_kwargs
    assert "stop_token_ids" not in captured_kwargs

    # Unrelated options should remain top-level
    assert captured_kwargs["temperature"] == 0.7
    assert captured_kwargs["max_tokens"] == 25

    # Streaming should add stream_options.include_usage
    assert captured_kwargs.get("stream_options") == {"include_usage": True}

    # Sanity check for other required arguments
    assert isinstance(captured_kwargs.get("messages"), list)
