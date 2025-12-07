"""Tests for VeniceChat model functionality"""

import pytest
from pydantic import ValidationError
from llm import Prompt, Response
from llm_venice import VeniceChat, VeniceChatOptions


def test_venice_chat_options_fields():
    """Test that Venice-specific option fields work correctly."""
    # Test venice_parameters fields
    options = VeniceChatOptions(
        strip_thinking_response=True,
        disable_thinking=False,
        include_venice_system_prompt=True,
        character_slug="alan-watts",
    )
    assert options.strip_thinking_response is True
    assert options.disable_thinking is False
    assert options.include_venice_system_prompt is True
    assert options.character_slug == "alan-watts"

    # Test top-level Venice parameters
    options = VeniceChatOptions(
        min_p=0.05,
        top_k=40,
        repetition_penalty=1.1,
    )
    assert options.min_p == 0.05
    assert options.top_k == 40
    assert options.repetition_penalty == 1.1

    # Test web search validation
    options = VeniceChatOptions(enable_web_search="auto")
    assert options.enable_web_search == "auto"

    # Invalid web search value
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(enable_web_search="invalid")
    assert "enable_web_search must be one of" in str(exc_info.value)


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


def test_cli_venice_parameters_registration(cli_runner, monkeypatch, mock_venice_api_key):
    """Test that venice parameter options are registered."""
    from llm import cli as llm_cli

    # Verify Venice parameters are present in the prompt help text
    result = cli_runner.invoke(llm_cli.cli, ["prompt", "--help"])
    assert result.exit_code == 0
    assert "--no-venice-system-prompt" in result.output
    assert "--web-search" in result.output
    assert "--web-scraping" in result.output
    assert "--character" in result.output
    assert "--strip-thinking-response" in result.output
    assert "--disable-thinking" in result.output

    # Verify Venice parameters are present in the chat help text
    result = cli_runner.invoke(llm_cli.cli, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--no-venice-system-prompt" in result.output
    assert "--web-search" in result.output
    assert "--web-scraping" in result.output
    assert "--character" in result.output
    assert "--strip-thinking-response" in result.output
    assert "--disable-thinking" in result.output


def test_venice_parameters_validation():
    """Test validation of Venice parameter values."""
    # Test thinking parameters
    options = VeniceChatOptions(disable_thinking=True)
    assert options.disable_thinking is True

    options = VeniceChatOptions(strip_thinking_response=True)
    assert options.strip_thinking_response is True

    # Test min_p validation (must be between 0 and 1)
    with pytest.raises(ValidationError):
        VeniceChatOptions(min_p=1.5)

    with pytest.raises(ValidationError):
        VeniceChatOptions(min_p=-0.1)


def test_venice_chat_options_invalid_values_raise_validation_errors():
    """Ensure Pydantic validators surface errors for bad option values."""
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(min_p=-0.1)
    assert any(error["loc"] == ("min_p",) for error in exc_info.value.errors())

    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(enable_web_search="maybe")
    assert any(error["type"] == "value_error" for error in exc_info.value.errors())
    assert "enable_web_search must be one of" in str(exc_info.value)


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
    options = VeniceChatOptions(strip_thinking_response=True)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert "extra_body" in kwargs, "extra_body should be present in kwargs"
    assert "venice_parameters" in kwargs["extra_body"], "venice_parameters should be in extra_body"
    assert kwargs["extra_body"]["venice_parameters"]["strip_thinking_response"] is True, (
        "strip_thinking_response should be True"
    )

    # Test with streaming enabled
    kwargs_stream = chat.build_kwargs(prompt, stream=True)
    assert "extra_body" in kwargs_stream, "extra_body should be present when streaming"
    assert kwargs_stream["extra_body"]["venice_parameters"]["strip_thinking_response"] is True, (
        "strip_thinking_response should be preserved when streaming"
    )

    # Test single parameter: disable_thinking
    options = VeniceChatOptions(disable_thinking=True)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert kwargs["extra_body"]["venice_parameters"]["disable_thinking"] is True, (
        "disable_thinking should be True"
    )

    # Test both parameters together
    options = VeniceChatOptions(
        strip_thinking_response=True,
        disable_thinking=False,
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

    # Test that Venice top-level params (min_p, top_k) go in extra_body
    options = VeniceChatOptions(
        min_p=0.05,
        top_k=40,
        strip_thinking_response=True,
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    # Top-level Venice params should be in extra_body directly
    assert kwargs["extra_body"]["min_p"] == 0.05, "min_p should be in extra_body"
    assert kwargs["extra_body"]["top_k"] == 40, "top_k should be in extra_body"
    # Venice parameters should be nested
    assert kwargs["extra_body"]["venice_parameters"]["strip_thinking_response"] is True, (
        "venice_parameters should coexist with top-level params"
    )

    # Test without any Venice parameters (shouldn't create extra_body)
    options = VeniceChatOptions()
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    # Should not raise an error and should return a dict (may be empty)
    assert isinstance(kwargs, dict), "build_kwargs should return a dict even without extra_body"
    # When no options are provided, kwargs may be empty
    assert "extra_body" not in kwargs or "venice_parameters" not in kwargs.get("extra_body", {}), (
        "venice_parameters should not be added when not specified in options"
    )


def test_venice_parameters_edge_cases():
    """Test edge cases and validation for venice_parameters."""
    chat = VeniceChat(
        model_id="venice/qwen3-4b",
        model_name="qwen3-4b",
        api_base="https://api.venice.ai/api/v1",
    )

    # Test with no Venice parameters
    options = VeniceChatOptions()
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)
    # Should not raise an error and should not have extra_body key
    assert "extra_body" not in kwargs, (
        "extra_body key should not exist when no Venice params are set"
    )

    # Test with multiple Venice parameters combined
    options = VeniceChatOptions(
        strip_thinking_response=True,
        character_slug="test-character",
        min_p=0.1,
        top_k=50,
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    # venice_parameters should be nested
    assert kwargs["extra_body"]["venice_parameters"]["strip_thinking_response"] is True
    assert kwargs["extra_body"]["venice_parameters"]["character_slug"] == "test-character"
    # Top-level params should be at extra_body root
    assert kwargs["extra_body"]["min_p"] == 0.1
    assert kwargs["extra_body"]["top_k"] == 50


def test_new_parameters_validation():
    """Test validation for new parameters focusing on error cases and JSON parsing."""
    # Invalid min_p values
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(min_p=-0.1)
    assert "greater than or equal to 0" in str(exc_info.value).lower()

    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(min_p=1.1)
    assert "less than or equal to 1" in str(exc_info.value).lower()

    # Invalid top_k values
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(top_k=-1)
    assert "greater than or equal to 0" in str(exc_info.value).lower()

    # Invalid repetition_penalty values
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(repetition_penalty=-0.1)
    assert "greater than or equal to 0" in str(exc_info.value).lower()

    # Valid stop_token_ids values (JSON string) should parse
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
        VeniceChatOptions(stop_token_ids=[1.5, 2.5])  # type: ignore[invalid-argument-type]

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
    assert "repetition_penalty" not in kwargs, "repetition_penalty should not be at top level"
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

    # Enable web search validator (invalid value)
    with pytest.raises(ValidationError) as exc_info:
        VeniceChatOptions(enable_web_search="invalid")
    assert "enable_web_search must be one of" in str(exc_info.value)


def test_new_parameters_merge_with_venice_parameters():
    """Test that new parameters merge correctly with venice_parameters."""
    chat = VeniceChat(
        model_id="venice/test-model",
        model_name="test-model",
        api_base="https://api.venice.ai/api/v1",
    )

    # Test merging top-level Venice params with venice_parameters
    options = VeniceChatOptions(
        min_p=0.05,
        top_k=40,
        strip_thinking_response=True,
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert "extra_body" in kwargs
    # Venice parameters should be nested
    assert kwargs["extra_body"]["venice_parameters"]["strip_thinking_response"] is True
    # Top-level parameters should be at extra_body root
    assert kwargs["extra_body"]["min_p"] == 0.05
    assert kwargs["extra_body"]["top_k"] == 40

    # Test all top-level parameters together
    options = VeniceChatOptions(
        min_p=0.05,
        top_k=40,
        repetition_penalty=1.2,
        stop_token_ids=[151643, 151645],
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert kwargs["extra_body"]["min_p"] == 0.05
    assert kwargs["extra_body"]["top_k"] == 40
    assert kwargs["extra_body"]["repetition_penalty"] == 1.2
    assert kwargs["extra_body"]["stop_token_ids"] == [151643, 151645]

    # Test that all parameters work together
    options = VeniceChatOptions(
        min_p=0.05,
        top_k=40,
        repetition_penalty=1.2,
        stop_token_ids=[151643, 151645],
        strip_thinking_response=True,
        disable_thinking=False,
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    # All venice_parameters should be preserved
    assert kwargs["extra_body"]["venice_parameters"]["strip_thinking_response"] is True
    assert kwargs["extra_body"]["venice_parameters"]["disable_thinking"] is False
    # Top-level parameters should all be present
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

    # Test with only venice_parameters, no top-level parameters
    options = VeniceChatOptions(
        strip_thinking_response=True,
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["venice_parameters"]["strip_thinking_response"] is True
    # Top-level parameters should not appear
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

    # Execute with streaming to take the simpler code path
    response = Response(prompt_obj, chat, stream=True)
    list(chat.execute(prompt_obj, stream=True, response=response))

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


def test_web_search_capability_guard():
    """Test that ModelError is raised when web search is requested on unsupported model.

    This verifies that the error handling works for both CLI and Python API usage.
    """
    import llm

    # Create a model that doesn't support web search
    chat = VeniceChat(
        model_id="venice/test-model",
        model_name="test-model",
        api_base="https://api.venice.ai/api/v1",
    )
    chat.supports_web_search = False

    # Create prompt with web search enabled
    options = VeniceChatOptions(enable_web_search="on")
    prompt_obj = Prompt(prompt="Test", model=chat, options=options)

    # Should raise llm.ModelError (not click.ClickException)
    with pytest.raises(llm.ModelError) as exc_info:
        chat.build_kwargs(prompt_obj, stream=False)

    assert "does not support web search" in str(exc_info.value)
    assert "venice/test-model" in str(exc_info.value)

    # Verify that a model WITH web search support doesn't raise
    chat.supports_web_search = True
    kwargs = chat.build_kwargs(prompt_obj, stream=False)
    assert "extra_body" in kwargs
    assert "venice_parameters" in kwargs["extra_body"]
    assert kwargs["extra_body"]["venice_parameters"]["enable_web_search"] == "on"

    # Web scraping should work regardless of web_search
    scraping_options = VeniceChatOptions(enable_web_scraping=True)
    scraping_prompt = Prompt(prompt="Test", model=chat, options=scraping_options)
    kwargs = chat.build_kwargs(scraping_prompt, stream=False)
    assert kwargs["extra_body"]["venice_parameters"]["enable_web_scraping"] is True

    kwargs = chat.build_kwargs(scraping_prompt, stream=False)
    assert kwargs["extra_body"]["venice_parameters"]["enable_web_scraping"] is True


def test_web_search_citation_parameters_options():
    """Test that web search citation parameters can be set in VeniceChatOptions."""
    # Test enable_web_citations
    options = VeniceChatOptions(enable_web_citations=True)
    assert options.enable_web_citations is True

    options = VeniceChatOptions(enable_web_citations=False)
    assert options.enable_web_citations is False

    # Test include_search_results_in_stream
    options = VeniceChatOptions(include_search_results_in_stream=True)
    assert options.include_search_results_in_stream is True

    options = VeniceChatOptions(include_search_results_in_stream=False)
    assert options.include_search_results_in_stream is False

    # Test both together
    options = VeniceChatOptions(
        enable_web_citations=True,
        include_search_results_in_stream=True,
    )
    assert options.enable_web_citations is True
    assert options.include_search_results_in_stream is True


def test_web_search_citation_parameters_build_kwargs():
    """Test that web search citation parameters are properly passed in build_kwargs."""
    chat = VeniceChat(
        model_id="venice/test-model",
        model_name="test-model",
        api_base="https://api.venice.ai/api/v1",
    )
    import llm

    # Should reject citations/search-results when model lacks web search
    options = VeniceChatOptions(enable_web_citations=True)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    with pytest.raises(llm.ModelError, match="does not support web search"):
        chat.build_kwargs(prompt, stream=False)

    options = VeniceChatOptions(include_search_results_in_stream=True)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    with pytest.raises(llm.ModelError, match="does not support web search"):
        chat.build_kwargs(prompt, stream=False)

    # Enable support but require web search to be turned on
    chat.supports_web_search = True

    options = VeniceChatOptions(enable_web_scraping=True)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)
    assert kwargs["extra_body"]["venice_parameters"]["enable_web_scraping"] is True

    options = VeniceChatOptions(enable_web_citations=True)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    with pytest.raises(llm.ModelError, match="enable_web_search must be set to 'on' or 'auto'"):
        chat.build_kwargs(prompt, stream=False)

    options = VeniceChatOptions(include_search_results_in_stream=True)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    with pytest.raises(llm.ModelError, match="enable_web_search must be set to 'on' or 'auto'"):
        chat.build_kwargs(prompt, stream=False)

    # Test enable_web_citations with web search enabled
    options = VeniceChatOptions(enable_web_search="on", enable_web_citations=True)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert "extra_body" in kwargs
    assert "venice_parameters" in kwargs["extra_body"]
    assert kwargs["extra_body"]["venice_parameters"]["enable_web_citations"] is True

    # Test include_search_results_in_stream with web search enabled
    options = VeniceChatOptions(enable_web_search="auto", include_search_results_in_stream=True)
    prompt = Prompt(prompt="Test", model=chat, options=options)
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert "extra_body" in kwargs
    assert "venice_parameters" in kwargs["extra_body"]
    assert kwargs["extra_body"]["venice_parameters"]["include_search_results_in_stream"] is True

    # Test both together with web search enabled
    options = VeniceChatOptions(
        enable_web_search="on",
        enable_web_citations=True,
        include_search_results_in_stream=True,
    )
    prompt = Prompt(prompt="Test", model=chat, options=options)

    # Set web search support to avoid validation error
    chat.supports_web_search = True
    kwargs = chat.build_kwargs(prompt, stream=False)

    assert "extra_body" in kwargs
    assert "venice_parameters" in kwargs["extra_body"]
    venice_params = kwargs["extra_body"]["venice_parameters"]
    assert venice_params["enable_web_search"] == "on"
    assert venice_params["enable_web_citations"] is True
    assert venice_params["include_search_results_in_stream"] is True


def test_cli_web_search_citation_parameters_registration(
    cli_runner, monkeypatch, mock_venice_api_key
):
    """Test that web search citation parameters are registered in CLI."""
    from llm import cli as llm_cli

    # Check prompt command help
    result = cli_runner.invoke(llm_cli.cli, ["prompt", "--help"])
    assert result.exit_code == 0
    assert "--web-citations" in result.output
    assert "--include-search-results-in-stream" in result.output

    # Check chat command help
    result = cli_runner.invoke(llm_cli.cli, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--web-citations" in result.output
    assert "--include-search-results-in-stream" in result.output


def test_cli_web_search_citation_parameters_usage(cli_runner, monkeypatch):
    """Test that CLI properly accepts web search citation parameters."""
    from llm import cli as llm_cli
    import llm
    from unittest.mock import patch, MagicMock

    monkeypatch.setenv("LLM_VENICE_KEY", "test-venice-key")
    mock_response = MagicMock()
    mock_response.text = lambda: "Mock response with citations"
    mock_response.usage = lambda: (10, 5, 15)

    # Ensure model supports web search to satisfy validation
    model = llm.get_model("venice/qwen3-4b")
    model.supports_web_search = True  # type: ignore[invalid-argument-type]

    # Spy on process_venice_options to verify options are forwarded
    from llm_venice.cli import command_hooks

    captured_options = []
    original_process = command_hooks.process_venice_options

    def spy_process(kwargs):
        result = original_process(kwargs)
        captured_options.append(list(result.get("options", [])))
        return result

    monkeypatch.setattr(command_hooks, "process_venice_options", spy_process)

    with patch.object(VeniceChat, "prompt", return_value=mock_response):
        # Test --web-citations
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/qwen3-4b",
                "--web-citations",
                "--web-search",
                "on",
                "--no-log",
                "Test with citations",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"
        options = captured_options[-1]
        assert ("enable_web_search", "on") in options
        assert ("enable_web_citations", True) in options

        # Test --web-scraping
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/qwen3-4b",
                "--web-scraping",
                "--no-log",
                "Test with scraping",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"
        options = captured_options[-1]
        assert ("enable_web_scraping", True) in options

        # Test --include-search-results-in-stream
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/qwen3-4b",
                "--include-search-results-in-stream",
                "--web-search",
                "on",
                "--no-log",
                "Test with search results in stream",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"
        options = captured_options[-1]
        assert ("enable_web_search", "on") in options
        assert ("include_search_results_in_stream", True) in options

        # Test both together
        result = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                "venice/qwen3-4b",
                "--web-citations",
                "--include-search-results-in-stream",
                "--web-search",
                "on",
                "--no-log",
                "Test with all citation parameters",
            ],
        )
        assert result.exit_code == 0, f"Command failed with: {result.output}"
        options = captured_options[-1]
        assert ("enable_web_search", "on") in options
        assert ("enable_web_citations", True) in options
        assert ("include_search_results_in_stream", True) in options
