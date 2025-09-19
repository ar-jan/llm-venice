"""Tests for VeniceChat tool support functionality"""

from typing import List, Literal
from unittest.mock import Mock, patch
import pytest
from llm import Prompt, Response, Tool
from llm_venice import VeniceChat
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types import CompletionUsage


@pytest.fixture
def chat_model_with_tools():
    return VeniceChat(
        model_id="venice/test-model-tools",
        model_name="test-model-tools",
        api_base="https://api.venice.ai/api/v1",
        supports_tools=True,
    )


@pytest.fixture
def chat_model_without_tools():
    return VeniceChat(
        model_id="venice/test-model-no-tools",
        model_name="test-model-no-tools",
        api_base="https://api.venice.ai/api/v1",
        supports_tools=False,
    )


def test_build_kwargs_with_various_parameter_types(chat_model_with_tools):
    """Test that a tool with a variety of parameter types is properly formatted."""

    def complex_tool(
        name: str,
        age: int,
        score: float,
        is_active: bool,
        tags: List[str],
        unit: Literal["metric", "imperial"] = "metric",
    ):
        """A tool with multiple parameter types."""
        pass

    tool = Tool.function(complex_tool)
    prompt = Prompt(prompt="Test", model=chat_model_with_tools, tools=[tool])
    kwargs = chat_model_with_tools.build_kwargs(prompt, stream=False)

    assert "tools" in kwargs
    tool_def = kwargs["tools"][0]["function"]
    assert tool_def["name"] == "complex_tool"

    params = tool_def["parameters"]
    assert params["type"] == "object"
    assert "required" in params
    assert sorted(params["required"]) == ["age", "is_active", "name", "score", "tags"]

    props = params["properties"]
    assert props["name"]["type"] == "string"
    assert props["age"]["type"] == "integer"
    assert props["score"]["type"] == "number"
    assert props["is_active"]["type"] == "boolean"
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"
    assert props["unit"]["type"] == "string"
    assert props["unit"]["enum"] == ["metric", "imperial"]
    assert props["unit"]["default"] == "metric"


def test_build_kwargs_with_multiple_tools(chat_model_with_tools):
    """Test that multiple tools are properly formatted."""

    def add(a: int, b: int):
        """Add two numbers."""
        return a + b

    def multiply(x: float, y: float):
        """Multiply two numbers."""
        return x * y

    prompt = Prompt(
        prompt="Calculate",
        model=chat_model_with_tools,
        tools=[Tool.function(add), Tool.function(multiply)],
    )
    kwargs = chat_model_with_tools.build_kwargs(prompt, stream=False)

    assert "tools" in kwargs
    assert len(kwargs["tools"]) == 2
    assert kwargs["tools"][0]["function"]["name"] == "add"
    assert kwargs["tools"][1]["function"]["name"] == "multiply"


def test_build_kwargs_with_tool_without_parameters(chat_model_with_tools):
    """Test that tools without parameters are handled correctly."""

    def get_time():
        """Get the current time."""
        import datetime

        return datetime.datetime.now().isoformat()

    prompt = Prompt(
        prompt="What time is it?",
        model=chat_model_with_tools,
        tools=[Tool.function(get_time)],
    )
    kwargs = chat_model_with_tools.build_kwargs(prompt, stream=False)

    params = kwargs["tools"][0]["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"] == {}


def test_build_kwargs_with_empty_tools_list(chat_model_with_tools):
    """Test that an empty tools list does not add the 'tools' key."""
    prompt = Prompt(prompt="Test", model=chat_model_with_tools, tools=[])
    kwargs = chat_model_with_tools.build_kwargs(prompt, stream=False)
    assert "tools" not in kwargs


def test_execute_sends_tools_and_parses_response(
    chat_model_with_tools, mock_venice_api_key
):
    """
    Integration-style test to verify that the 'tools' payload is sent
    and a 'tool_calls' response from the API is correctly parsed.
    """

    def get_weather(city: str):
        """Gets the weather for a city."""
        return f"Weather in {city} is sunny."

    # Create an OpenAI response object with tool calls
    tool_call = ChatCompletionMessageToolCall(
        id="call_abc123",
        type="function",
        function=Function(name="get_weather", arguments='{"city": "Leiden"}'),
    )

    message = ChatCompletionMessage(
        role="assistant", content=None, tool_calls=[tool_call]
    )

    choice = Choice(index=0, message=message, finish_reason="tool_calls")

    usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    mock_completion = ChatCompletion(
        id="chatcmpl-123",
        object="chat.completion",
        created=1677652288,
        model="test-model-tools",
        choices=[choice],
        usage=usage,
    )

    # Mock the OpenAI client's chat.completions.create method
    with patch("llm.default_plugins.openai_models.openai.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_completion

        # Execute the prompt
        prompt = "What is the weather in Leiden?"
        response = chat_model_with_tools.prompt(
            prompt, tools=[Tool.function(get_weather)], stream=False
        )

        # Force the response to execute by accessing the text
        # This triggers the API call
        response_text = response.text()

        # Verify the client was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs

        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["function"]["name"] == "get_weather"
        assert "city" in call_kwargs["tools"][0]["function"]["parameters"]["properties"]

        # Verify the Response object correctly parsed the tool call
        assert isinstance(response, Response)
        # When a response contains only tool calls and no content, text should be empty
        assert response_text == ""

        # Get the tool_calls
        tool_calls = response.tool_calls()
        assert tool_calls is not None
        assert len(tool_calls) == 1

        tool_call = tool_calls[0]
        assert tool_call.tool_call_id == "call_abc123"
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == {
            "city": "Leiden"
        }  # llm auto-parses the JSON string


def test_prompt_with_tools_raises_error_if_unsupported(chat_model_without_tools):
    """
    Verify that calling .prompt() with tools on a model that has
    supports_tools=False raises a ValueError before making an API call.
    """

    # Arrange: Define a simple tool that will be passed to the model
    def get_city_weather(city: str):
        """A simple tool to get weather."""
        return f"Weather in {city} is sunny."

    tool = Tool.function(get_city_weather)

    # Trying to use the tool should raise a ValueError
    with pytest.raises(ValueError) as excinfo:
        chat_model_without_tools.prompt("What's the weather in London?", tools=[tool])

    assert "does not support tools" in str(excinfo.value)
