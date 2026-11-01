"""Function tools for /augment/search: registration, schemas and chains."""

import asyncio
import json

import llm
import pytest
from llm.cli import cli

from llm_venice import (
    AsyncVeniceChat,
    VeniceChat,
    venice_augment_search,
    venice_augment_search_brave,
    venice_augment_search_exa,
    venice_augment_search_google,
)

API_BASE = "https://api.venice.ai/api/v1"
SEARCH_URL = f"{API_BASE}/augment/search"
CHAT_URL = f"{API_BASE}/chat/completions"
PAYLOAD = {
    "query": "venice ai",
    "results": [{"title": "Venice", "url": "https://venice.ai", "content": "AI", "date": ""}],
}
TOOLS = {
    "venice_augment_search": venice_augment_search,
    "venice_augment_search_brave": venice_augment_search_brave,
    "venice_augment_search_google": venice_augment_search_google,
    "venice_augment_search_exa": venice_augment_search_exa,
}


def completion(message):
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def tool_call(name, arguments):
    return completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }
    )


def answer(text="Done"):
    return completion({"role": "assistant", "content": text})


def chat_requests(httpx2_mock):
    return [
        json.loads(request.read())
        for request in httpx2_mock.get_requests()
        if str(request.url) == CHAT_URL
    ]


def test_tools_are_registered_by_the_plugin():
    registered = llm.get_tools()
    for name, function in TOOLS.items():
        tool = registered[name]
        assert isinstance(tool, llm.Tool)
        assert tool.implementation is function
        assert tool.plugin == "venice"


def test_tool_schemas():
    general = llm.Tool.function(venice_augment_search)
    assert general.input_schema == {
        "properties": {
            "query": {"type": "string"},
            "provider": {
                "default": "brave",
                "enum": ["brave", "google", "exa"],
                "type": "string",
            },
            "limit": {"default": 10, "type": "integer"},
        },
        "required": ["query"],
        "type": "object",
    }
    for function in TOOLS.values():
        tool = llm.Tool.function(function)
        assert tool.description and tool.description == tool.description.strip()
        assert "\n " not in tool.description
        if function is not venice_augment_search:
            assert tool.input_schema == {
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"default": 10, "type": "integer"},
                },
                "required": ["query"],
                "type": "object",
            }


@pytest.mark.parametrize(
    "function,kwargs,body",
    [
        (venice_augment_search, {}, {"limit": 10, "search_provider": "brave"}),
        (
            venice_augment_search,
            {"provider": "exa", "limit": 3},
            {"limit": 3, "search_provider": "exa"},
        ),
        (venice_augment_search_brave, {}, {"limit": 10, "search_provider": "brave"}),
        (venice_augment_search_google, {"limit": 2}, {"limit": 2, "search_provider": "google"}),
        (venice_augment_search_exa, {}, {"limit": 10, "search_provider": "exa"}),
    ],
)
def test_tools_send_their_provider(function, kwargs, body, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(
        method="POST", url=SEARCH_URL, match_json={"query": "venice ai", **body}, json=PAYLOAD
    )
    assert function("venice ai", **kwargs) == PAYLOAD
    request = httpx2_mock.get_request()
    assert request.headers["authorization"] == f"Bearer {mock_venice_api_key}"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"query": "", "limit": 10}, "query: String should have at least 1 character"),
        ({"query": "q", "limit": 21}, "limit: Input should be less than or equal to 20"),
        (
            {"query": "q", "provider": "bing"},
            "provider: Input should be 'brave', 'google' or 'exa'",
        ),
    ],
)
def test_invalid_input_gives_a_short_error(kwargs, message, httpx2_mock, mock_venice_api_key):
    with pytest.raises(ValueError, match="^Invalid search input: ") as caught:
        venice_augment_search(**kwargs)
    assert message in str(caught.value)
    assert "pydantic" not in str(caught.value)
    assert not httpx2_mock.get_requests()


def test_cli_chain_runs_the_tool_and_continue_restores_it(
    cli_runner, httpx2_mock, monkeypatch, hermetic_venice_model
):
    model_id = hermetic_venice_model(capabilities={"supportsFunctionCalling": True})
    monkeypatch.setenv("LLM_VENICE_KEY", "test-venice-key")
    httpx2_mock.add_response(
        url=CHAT_URL, json=tool_call("venice_augment_search_exa", {"query": "venice ai"})
    )
    httpx2_mock.add_response(
        method="POST",
        url=SEARCH_URL,
        match_json={"query": "venice ai", "limit": 10, "search_provider": "exa"},
        json=PAYLOAD,
    )
    httpx2_mock.add_response(url=CHAT_URL, json=answer("Venice is an AI platform."))
    httpx2_mock.add_response(url=CHAT_URL, json=answer("Continued."))

    first = cli_runner.invoke(
        cli,
        ["prompt", "-m", model_id, "-T", "venice_augment_search_exa", "--no-stream", "Search"],
    )
    assert first.exit_code == 0, first.output
    assert "Venice is an AI platform." in first.output

    follow_up = cli_runner.invoke(cli, ["prompt", "-c", "--no-stream", "Continue"])
    assert follow_up.exit_code == 0, follow_up.output

    requests = chat_requests(httpx2_mock)
    assert len(requests) == 3
    for request in requests:
        assert [tool["function"]["name"] for tool in request["tools"]] == [
            "venice_augment_search_exa"
        ]
    tool_message = requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"]) == PAYLOAD


def test_search_failure_is_reported_to_the_model(httpx2_mock, mock_venice_api_key):
    chat = VeniceChat(model_id="test-model", api_base=API_BASE, supports_tools=True)
    httpx2_mock.add_response(
        url=CHAT_URL, json=tool_call("venice_augment_search", {"query": "q", "provider": "exa"})
    )
    httpx2_mock.add_response(url=SEARCH_URL, status_code=402, json={"code": "INSUFFICIENT_BALANCE"})
    httpx2_mock.add_response(url=CHAT_URL, json=answer("Search is unavailable."))

    chain = chat.chain("Search", tools=[venice_augment_search], stream=False)
    assert chain.text() == "Search is unavailable."

    tool_message = chat_requests(httpx2_mock)[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"].startswith("Error: Searching the web failed:")
    assert "INSUFFICIENT_BALANCE" in tool_message["content"]


def test_async_chain_runs_the_sync_tool(httpx2_mock, mock_venice_api_key):
    chat = AsyncVeniceChat(model_id="test-model", api_base=API_BASE, supports_tools=True)
    httpx2_mock.add_response(
        url=CHAT_URL, json=tool_call("venice_augment_search_google", {"query": "q", "limit": 2})
    )
    httpx2_mock.add_response(
        url=SEARCH_URL,
        match_json={"query": "q", "limit": 2, "search_provider": "google"},
        json=PAYLOAD,
    )
    httpx2_mock.add_response(url=CHAT_URL, json=answer())

    async def run():
        chain = chat.chain("Search", tools=[venice_augment_search_google], stream=False)
        return await chain.text()

    assert asyncio.run(run()) == "Done"
    assert json.loads(chat_requests(httpx2_mock)[1]["messages"][-1]["content"]) == PAYLOAD


def test_models_without_function_calling_reject_the_tools():
    chat = VeniceChat(model_id="test-model", api_base=API_BASE, supports_tools=False)
    with pytest.raises(ValueError, match="does not support tools"):
        chat.prompt("Search", tools=[venice_augment_search])
