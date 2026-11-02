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


def tool_call(name, arguments, call_id="call_1"):
    return completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
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
                "enum": ["brave", "google", "exa"],
                "type": "string",
            },
            "limit": {"default": 10, "type": "integer"},
        },
        "required": ["query", "provider"],
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
        (
            venice_augment_search,
            {"provider": "brave"},
            {"limit": 10, "search_provider": "brave"},
        ),
        (
            venice_augment_search,
            {"provider": "google"},
            {"limit": 10, "search_provider": "google"},
        ),
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
        (
            {"query": "", "provider": "brave", "limit": 10},
            "query: String should have at least 1 character",
        ),
        (
            {"query": "q", "provider": "exa", "limit": 21},
            "limit: Input should be less than or equal to 20",
        ),
    ],
)
def test_invalid_input_gives_a_short_error(kwargs, message, httpx2_mock, mock_venice_api_key):
    with pytest.raises(ValueError, match="^Invalid search input: ") as caught:
        venice_augment_search(**kwargs)
    assert message in str(caught.value)
    assert "pydantic" not in str(caught.value)
    assert not httpx2_mock.get_requests()


@pytest.mark.parametrize("provider", [None, "bing", "", 1, ["exa"], {"provider": "exa"}])
def test_generic_tool_rejects_invalid_provider_before_http(provider, httpx2_mock):
    with pytest.raises(ValueError, match="provider: choose 'brave', 'google' or 'exa' explicitly"):
        venice_augment_search("q", provider=provider)
    assert not httpx2_mock.get_requests()


def test_generic_tool_requires_provider_before_http(httpx2_mock):
    tool = llm.Tool.function(venice_augment_search)
    assert tool.implementation is not None
    with pytest.raises(ValueError, match="provider: choose 'brave', 'google' or 'exa' explicitly"):
        tool.implementation(query="q")
    assert not httpx2_mock.get_requests()


def test_tool_reports_all_problems_in_one_error(httpx2_mock):
    with pytest.raises(ValueError) as caught:
        venice_augment_search(query="", provider="bing", limit=21, engine="x")  # type: ignore[call-arg]
    assert str(caught.value) == (
        "Invalid search input: query: String should have at least 1 character; "
        "provider: choose 'brave', 'google' or 'exa' explicitly; "
        "limit: Input should be less than or equal to 20; "
        "engine: unexpected argument"
    )
    assert not httpx2_mock.get_requests()


@pytest.mark.parametrize("function", TOOLS.values())
def test_tools_report_a_missing_query(function, httpx2_mock):
    implementation = llm.Tool.function(function).implementation
    assert implementation is not None
    with pytest.raises(ValueError, match="^Invalid search input: query: Field required"):
        implementation(provider="exa") if function is venice_augment_search else implementation()
    assert not httpx2_mock.get_requests()


def test_cli_generic_tool_changes_provider_and_continue_restores_it(
    cli_runner, httpx2_mock, monkeypatch, hermetic_venice_model
):
    model_id = hermetic_venice_model(capabilities={"supportsFunctionCalling": True})
    monkeypatch.setenv("LLM_VENICE_KEY", "test-venice-key")
    providers = ["exa", "google", "brave"]
    for provider in providers:
        query = f"{provider} query"
        httpx2_mock.add_response(
            url=CHAT_URL,
            json=tool_call(
                "venice_augment_search",
                {"query": query, "provider": provider, "limit": 2},
                call_id=f"call_{provider}",
            ),
        )
        httpx2_mock.add_response(
            method="POST",
            url=SEARCH_URL,
            match_json={"query": query, "search_provider": provider, "limit": 2},
            json={"query": query, "results": PAYLOAD["results"]},
        )
    httpx2_mock.add_response(url=CHAT_URL, json=answer())
    httpx2_mock.add_response(url=CHAT_URL, json=answer("Continued."))

    first = cli_runner.invoke(
        cli,
        ["prompt", "-m", model_id, "-T", "venice_augment_search", "--td", "--no-stream", "Search"],
    )
    assert first.exit_code == 0, first.output
    assert "Done" in first.output
    for provider in providers:
        assert provider in first.output

    follow_up = cli_runner.invoke(cli, ["prompt", "-c", "--no-stream", "Continue"])
    assert follow_up.exit_code == 0, follow_up.output
    assert "Continued." in follow_up.output

    requests = chat_requests(httpx2_mock)
    assert len(requests) == 5
    for request in requests:
        assert len(request["tools"]) == 1
        function = request["tools"][0]["function"]
        assert function["name"] == "venice_augment_search"
        assert function["parameters"]["required"] == ["query", "provider"]
        assert "default" not in function["parameters"]["properties"]["provider"]
        assert function["description"] == llm.Tool.function(venice_augment_search).description
        assert "Choose provider explicitly" in function["description"]
    for index, provider in enumerate(providers, 1):
        tool_message = requests[index]["messages"][-1]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == f"call_{provider}"
        assert json.loads(tool_message["content"])["query"] == f"{provider} query"


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "arguments",
    [{"query": "q"}, {"query": "q", "provider": None}, {"query": "q", "provider": "bing"}],
)
def test_generic_tool_recovers_from_invalid_provider(
    is_async, arguments, httpx2_mock, mock_venice_api_key
):
    model_class = AsyncVeniceChat if is_async else VeniceChat
    chat = model_class(model_id="test-model", api_base=API_BASE, supports_tools=True)
    httpx2_mock.add_response(url=CHAT_URL, json=tool_call("venice_augment_search", arguments))
    httpx2_mock.add_response(
        url=CHAT_URL,
        json=tool_call("venice_augment_search", {"query": "q", "provider": "exa"}, "corrected"),
    )
    httpx2_mock.add_response(
        method="POST",
        url=SEARCH_URL,
        match_json={"query": "q", "search_provider": "exa", "limit": 10},
        json=PAYLOAD,
    )
    httpx2_mock.add_response(url=CHAT_URL, json=answer())
    if isinstance(chat, AsyncVeniceChat):
        chain = chat.chain("Search", tools=[venice_augment_search], stream=False)
        assert asyncio.run(chain.text()) == "Done"
    else:
        assert chat.chain("Search", tools=[venice_augment_search], stream=False).text() == "Done"

    requests = chat_requests(httpx2_mock)
    error = requests[1]["messages"][-1]
    assert error["role"] == "tool"
    assert error["content"] == (
        "Error: Invalid search input: provider: choose 'brave', 'google' or 'exa' explicitly"
    )
    assert json.loads(requests[2]["messages"][-1]["content"]) == PAYLOAD
    assert len(httpx2_mock.get_requests(url=SEARCH_URL)) == 1


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
