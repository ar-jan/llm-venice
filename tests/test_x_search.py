"""X-search options and tools share capability checks and request shaping."""

import asyncio
import json

import llm
import pytest
from llm import Prompt
from llm import cli as llm_cli

from llm_venice import AsyncVeniceChat, VeniceChat, VeniceWebSearch, VeniceXSearch


def make_chat(cls=VeniceChat):
    return cls(
        model_id="venice/test-model",
        api_base="https://api.venice.ai/api/v1",
        supports_x_search=True,
    )


@pytest.fixture(params=[VeniceChat, AsyncVeniceChat])
def chat(request):
    return make_chat(request.param)


@pytest.fixture
def mock_completion(httpx2_mock):
    """Mock the HTTP boundary so prompt and CLI tests exercise the real SDK."""
    httpx2_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/chat/completions",
        is_reusable=True,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def response_text(chat, **kwargs):
    if isinstance(chat, AsyncVeniceChat):

        async def run():
            return await chat.prompt("Search", stream=False, **kwargs).text()

        return asyncio.run(run())
    return chat.prompt("Search", stream=False, **kwargs).text()


@pytest.mark.parametrize(
    "options",
    [{}, {"enable_x_search": None}, {"enable_x_search": False}, {"enable_x_search": True}],
)
@pytest.mark.parametrize("supports_x_search", [False, True])
def test_python_option_request(
    chat, options, supports_x_search, httpx2_mock, request_json, mock_venice_api_key, request
):
    chat.supports_x_search = supports_x_search
    if options.get("enable_x_search") is True and not supports_x_search:
        with pytest.raises(
            llm.ModelError, match="Model venice/test-model does not support X search"
        ):
            response_text(chat, **options)
        assert httpx2_mock.get_requests() == []
        return

    request.getfixturevalue("mock_completion")
    assert response_text(chat, **options) == "ok"
    payload = request_json()
    if options.get("enable_x_search") is None:
        assert "venice_parameters" not in payload
    else:
        assert payload["venice_parameters"] == options
    assert "enable_x_search" not in payload


@pytest.mark.parametrize(
    "options,tools",
    [
        ({"enable_x_search": True}, []),
        ({}, [VeniceXSearch()]),
        ({"enable_x_search": False}, [VeniceXSearch()]),
        ({"enable_x_search": True}, [VeniceWebSearch()]),
        ({"enable_x_search": False}, [VeniceWebSearch(), VeniceXSearch()]),
        ({"enable_x_search": False}, [VeniceXSearch(), VeniceWebSearch()]),
    ],
)
def test_effective_x_search_replaces_web_settings(options, tools):
    chat = make_chat()
    # The model deliberately lacks Venice web search: its guard must run only
    # after the replaced settings have been removed.
    prompt = Prompt(
        prompt="Search",
        model=chat,
        options=chat.Options(
            **options,
            enable_web_search="off",
            enable_web_citations=True,
            enable_web_scraping=True,
            include_search_results_in_stream=True,
            include_venice_system_prompt=False,
            extra_body={"venice_parameters": {"future_flag": True}},
        ),
        tools=tools,
    )
    original_options = prompt.options.model_dump()

    kwargs = chat.build_kwargs(prompt, stream=False)

    assert kwargs["extra_body"]["venice_parameters"] == {
        "enable_x_search": True,
        "include_venice_system_prompt": False,
        "future_flag": True,
    }
    assert "tools" not in kwargs
    assert prompt.tools == tools
    assert prompt.options.model_dump() == original_options


def test_x_search_tool_has_shared_capability_guard(chat):
    """Direct request shaping also validates tools after they override options."""
    chat.supports_x_search = False
    prompt = Prompt(
        prompt="Search",
        model=chat,
        options=chat.Options(enable_x_search=False),
        tools=[VeniceXSearch()],
    )
    with pytest.raises(llm.ModelError, match="does not support X search"):
        chat.build_kwargs(prompt, stream=False)


def test_llm_still_rejects_unadvertised_tools(chat):
    chat.supports_x_search = False
    with pytest.raises(ValueError, match="does not support server-side tool"):
        chat.prompt("Search", tools=[VeniceXSearch()], enable_x_search=False)
    chat.supports_x_search = True
    with pytest.raises(ValueError, match="does not support server-side tool"):
        chat.prompt("Search", tools=[VeniceWebSearch()], enable_x_search=True)


@pytest.mark.parametrize("x_search", [None, False])
@pytest.mark.parametrize(
    "supports_web_search,web_options,tool",
    [
        (False, {"enable_web_search": "on"}, None),
        (
            True,
            {
                "enable_web_search": "on",
                "enable_web_citations": True,
                "enable_web_scraping": True,
                "include_search_results_in_stream": True,
            },
            None,
        ),
        (True, {"enable_web_search": "off"}, VeniceWebSearch()),
    ],
    ids=["unsupported", "web-options", "web-tool"],
)
def test_disabled_x_search_preserves_web_behavior(x_search, supports_web_search, web_options, tool):
    chat = make_chat()
    chat.supports_web_search = supports_web_search
    prompt = Prompt(
        prompt="Search",
        model=chat,
        options=chat.Options(enable_x_search=x_search, **web_options),
        tools=[tool] if tool else [],
    )
    if not supports_web_search:
        with pytest.raises(llm.ModelError, match="does not support web search"):
            chat.build_kwargs(prompt, stream=False)
    else:
        expected = dict(web_options)
        if tool:
            expected.update(tool.venice_parameters())
        if x_search is not None:
            expected["enable_x_search"] = x_search
        assert (
            chat.build_kwargs(prompt, stream=False)["extra_body"]["venice_parameters"] == expected
        )


@pytest.mark.parametrize(
    "value,supports_x_search",
    [("false", False), ("true", False), ("true", True), ("invalid", True)],
)
def test_cli_boolean_option(
    cli_runner,
    hermetic_venice_model,
    mock_venice_api_key,
    httpx2_mock,
    request_json,
    request,
    value,
    supports_x_search,
):
    model_id = hermetic_venice_model(capabilities={"supportsXSearch": supports_x_search})
    succeeds = value == "false" or (value == "true" and supports_x_search)
    if succeeds:
        request.getfixturevalue("mock_completion")
    result = cli_runner.invoke(
        llm_cli.cli,
        [
            "prompt",
            "-m",
            model_id,
            "-o",
            "enable_x_search",
            value,
            "--no-log",
            "--no-stream",
            "Search",
        ],
    )
    if succeeds:
        assert result.exit_code == 0, result.output
        assert request_json()["venice_parameters"] == {"enable_x_search": value == "true"}
    else:
        assert result.exit_code != 0
        assert (
            "enable_x_search" if value == "invalid" else "does not support X search"
        ) in result.output
        assert httpx2_mock.get_requests() == []


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("nested", [False, True])
def test_reserved_x_search_extra_body(chat, value, nested):
    extra_body = {"enable_x_search": value}
    if nested:
        extra_body = {"venice_parameters": extra_body}
    for encoded in (extra_body, json.dumps(extra_body)):
        with pytest.raises(
            ValueError, match="enable_x_search; use the corresponding option instead"
        ):
            chat.prompt("Search", extra_body=encoded)


def test_x_search_tool_discovery_and_continuation(
    cli_runner, hermetic_venice_model, mock_venice_api_key, mock_completion, request_json
):
    model_id = hermetic_venice_model(
        capabilities={"supportsXSearch": True, "supportsWebSearch": True}
    )
    result = cli_runner.invoke(llm_cli.cli, ["tools", "-m", model_id])
    assert result.exit_code == 0, result.output
    assert "VeniceXSearch" in result.output

    for args in (
        ["-m", model_id, "-T", "VeniceWebSearch", "-T", "VeniceXSearch"],
        ["-c"],
    ):
        result = cli_runner.invoke(
            llm_cli.cli,
            ["prompt", *args, "-o", "enable_x_search", "false", "--no-stream", "Search"],
        )
        assert result.exit_code == 0, result.output
        payload = request_json()
        assert payload["venice_parameters"] == {"enable_x_search": True}
        assert "tools" not in payload
