"""Tests for Venice server-side tools (web search and X search)."""

from typing import Any, cast
from unittest.mock import patch

import llm
import pytest
from llm import Prompt, Tool

from llm_venice import VeniceChat, VeniceWebSearch, VeniceXSearch


def make_chat(**kwargs):
    """Build a Venice chat model for tool tests."""
    defaults = {
        "model_id": "venice/test-model",
        "model_name": "test-model",
        "api_base": "https://api.venice.ai/api/v1",
        "supports_tools": True,
        "supports_web_search": True,
    }
    defaults.update(kwargs)
    return VeniceChat(**defaults)


def test_venice_web_search_tool_parameters():
    """Web search tool defaults and configuration become venice_parameters."""
    assert VeniceWebSearch().venice_parameters() == {
        "enable_web_search": "auto",
        "enable_web_citations": True,
    }
    assert VeniceWebSearch(mode="on", citations=False, scraping=True).venice_parameters() == {
        "enable_web_search": "on",
        "enable_web_scraping": True,
    }

    with pytest.raises(ValueError, match="mode must be 'auto' or 'on'"):
        VeniceWebSearch(mode=cast(Any, "off"))
    with pytest.raises(TypeError, match="citations and scraping must be booleans"):
        VeniceWebSearch(citations=cast(Any, "yes"))


def test_venice_x_search_tool_parameters():
    """X search requests xAI native search."""
    assert VeniceXSearch().venice_parameters() == {"enable_x_search": True}


def test_supported_server_side_tools_follow_capabilities():
    """Only catalog-allowed tools are advertised for a model."""
    assert (
        make_chat(supports_web_search=False, supports_x_search=False).supported_server_side_tools
        == ()
    )
    assert make_chat().supported_server_side_tools == (VeniceWebSearch,)
    assert make_chat(
        supports_web_search=False, supports_x_search=True
    ).supported_server_side_tools == (VeniceXSearch,)
    assert make_chat(supports_x_search=True).supported_server_side_tools == (
        VeniceWebSearch,
        VeniceXSearch,
    )


def test_build_kwargs_applies_web_search_tool():
    """The server-side tool becomes venice_parameters, not a function tool."""
    chat = make_chat()
    tool = VeniceWebSearch(mode="on", scraping=True)
    prompt = Prompt(prompt="What happened today?", model=chat, tools=[tool])

    kwargs = chat.build_kwargs(prompt, stream=False)

    assert "tools" not in kwargs
    assert kwargs["extra_body"]["venice_parameters"] == {
        "enable_web_search": "on",
        "enable_web_citations": True,
        "enable_web_scraping": True,
    }
    # The caller's prompt still owns its tools
    assert prompt.tools == [tool]


def test_build_kwargs_splits_function_and_server_side_tools():
    """Function tools stay in tools while the server-side tool is applied."""

    def get_weather(city: str) -> str:
        """Look up the weather."""
        return city

    chat = make_chat()
    prompt = Prompt(
        prompt="Test",
        model=chat,
        tools=[Tool.function(get_weather), VeniceWebSearch()],
    )

    kwargs = chat.build_kwargs(prompt, stream=False)

    assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["get_weather"]
    assert kwargs["extra_body"]["venice_parameters"]["enable_web_search"] == "auto"


def test_build_kwargs_x_search_replaces_web_search_options():
    """X search drops web-search parameters, as in the official venice-cli."""
    chat = make_chat(supports_x_search=True)
    options = chat.Options(enable_web_search="on", enable_web_citations=True)
    prompt = Prompt(
        prompt="What is trending?",
        model=chat,
        options=options,
        tools=[VeniceXSearch()],
    )

    kwargs = chat.build_kwargs(prompt, stream=False)

    assert kwargs["extra_body"]["venice_parameters"] == {"enable_x_search": True}


def test_build_kwargs_tool_search_satisfies_citation_option():
    """A web search tool supplies the mode an explicit citation option needs."""
    chat = make_chat()
    options = chat.Options(enable_web_citations=True)
    prompt = Prompt(
        prompt="Test",
        model=chat,
        options=options,
        tools=[VeniceWebSearch(mode="auto")],
    )

    kwargs = chat.build_kwargs(prompt, stream=False)

    venice_parameters = kwargs["extra_body"]["venice_parameters"]
    assert venice_parameters["enable_web_search"] == "auto"
    assert venice_parameters["enable_web_citations"] is True


def test_server_side_tool_requires_capability(mock_venice_api_key):
    """llm rejects a server-side tool the model does not advertise."""
    chat = make_chat(supports_web_search=False)

    with pytest.raises(ValueError, match="does not support server-side tool"):
        chat.prompt("Search the web", tools=[VeniceWebSearch()])


def test_tools_command_lists_server_side_tools(cli_runner, hermetic_venice_model):
    """`llm tools -m venice/<model>` lists the catalog-allowed tools."""
    from llm import cli as llm_cli

    model_id = hermetic_venice_model(capabilities={"supportsWebSearch": True})

    result = cli_runner.invoke(llm_cli.cli, ["tools", "-m", model_id])

    assert result.exit_code == 0, result.output
    assert "Server-side tools for" in result.output
    assert "VeniceWebSearch(mode:" in result.output
    assert "Search the web with Venice's hosted web search." in result.output
    assert "VeniceXSearch" not in result.output


def test_conversation_restores_server_side_tools(cli_runner, monkeypatch, hermetic_venice_model):
    """`-c` restores the server-side tool and its configuration."""
    from llm import cli as llm_cli

    model_id = hermetic_venice_model(capabilities={"supportsWebSearch": True})
    monkeypatch.setenv("LLM_VENICE_KEY", "test-venice-key")
    seen = []

    def fake_execute(self, prompt, stream=False, **kwargs):
        seen.append(
            [
                (type(tool).__name__, getattr(tool, "mode", None))
                for tool in prompt.tools
                if isinstance(tool, llm.ServerSideTool)
            ]
        )
        return iter(["Mock response"])

    with patch.object(VeniceChat, "execute", fake_execute):
        first = cli_runner.invoke(
            llm_cli.cli,
            [
                "prompt",
                "-m",
                model_id,
                "-T",
                'VeniceWebSearch(mode="on", citations=false)',
                "Start",
            ],
        )
        assert first.exit_code == 0, first.output
        assert seen[-1] == [("VeniceWebSearch", "on")]

        follow_up = cli_runner.invoke(llm_cli.cli, ["prompt", "-c", "Continue"])
        assert follow_up.exit_code == 0, follow_up.output
        assert seen[-1] == [("VeniceWebSearch", "on")]
