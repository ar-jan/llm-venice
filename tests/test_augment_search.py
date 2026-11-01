"""Web search through /augment/search: request contract, clients and CLI."""

import asyncio
import json

import httpx2
import pytest
from llm.cli import cli
from pydantic import ValidationError
from typing_extensions import assert_type

from llm_venice.client import AsyncVeniceClient, VeniceClient
from llm_venice._endpoints import AugmentSearchInput, json_operation, parse_envelope
from llm_venice.errors import RateLimitError, VeniceAPIError, VeniceResponseError
from llm_venice.types import AugmentSearchResponse

URL = "https://api.venice.ai/api/v1/augment/search"
RESULT = {
    "title": "Venice",
    "url": "https://venice.ai",
    "content": "Private and uncensored AI",
    "date": "2026-10-01",
}
PAYLOAD = {"query": "venice ai", "results": [RESULT]}


def invoke(is_async, *args, **kwargs):
    if is_async:

        async def run():
            async with AsyncVeniceClient(key="test") as client:
                return await client.augment_search(*args, **kwargs)

        return asyncio.run(run())
    with VeniceClient(key="test") as client:
        return client.augment_search(*args, **kwargs)


def parse(payload):
    response = httpx2.Response(200, json=payload, request=httpx2.Request("POST", URL))
    return parse_envelope(response, "augment_search", "Searching the web")


def test_request_body_omits_unset_options_and_renames_provider():
    assert AugmentSearchInput(query="q").body() == {"query": "q"}
    assert AugmentSearchInput(query="q", limit=5, provider="exa").body() == {
        "query": "q",
        "limit": 5,
        "search_provider": "exa",
    }
    method, path, action, kwargs = json_operation(
        "augment_search", {"query": "q", "limit": None, "provider": "google"}
    )
    assert (method, path, action) == ("POST", "augment/search", "Searching the web")
    assert kwargs == {"json": {"query": "q", "search_provider": "google"}}


@pytest.mark.parametrize(
    "inputs",
    [
        {"query": ""},
        {"query": "x" * 401},
        {"query": "q", "limit": 0},
        {"query": "q", "limit": 21},
        {"query": "q", "provider": "bing"},
        {"query": "q", "search_provider": "exa"},
    ],
)
def test_request_limits(inputs):
    with pytest.raises(ValidationError):
        AugmentSearchInput.model_validate(inputs)


def test_request_bounds_are_inclusive():
    assert AugmentSearchInput(query="x" * 400, limit=1).limit == 1
    assert AugmentSearchInput(query="q", limit=20).limit == 20


def test_response_tolerates_optional_drift_and_keeps_unknown_fields():
    drifted = {"title": "t", "url": "u", "date": None, "score": 0.9}
    payload = {"query": "q", "results": [drifted, {"title": "", "url": "u2"}], "future": 1}
    assert parse(payload) == payload
    assert parse({"query": "q", "results": []}) == {"query": "q", "results": []}


@pytest.mark.parametrize(
    "payload,detail",
    [
        ({"results": []}, "query"),
        ({"query": "q"}, "results"),
        ({"query": "q", "results": {"title": "t"}}, "results"),
        ({"query": "q", "results": [None]}, "expected an object"),
        ({"query": "q", "results": [{"title": "t"}]}, "url"),
        ({"query": "q", "results": [{"url": "u", "title": None}]}, "title"),
        ([PAYLOAD], "expected a JSON object"),
    ],
)
def test_malformed_responses(payload, detail):
    with pytest.raises(VeniceResponseError, match=detail):
        parse(payload)


@pytest.mark.parametrize("is_async", [False, True])
def test_client_parity(is_async, httpx2_mock):
    httpx2_mock.add_response(
        method="POST",
        url=URL,
        match_json={"query": "venice ai", "limit": 5, "search_provider": "exa"},
        json=PAYLOAD,
    )
    result = invoke(is_async, "venice ai", limit=5, provider="exa")
    assert_type(result, AugmentSearchResponse)
    assert result == PAYLOAD
    request = httpx2_mock.get_request()
    assert request.headers["authorization"] == "Bearer test"
    assert request.headers["content-type"] == "application/json"


@pytest.mark.parametrize("is_async", [False, True])
def test_client_omits_defaults(is_async, httpx2_mock):
    httpx2_mock.add_response(method="POST", url=URL, match_json={"query": "q"}, json=PAYLOAD)
    assert invoke(is_async, "q") == PAYLOAD


@pytest.mark.parametrize("is_async", [False, True])
def test_client_validation_precedes_http(is_async, httpx2_mock):
    with pytest.raises(ValidationError):
        invoke(is_async, "q", limit=21)
    with pytest.raises(ValidationError):
        invoke(is_async, "")
    assert not httpx2_mock.get_requests()


@pytest.mark.parametrize("is_async", [False, True])
def test_client_errors(is_async, httpx2_mock):
    httpx2_mock.add_response(
        url=URL, status_code=400, json={"error": "Invalid request", "details": {}}
    )
    with pytest.raises(VeniceAPIError, match="Searching the web failed: .*400.*Invalid request"):
        invoke(is_async, "q")
    httpx2_mock.add_response(url=URL, status_code=402, json={"code": "INSUFFICIENT_BALANCE"})
    with pytest.raises(VeniceAPIError, match="INSUFFICIENT_BALANCE"):
        invoke(is_async, "q")
    httpx2_mock.add_response(url=URL, status_code=429, json={"message": "slow down"})
    with pytest.raises(RateLimitError):
        invoke(is_async, "q")
    httpx2_mock.add_response(url=URL, json={"query": "q"})
    with pytest.raises(VeniceResponseError, match="results"):
        invoke(is_async, "q")


def test_cli_lists_results(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(
        method="POST",
        url=URL,
        match_json={"query": "venice ai", "limit": 5, "search_provider": "exa"},
        json={
            "query": "venice ai",
            "results": [
                RESULT,
                {"title": "Second", "url": "https://example.com", "content": "", "date": None},
            ],
        },
    )

    result = cli_runner.invoke(
        cli, ["venice", "search", "-p", "exa", "-n", "5", "venice", "ai"], color=False
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "1. Venice\n"
        "   https://venice.ai\n"
        "   Private and uncensored AI\n"
        "   2026-10-01\n"
        "\n"
        "2. Second\n"
        "   https://example.com\n"
    )


def test_cli_json_output_and_defaults(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(
        method="POST", url=URL, match_json={"query": "venice ai"}, json=PAYLOAD
    )

    result = cli_runner.invoke(cli, ["venice", "search", "--json", "venice ai"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == PAYLOAD


def test_cli_strips_control_characters(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(
        url=URL,
        json={
            "query": "q",
            "results": [
                {
                    "title": "Evil\x1b[2J\ntitle",
                    "url": "https://e.example\r",
                    "content": "line one\n\x9b31mline two",
                    "date": None,
                }
            ],
        },
    )

    result = cli_runner.invoke(cli, ["venice", "search", "q"], color=False)

    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.stdout and "\x9b" not in result.stdout
    assert "\r" not in result.stdout
    assert result.stdout == (
        "1. Evil[2J title\n   https://e.example\n   line one\n   31mline two\n"
    )


def test_cli_reports_no_results(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(url=URL, json={"query": "q", "results": []})

    result = cli_runner.invoke(cli, ["venice", "search", "q"])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "No results found." in result.stderr


@pytest.mark.parametrize(
    "args,message",
    [
        (["-p", "bing", "q"], "'bing' is not one of"),
        (["-n", "21", "q"], "21 is not in the range"),
        (["-n", "0", "q"], "0 is not in the range"),
        ([], "Missing argument 'QUERY...'"),
        (["x" * 401], "Invalid value for QUERY: String should have at most 400 characters"),
        ([""], "Invalid value for QUERY: String should have at least 1 character"),
    ],
)
def test_cli_rejects_invalid_input_before_http(
    cli_runner, httpx2_mock, mock_venice_api_key, args, message
):
    result = cli_runner.invoke(cli, ["venice", "search", *args])

    assert result.exit_code == 2
    assert message in result.output
    assert not httpx2_mock.get_requests()


def test_cli_http_error(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(url=URL, status_code=500, json={"message": "server unavailable"})

    result = cli_runner.invoke(cli, ["venice", "search", "q"])

    assert result.exit_code == 1
    assert "500" in result.output
    assert "server unavailable" in result.output
    assert "Traceback" not in result.output


def test_cli_network_error(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_exception(httpx2.TimeoutException("Request timed out"), url=URL)

    result = cli_runner.invoke(cli, ["venice", "search", "q"])

    assert result.exit_code == 1
    assert "timed out" in result.output.lower()
    assert "Traceback" not in result.output


def test_cli_missing_key(cli_runner, httpx2_mock, monkeypatch):
    monkeypatch.setattr("llm.get_key", lambda *args: None)

    result = cli_runner.invoke(cli, ["venice", "search", "q"])

    assert result.exit_code == 1
    assert "No key found for Venice" in result.output
    assert not httpx2_mock.get_requests()
