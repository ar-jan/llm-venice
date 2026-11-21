import asyncio
import json

import httpx2
import llm
import pytest
from pydantic import ValidationError
from llm.cli import cli

from llm_venice.client import VeniceClient
from llm_venice.cli import characters as characters_cli
from llm_venice.client import AsyncVeniceClient
from llm_venice.errors import VeniceAPIError

CHARACTERS_URL = "https://api.venice.ai/api/v1/characters"
FIRST_PAGE = f"{CHARACTERS_URL}?limit=100&offset=0"


def test_characters_http_error_cli(cli_runner, httpx2_mock, mock_venice_api_key):
    """CLI should surface HTTP errors without traceback."""
    httpx2_mock.add_response(
        method="GET",
        url=FIRST_PAGE,
        status_code=500,
        json={"message": "server unavailable"},
    )

    result = cli_runner.invoke(cli, ["venice", "characters"])

    assert result.exit_code == 1
    output = result.output
    assert "500" in output
    assert "server unavailable" in output.lower()
    assert "Traceback" not in output


def test_characters_network_error_cli(cli_runner, httpx2_mock, mock_venice_api_key):
    """CLI should surface network errors without traceback."""
    httpx2_mock.add_exception(
        httpx2.TimeoutException("Request timed out"),
        method="GET",
        url=FIRST_PAGE,
    )

    result = cli_runner.invoke(cli, ["venice", "characters"])

    assert result.exit_code == 1
    output = result.output
    assert "timed out" in output.lower()
    assert "Traceback" not in output


def test_characters_http_error_programmatic(httpx2_mock, mock_venice_api_key):
    """Programmatic consumers should receive VeniceAPIError."""
    httpx2_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/characters",
        status_code=404,
        json={"error": "not found"},
    )

    with pytest.raises(VeniceAPIError) as excinfo:
        with VeniceClient(key=mock_venice_api_key) as client:
            client.list_characters()

    message = str(excinfo.value)
    assert "404" in message
    assert "not found" in message.lower()


def test_characters_cli_tolerates_optional_field_drift(
    cli_runner, httpx2_mock, mock_venice_api_key, tmp_path, monkeypatch
):
    """Unexpected optional values are saved as returned instead of failing the command."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    payload = {"data": [{"id": "c", "slug": "s", "modelId": None, "stats": None}]}
    httpx2_mock.add_response(method="GET", url=FIRST_PAGE, json=payload)

    result = cli_runner.invoke(cli, ["venice", "characters"])

    assert result.exit_code == 0, result.output
    assert json.loads((llm.user_dir() / "venice_characters.json").read_text()) == payload


def page(start, count):
    return {
        "object": "list",
        "data": [{"id": f"c{i}", "slug": f"s{i}"} for i in range(start, start + count)],
    }


def saved_characters():
    return json.loads((llm.user_dir() / "venice_characters.json").read_text())


def requested_offsets(httpx2_mock):
    return [request.url.params["offset"] for request in httpx2_mock.get_requests()]


def test_characters_cli_fetches_every_page(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(url=FIRST_PAGE, json=page(0, 100))
    httpx2_mock.add_response(url=f"{CHARACTERS_URL}?limit=100&offset=100", json=page(100, 30))

    result = cli_runner.invoke(cli, ["venice", "characters"])

    assert result.exit_code == 0, result.output
    assert "130 characters saved to" in result.stderr
    saved = saved_characters()
    assert saved["object"] == "list"
    assert [c["id"] for c in saved["data"]] == [f"c{i}" for i in range(130)]
    assert requested_offsets(httpx2_mock) == ["0", "100"]


def test_characters_cli_stops_at_empty_page(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(url=FIRST_PAGE, json=page(0, 100))
    httpx2_mock.add_response(url=f"{CHARACTERS_URL}?limit=100&offset=100", json=page(0, 0))

    result = cli_runner.invoke(cli, ["venice", "characters"])

    assert result.exit_code == 0, result.output
    assert len(saved_characters()["data"]) == 100
    assert requested_offsets(httpx2_mock) == ["0", "100"]


def test_characters_cli_stops_when_offset_is_ignored(cli_runner, httpx2_mock, mock_venice_api_key):
    """A server that returns the same full page again cannot loop forever."""
    httpx2_mock.add_response(json=page(0, 100), is_reusable=True)

    result = cli_runner.invoke(cli, ["venice", "characters"])

    assert result.exit_code == 0, result.output
    assert len(saved_characters()["data"]) == 100
    assert requested_offsets(httpx2_mock) == ["0", "100"]


def test_characters_cli_warns_when_page_cap_is_reached(
    cli_runner, httpx2_mock, mock_venice_api_key, monkeypatch
):
    monkeypatch.setattr(characters_cli, "MAX_CHARACTER_PAGES", 2)
    httpx2_mock.add_response(url=FIRST_PAGE, json=page(0, 100))
    httpx2_mock.add_response(url=f"{CHARACTERS_URL}?limit=100&offset=100", json=page(100, 100))

    result = cli_runner.invoke(cli, ["venice", "characters"])

    assert result.exit_code == 0, result.output
    assert "Warning: stopped after 2 pages; the saved list may be incomplete" in result.stderr
    assert len(saved_characters()["data"]) == 200


def test_characters_cli_sends_filters_with_each_page(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(json=page(0, 3))

    result = cli_runner.invoke(
        cli, ["venice", "characters", "--adult", "false", "--web-enabled", "true"]
    )

    assert result.exit_code == 0, result.output
    (request,) = httpx2_mock.get_requests()
    assert dict(request.url.params) == {
        "isWebEnabled": "true",
        "isAdult": "false",
        "limit": "100",
        "offset": "0",
    }


@pytest.mark.parametrize("is_async", [False, True])
def test_list_characters_pages_with_limit_and_offset(is_async, httpx2_mock):
    httpx2_mock.add_response(url=f"{CHARACTERS_URL}?limit=10&offset=20", json=page(20, 10))

    async def run_async():
        async with AsyncVeniceClient(key="k") as client:
            return await client.list_characters(limit=10, offset=20)

    def run_sync():
        with VeniceClient(key="k") as client:
            return client.list_characters(limit=10, offset=20)

    characters = asyncio.run(run_async()) if is_async else run_sync()

    assert [c["id"] for c in characters] == [f"c{i}" for i in range(20, 30)]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "kwargs",
    [{"limit": 0}, {"limit": 101}, {"limit": True}, {"limit": "10"}, {"offset": -1}],
    ids=["limit-zero", "limit-over-max", "limit-bool", "limit-str", "negative-offset"],
)
def test_list_characters_rejects_invalid_paging_before_request(is_async, kwargs, httpx2_mock):
    async def run_async():
        async with AsyncVeniceClient(key="k") as client:
            await client.list_characters(**kwargs)

    def run_sync():
        with VeniceClient(key="k") as client:
            client.list_characters(**kwargs)

    with pytest.raises(ValidationError):
        asyncio.run(run_async()) if is_async else run_sync()
    assert httpx2_mock.get_requests() == []
