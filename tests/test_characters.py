import json

import httpx2
import llm
import pytest
from llm.cli import cli

from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceAPIError


def test_characters_http_error_cli(cli_runner, httpx2_mock, mock_venice_api_key):
    """CLI should surface HTTP errors without traceback."""
    httpx2_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/characters",
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
        url="https://api.venice.ai/api/v1/characters",
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
    httpx2_mock.add_response(
        method="GET", url="https://api.venice.ai/api/v1/characters", json=payload
    )

    result = cli_runner.invoke(cli, ["venice", "characters"])

    assert result.exit_code == 0, result.output
    assert json.loads((llm.user_dir() / "venice_characters.json").read_text()) == payload
