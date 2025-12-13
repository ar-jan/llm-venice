import httpx
import pytest
from llm.cli import cli

from llm_venice.api.characters import list_characters
from llm_venice.api.errors import VeniceAPIError


def test_characters_http_error_cli(cli_runner, httpx_mock, mock_venice_api_key):
    """CLI should surface HTTP errors without traceback."""
    httpx_mock.add_response(
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


def test_characters_network_error_cli(cli_runner, httpx_mock, mock_venice_api_key):
    """CLI should surface network errors without traceback."""
    httpx_mock.add_exception(
        httpx.TimeoutException("Request timed out"),
        method="GET",
        url="https://api.venice.ai/api/v1/characters",
    )

    result = cli_runner.invoke(cli, ["venice", "characters"])

    assert result.exit_code == 1
    output = result.output
    assert "timed out" in output.lower()
    assert "Traceback" not in output


def test_characters_http_error_programmatic(httpx_mock, mock_venice_api_key):
    """Programmatic consumers should receive VeniceAPIError."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/characters",
        status_code=404,
        json={"error": "not found"},
    )

    with pytest.raises(VeniceAPIError) as excinfo:
        list_characters(mock_venice_api_key)

    message = str(excinfo.value)
    assert "404" in message
    assert "not found" in message.lower()
