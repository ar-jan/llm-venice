import httpx
import pytest
from llm.cli import cli

from llm_venice.api.errors import VeniceAPIError
from llm_venice.api.refresh import fetch_models


def test_refresh_http_error_cli(cli_runner, httpx_mock, mock_venice_api_key):
    """CLI should present HTTP errors cleanly when refresh fails."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models?type=all",
        status_code=502,
        json={"message": "bad gateway"},
    )

    result = cli_runner.invoke(cli, ["venice", "refresh"])

    assert result.exit_code == 1
    output = result.output
    assert "502" in output
    assert "bad gateway" in output.lower()
    assert "Traceback" not in output


def test_refresh_network_error_cli(cli_runner, httpx_mock, mock_venice_api_key):
    """CLI should present network errors cleanly when refresh fails."""
    httpx_mock.add_exception(
        httpx.TimeoutException("Request timed out"),
        method="GET",
        url="https://api.venice.ai/api/v1/models?type=all",
    )

    result = cli_runner.invoke(cli, ["venice", "refresh"])

    assert result.exit_code == 1
    output = result.output
    assert "timed out" in output.lower()
    assert "Traceback" not in output


def test_fetch_models_http_error_programmatic(httpx_mock, mock_venice_api_key):
    """Programmatic refresh should raise VeniceAPIError."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models?type=all",
        status_code=401,
        json={"code": "INVALID_API_KEY"},
    )

    with pytest.raises(VeniceAPIError) as excinfo:
        fetch_models(mock_venice_api_key)

    message = str(excinfo.value)
    assert "401" in message
    assert "INVALID_API_KEY" in message
