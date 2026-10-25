import httpx2
import pytest
from llm.cli import cli

from llm_venice.api.errors import VeniceAPIError
from llm_venice.api.refresh import fetch_models


def test_refresh_http_error_cli(cli_runner, httpx2_mock, mock_venice_api_key):
    """CLI should present HTTP errors cleanly when refresh fails."""
    httpx2_mock.add_response(
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


def test_refresh_network_error_cli(cli_runner, httpx2_mock, mock_venice_api_key):
    """CLI should present network errors cleanly when refresh fails."""
    httpx2_mock.add_exception(
        httpx2.TimeoutException("Request timed out"),
        method="GET",
        url="https://api.venice.ai/api/v1/models?type=all",
    )

    result = cli_runner.invoke(cli, ["venice", "refresh"])

    assert result.exit_code == 1
    output = result.output
    assert "timed out" in output.lower()
    assert "Traceback" not in output


def test_fetch_models_http_error_programmatic(httpx2_mock, mock_venice_api_key):
    """Programmatic refresh should raise VeniceAPIError."""
    httpx2_mock.add_response(
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


def test_persist_models_is_atomic(tmp_path, monkeypatch):
    """An interrupted write must leave the previous cache intact."""
    import json

    from llm_venice.api.refresh import persist_models

    target = tmp_path / "venice_models.json"
    old = [{"id": "old"}]
    persist_models(old, target)
    assert json.loads(target.read_text()) == old

    def failing_dump(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("llm_venice.utils.json.dump", failing_dump)
    with pytest.raises(KeyboardInterrupt):
        persist_models([{"id": "new"}], target)

    assert json.loads(target.read_text()) == old
    # No temporary files left behind
    assert [p.name for p in tmp_path.iterdir()] == ["venice_models.json"]


def test_fetch_models_malformed_payload_raises_value_error(httpx2_mock, mock_venice_api_key):
    """A response without a model list should raise ValueError, not KeyError."""
    httpx2_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models?type=all",
        json={"unexpected": True},
    )

    with pytest.raises(ValueError):
        fetch_models(mock_venice_api_key)
