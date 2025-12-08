import json
import pathlib

from jsonschema import Draft202012Validator
from llm.cli import cli
import pytest

from llm_venice.api.errors import VeniceAPIError
from llm_venice.api.keys import list_api_keys


def test_list_api_keys_http_error(cli_runner, httpx_mock, mock_venice_api_key):
    """Handle unexpected HTTP errors without a traceback."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys",
        status_code=401,
        json={"message": "admin privileges required"},
    )

    result = cli_runner.invoke(cli, ["venice", "api-keys", "list"])

    assert result.exit_code == 1
    assert "401" in result.output
    assert "admin privileges required" in result.output.lower()
    assert "Traceback" not in result.output


def test_list_api_keys_http_error_programmatic(httpx_mock, mock_venice_api_key):
    """Programmatic consumers should get a friendly VeniceAPIError."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys",
        status_code=403,
        json={"error": "forbidden"},
    )

    with pytest.raises(VeniceAPIError) as excinfo:
        list_api_keys({"Authorization": f"Bearer {mock_venice_api_key}"})

    message = str(excinfo.value)
    assert "403" in message
    assert "forbidden" in message.lower()
    assert "admin" not in message.lower()


def test_list_api_keys_maps_error_code(httpx_mock, mock_venice_api_key):
    """Ensure error code is surfaced for 401 responses."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys",
        status_code=401,
        json={"code": "INVALID_API_KEY"},
    )

    with pytest.raises(VeniceAPIError) as excinfo:
        list_api_keys({"Authorization": f"Bearer {mock_venice_api_key}"})

    message = str(excinfo.value)
    assert "INVALID_API_KEY" in message
    assert "admin" not in message.lower()


api_keys_rate_limits_path = pathlib.Path(__file__).parent / "schemas" / "api_keys_rate_limits.json"
with open(api_keys_rate_limits_path) as f:
    api_keys_rate_limits_schema = json.load(f)


@pytest.mark.integration
def test_rate_limits(cli_runner):
    """Test that 'api-keys rate-limits' output matches expected schema"""
    result = cli_runner.invoke(cli, ["venice", "api-keys", "rate-limits"])

    assert result.exit_code == 0

    try:
        data = json.loads(result.output)
        # jsonschema validate shows full response data on error
        validator = Draft202012Validator(api_keys_rate_limits_schema)
        errors = list(validator.iter_errors(data))
        if errors:
            error = errors[0]
            error_path = " -> ".join(str(p) for p in error.path)
            error_message = f"Schema validation failed at path: {error_path}"
            pytest.fail(error_message)
    except json.JSONDecodeError:
        pytest.fail("Response was not valid JSON")
