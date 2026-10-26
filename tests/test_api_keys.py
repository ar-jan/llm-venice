import httpx2
import json
import pathlib

from jsonschema import Draft202012Validator
from llm.cli import cli
import pytest

from llm_venice.errors import VeniceAPIError
from llm_venice.client import VeniceClient


@pytest.mark.parametrize(
    "args,method,path,data",
    [
        (["list"], "GET", "api_keys", [{"id": "key", "future": {"flag": True}}]),
        (["rate-limits"], "GET", "api_keys/rate_limits", {"accessPermitted": True}),
        (
            ["rate-limits-log"],
            "GET",
            "api_keys/rate_limits/log",
            [{"apiKeyId": "key", "rateLimitType": "RPM", "timestamp": "2026-10-26T00:00:00Z"}],
        ),
        (["create", "--type", "ADMIN"], "POST", "api_keys", {"id": "key", "apiKey": "created"}),
        (["delete", "key"], "DELETE", "api_keys?id=key", None),
    ],
)
def test_api_key_cli_preserves_json_envelope_bytes(
    cli_runner, httpx2_mock, mock_venice_api_key, args, method, path, data
):
    payload = {"success": True, "data": data, "future": {"message": "café"}}
    httpx2_mock.add_response(
        method=method, url=f"https://api.venice.ai/api/v1/{path}", json=payload
    )

    result = cli_runner.invoke(cli, ["venice", "api-keys", *args])

    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == (json.dumps(payload, indent=2) + "\n").encode()


def test_list_api_keys_http_error(cli_runner, httpx2_mock, mock_venice_api_key):
    """Handle unexpected HTTP errors without a traceback."""
    httpx2_mock.add_response(
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


@pytest.mark.parametrize(
    ("args", "method", "url"),
    [
        (["venice", "api-keys", "list"], "GET", "https://api.venice.ai/api/v1/api_keys"),
        (
            ["venice", "api-keys", "rate-limits"],
            "GET",
            "https://api.venice.ai/api/v1/api_keys/rate_limits",
        ),
        (
            ["venice", "api-keys", "rate-limits-log"],
            "GET",
            "https://api.venice.ai/api/v1/api_keys/rate_limits/log",
        ),
        (
            ["venice", "api-keys", "create", "--type", "ADMIN"],
            "POST",
            "https://api.venice.ai/api/v1/api_keys",
        ),
        (
            ["venice", "api-keys", "delete", "example-key"],
            "DELETE",
            "https://api.venice.ai/api/v1/api_keys?id=example-key",
        ),
    ],
)
def test_api_keys_network_error_cli(
    cli_runner, httpx2_mock, mock_venice_api_key, args, method, url
):
    """CLI should surface network errors without traceback."""
    httpx2_mock.add_exception(httpx2.TimeoutException("Request timed out"), method=method, url=url)

    result = cli_runner.invoke(cli, args)

    assert result.exit_code == 1
    output = result.output
    assert "timed out" in output.lower()
    assert "Traceback" not in output


def test_list_api_keys_http_error_programmatic(httpx2_mock, mock_venice_api_key):
    """Programmatic consumers should get a friendly VeniceAPIError."""
    httpx2_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys",
        status_code=403,
        json={"error": "forbidden"},
    )

    with pytest.raises(VeniceAPIError) as excinfo:
        with VeniceClient(key=mock_venice_api_key) as client:
            client.list_api_keys()

    message = str(excinfo.value)
    assert "403" in message
    assert "forbidden" in message.lower()
    assert "admin" not in message.lower()


def test_list_api_keys_maps_error_code(httpx2_mock, mock_venice_api_key):
    """Ensure error code is surfaced for 401 responses."""
    httpx2_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys",
        status_code=401,
        json={"code": "INVALID_API_KEY"},
    )

    with pytest.raises(VeniceAPIError) as excinfo:
        with VeniceClient(key=mock_venice_api_key) as client:
            client.list_api_keys()

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
