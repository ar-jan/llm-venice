import asyncio
import httpx2
import json
import pathlib

from jsonschema import Draft202012Validator
from llm.cli import cli
import pytest

from llm_venice.errors import VeniceAPIError, VeniceResponseError
from llm_venice.client import AsyncVeniceClient, VeniceClient


@pytest.mark.parametrize(
    "args,method,path,data",
    [
        (["list"], "GET", "api_keys", [{"id": "key", "future": {"flag": True}}]),
        # Optional fields with undocumented values still render unchanged.
        (["list"], "GET", "api_keys", [{"id": "key", "description": None, "ipAllowlist": 1}]),
        (["rate-limits"], "GET", "api_keys/rate_limits", {"accessPermitted": True}),
        (
            ["rate-limits"],
            "GET",
            "api_keys/rate_limits",
            {"accessPermitted": True, "balances": {"USD": None}, "apiTier": None},
        ),
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


def test_create_cli_sends_new_key_settings(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/api_keys",
        json={"success": True, "data": {"id": "key", "apiKey": "created"}},
    )

    result = cli_runner.invoke(
        cli,
        [
            "venice",
            "api-keys",
            "create",
            "--type",
            "INFERENCE",
            "--description",
            "ci",
            "--expiration-date",
            "2026-12-31",
            "--limits-usd",
            "5",
            "--limits-diem",
            "1.5",
            "--limit-period",
            "MONTH",
            "--model-privacy",
            "PRIVATE_ONLY",
            "--ip-allowlist",
            "11.22.33.44",
            "--ip-allowlist",
            "2a00:1::/48",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(httpx2_mock.get_request().read()) == {
        "apiKeyType": "INFERENCE",
        "description": "ci",
        "expiresAt": "2026-12-31T00:00:00.000Z",
        "consumptionLimit": {"usd": 5.0, "diem": 1.5},
        "limitPeriod": "MONTH",
        "modelPrivacy": "PRIVATE_ONLY",
        "ipAllowlist": ["11.22.33.44", "2a00:1::/48"],
    }


@pytest.mark.parametrize(
    "args",
    [
        ["--limit-period", "WEEK"],
        ["--model-privacy", "TEE"],
        ["--limits-diem", "-1"],
        ["--ip-allowlist", "example.com"],
    ],
)
def test_create_cli_rejects_invalid_settings_before_request(
    cli_runner, httpx2_mock, mock_venice_api_key, args
):
    result = cli_runner.invoke(cli, ["venice", "api-keys", "create", "--type", "ADMIN", *args])

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert not httpx2_mock.get_requests()


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


@pytest.mark.parametrize(
    "payload",
    [{"success": False}, {}, {"success": "true"}, {"success": 1}],
    ids=["false", "missing", "string", "int"],
)
def test_delete_without_confirmation_fails(cli_runner, httpx2_mock, mock_venice_api_key, payload):
    """Only an explicit success: true confirms that the key was revoked."""
    httpx2_mock.add_response(
        method="DELETE", url="https://api.venice.ai/api/v1/api_keys?id=key", json=payload
    )

    result = cli_runner.invoke(cli, ["venice", "api-keys", "delete", "key"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Deleting API key failed: Venice did not confirm the deletion" in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("payload", [{"success": False}, {}])
def test_delete_api_key_requires_confirmation(is_async, httpx2_mock, payload):
    httpx2_mock.add_response(json=payload)

    async def run_async():
        async with AsyncVeniceClient(key="k") as client:
            await client.delete_api_key("key")

    def run_sync():
        with VeniceClient(key="k") as client:
            client.delete_api_key("key")

    with pytest.raises(VeniceResponseError, match="did not confirm"):
        asyncio.run(run_async()) if is_async else run_sync()


@pytest.mark.parametrize("is_async", [False, True])
def test_delete_api_key_accepts_no_content(is_async, httpx2_mock):
    httpx2_mock.add_response(status_code=204)

    async def run_async():
        async with AsyncVeniceClient(key="k") as client:
            return await client.delete_api_key("key")

    def run_sync():
        with VeniceClient(key="k") as client:
            return client.delete_api_key("key")

    assert (asyncio.run(run_async()) if is_async else run_sync()) is None
