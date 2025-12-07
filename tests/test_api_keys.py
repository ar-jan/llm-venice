import json
import pathlib

from jsonschema import Draft202012Validator
from llm.cli import cli
import pytest


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
