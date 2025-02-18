import json
import pathlib

from click.testing import CliRunner
from jsonschema import validate, ValidationError
import llm
from llm.cli import cli
from llm.plugins import load_plugins
import pytest


api_keys_rate_limits_path = (
    pathlib.Path(__file__).parent / "schemas" / "api_keys_rate_limits.json"
)
with open(api_keys_rate_limits_path) as f:
    api_keys_rate_limits_schema = json.load(f)


def test_rate_limits():
    load_plugins()
    """Test that 'api-keys rate-limits' output matches expected schema"""
    runner = CliRunner()
    result = runner.invoke(cli, ["venice", "api-keys", "rate-limits"])

    assert result.exit_code == 0
    data = json.loads(result.output)

    try:
        validate(instance=data, schema=api_keys_rate_limits_schema)
    except ValidationError as e:
        pytest.fail(f"Response did not match schema: {str(e)}")
