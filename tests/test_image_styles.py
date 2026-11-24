"""Image style presets through /image/styles: clients and CLI."""

import asyncio
import json
from pathlib import Path

import pytest
from llm.cli import cli
from typing_extensions import assert_type

from llm_venice.client import AsyncVeniceClient, VeniceClient
from llm_venice.errors import VeniceAPIError, VeniceResponseError

URL = "https://api.venice.ai/api/v1/image/styles"
FIXTURE = Path(__file__).parent / "fixtures/openapi-2026-11-22.json"
STYLES = {"object": "list", "data": ["3D Model", "Analog Film", "Watercolor"]}


def invoke(is_async):
    if is_async:

        async def run():
            async with AsyncVeniceClient(key="test") as client:
                return await client.list_image_styles()

        return asyncio.run(run())
    with VeniceClient(key="test") as client:
        return client.list_image_styles()


def test_response_matches_dated_fixture():
    resource = json.loads(FIXTURE.read_text())["resources"]["GET /image/styles"]
    schema = resource["response"]["content"]["application/json"]["schema"]
    assert schema["properties"]["data"]["items"] == {"type": "string"}
    assert resource["requestBody"] is None and "parameters" not in resource


@pytest.mark.parametrize("is_async", [False, True])
def test_client_parity(is_async, httpx2_mock):
    httpx2_mock.add_response(method="GET", url=URL, json=STYLES)
    styles = invoke(is_async)
    assert_type(styles, list[str])
    assert styles == STYLES["data"]
    assert httpx2_mock.get_request().headers["authorization"] == "Bearer test"


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "body,detail", [({"data": ["a", 1]}, "must be strings"), ({"object": "list"}, "data array")]
)
def test_malformed_responses(is_async, body, detail, httpx2_mock):
    httpx2_mock.add_response(url=URL, json=body)
    with pytest.raises(VeniceResponseError, match=detail):
        invoke(is_async)


def test_cli_lists_one_style_per_line(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(url=URL, json={"data": ["Anime", "Evil\x1b[2J\nstyle"]})
    result = cli_runner.invoke(cli, ["venice", "image-styles"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "Anime\nEvil[2J style\n"


def test_cli_json_and_errors(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(url=URL, json=STYLES)
    result = cli_runner.invoke(cli, ["venice", "image-styles", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == STYLES
    httpx2_mock.add_response(url=URL, status_code=500, json={"error": "Unknown error"})
    result = cli_runner.invoke(cli, ["venice", "image-styles"])
    assert result.exit_code == 1
    assert result.output.startswith("Error: Listing image styles failed: Venice API returned 500")


def test_api_errors_raise(httpx2_mock):
    httpx2_mock.add_response(url=URL, status_code=401, json={"error": "Authentication failed"})
    with pytest.raises(VeniceAPIError, match="401"):
        invoke(False)
