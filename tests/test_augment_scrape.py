"""Web page scraping through /augment/scrape: contract, clients, CLI and tool."""

import asyncio
import json
from pathlib import Path

import httpx2
import llm
import pytest
from llm.cli import cli
from pydantic import ValidationError
from typing_extensions import assert_type

from llm_venice import venice_augment_scrape
from llm_venice._endpoints import AugmentScrapeInput, json_operation, parse_envelope
from llm_venice.client import AsyncVeniceClient, VeniceClient
from llm_venice.errors import VeniceAPIError, VeniceResponseError
from llm_venice.types import AugmentScrapeResponse

URL = "https://api.venice.ai/api/v1/augment/scrape"
FIXTURE = Path(__file__).parent / "fixtures/openapi-2026-11-22.json"
PAGE = {"url": "https://example.com", "content": "# Example\n\nText.", "format": "markdown"}


def invoke(is_async, url):
    if is_async:

        async def run():
            async with AsyncVeniceClient(key="test") as client:
                return await client.augment_scrape(url)

        return asyncio.run(run())
    with VeniceClient(key="test") as client:
        return client.augment_scrape(url)


def parse(payload):
    response = httpx2.Response(200, json=payload, request=httpx2.Request("POST", URL))
    return parse_envelope(response, "augment_scrape", "Scraping web page")


def test_request_is_the_url_as_given():
    method, path, action, kwargs = json_operation(
        "augment_scrape", {"url": "HTTPS://Example.com/a b?q=1"}
    )
    assert (method, path, action) == ("POST", "augment/scrape", "Scraping web page")
    assert kwargs == {"json": {"url": "HTTPS://Example.com/a b?q=1"}}


@pytest.mark.parametrize(
    "url", ["", "example.com", "ftp://example.com/file", "https://", "mailto:a@b.c", None, 1]
)
def test_url_must_be_an_absolute_web_url(url):
    with pytest.raises(ValidationError):
        AugmentScrapeInput.model_validate({"url": url})


def test_request_and_response_match_dated_fixture():
    schemas = json.loads(FIXTURE.read_text())["schemas"]
    resource = json.loads(FIXTURE.read_text())["resources"]["POST /augment/scrape"]
    request = schemas["WebScrapeRequest"]
    assert set(AugmentScrapeInput.model_fields) == set(request["properties"]) == {"url"}
    assert request["required"] == ["url"]
    response = schemas["WebScrapeResponse"]
    assert set(AugmentScrapeResponse.__annotations__) == set(response["properties"])
    assert AugmentScrapeResponse.__required_keys__ <= set(response["required"])
    assert resource["errors"]["502"]["description"] == "The URL could not be scraped."


def test_response_keeps_unknown_fields():
    assert parse({**PAGE, "title": "Example"}) == {**PAGE, "title": "Example"}


@pytest.mark.parametrize(
    "payload,detail",
    [
        ({"url": "u"}, "content"),
        ({"content": "c"}, "url"),
        ({"url": "u", "content": None}, "content"),
        ([PAGE], "expected a JSON object"),
    ],
)
def test_malformed_responses(payload, detail):
    with pytest.raises(VeniceResponseError, match=detail):
        parse(payload)


@pytest.mark.parametrize("is_async", [False, True])
def test_client_parity(is_async, httpx2_mock):
    httpx2_mock.add_response(
        method="POST", url=URL, match_json={"url": "https://example.com"}, json=PAGE
    )
    result = invoke(is_async, "https://example.com")
    assert_type(result, AugmentScrapeResponse)
    assert result == PAGE
    assert httpx2_mock.get_request().headers["authorization"] == "Bearer test"


@pytest.mark.parametrize("is_async", [False, True])
def test_client_errors(is_async, httpx2_mock):
    with pytest.raises(ValidationError):
        invoke(is_async, "example.com")
    assert not httpx2_mock.get_requests()
    httpx2_mock.add_response(
        url=URL,
        status_code=400,
        json={"error": "Reddit blocks automated access to their content."},
    )
    with pytest.raises(VeniceAPIError, match="Scraping web page failed: .*400.*Reddit blocks"):
        invoke(is_async, "https://www.reddit.com/r/python")
    httpx2_mock.add_response(url=URL, status_code=502, json={"error": "Could not resolve hostname"})
    with pytest.raises(VeniceAPIError) as caught:
        invoke(is_async, "https://nonexistent.invalid/")
    assert caught.value.status_code == 502
    assert caught.value.detail == "Could not resolve hostname"


def test_cli_prints_the_markdown(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(
        url=URL,
        match_json={"url": "https://example.com"},
        json={**PAGE, "content": "# Evil\x1b[2J title\r\n\tindented\x07\n"},
    )
    result = cli_runner.invoke(cli, ["venice", "scrape", "https://example.com"])
    assert result.exit_code == 0, result.output
    # Control characters other than newlines and tabs never reach the terminal.
    assert result.stdout == "# Evil[2J title\n\tindented\n"


def test_cli_json_output(cli_runner, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(url=URL, json=PAGE)
    result = cli_runner.invoke(cli, ["venice", "scrape", "--json", "https://example.com"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == PAGE


def test_cli_errors(cli_runner, httpx2_mock, mock_venice_api_key):
    result = cli_runner.invoke(cli, ["venice", "scrape", "example.com"])
    assert result.exit_code == 2
    assert "Invalid value for 'URL': url must be an absolute http or https URL" in result.output
    httpx2_mock.add_response(url=URL, status_code=502, json={"error": "Could not resolve hostname"})
    result = cli_runner.invoke(cli, ["venice", "scrape", "https://nonexistent.invalid/"])
    assert result.exit_code == 1
    assert result.output == (
        "Error: Scraping web page failed: Venice API returned 502 Bad Gateway"
        " - Could not resolve hostname\n"
    )


def test_tool_is_registered_with_its_schema():
    tool = llm.get_tools()["venice_augment_scrape"]
    assert isinstance(tool, llm.Tool)
    assert tool.implementation is venice_augment_scrape
    assert tool.plugin == "venice"
    assert tool.input_schema == {
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "type": "object",
    }
    assert tool.description and "\n " not in tool.description
    assert tool.description.startswith("Fetch a web page through Venice")


def test_tool_scrapes(httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(url=URL, match_json={"url": "https://example.com"}, json=PAGE)
    assert venice_augment_scrape(url="https://example.com") == PAGE


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({}, "url: Field required"),
        ({"url": "example.com"}, "url: Value error, url must be an absolute http or https URL"),
        ({"url": "https://a.b", "limit": 3}, "limit: unexpected argument"),
    ],
)
def test_tool_rejects_invalid_input_before_http(kwargs, message, httpx2_mock):
    with pytest.raises(ValueError, match="^Invalid scrape input: ") as caught:
        venice_augment_scrape(**kwargs)
    assert message in str(caught.value)
    assert not httpx2_mock.get_requests()


def test_tool_failure_raises_for_llm_to_report(httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(url=URL, status_code=502, json={"error": "Could not resolve hostname"})
    with pytest.raises(VeniceAPIError, match="502"):
        venice_augment_scrape(url="https://nonexistent.invalid/")
