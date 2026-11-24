"""Document text extraction through /augment/text-parser: contract, clients and CLI."""

import asyncio
import io
import json
from pathlib import Path

import httpx2
import pytest
from llm.cli import cli
from pydantic import ValidationError
from typing_extensions import assert_type

from llm_venice._endpoints import TextParserInput, document_part, parse_text_parser
from llm_venice.client import AsyncVeniceClient, VeniceClient, _document_file
from llm_venice.errors import VeniceAPIError, VeniceResponseError
from llm_venice.types import AugmentTextParserResponse

URL = "https://api.venice.ai/api/v1/augment/text-parser"
FIXTURE = Path(__file__).parent / "fixtures/openapi-2026-11-22.json"
PDF = b"%PDF-1.4\n%binary\x00\xff"
DOCX = b"PK\x03\x04\x14\x00docx"
PARSED = {"text": "Hello Venice", "tokens": 3}


def invoke(is_async, *args, **kwargs):
    if is_async:

        async def run():
            async with AsyncVeniceClient(key="test") as client:
                return await client.augment_parse_text(*args, **kwargs)

        return asyncio.run(run())
    with VeniceClient(key="test") as client:
        return client.augment_parse_text(*args, **kwargs)


def test_named_documents_are_labelled_by_their_extension():
    assert document_part("report.pdf", b"x") == ("report.pdf", b"x", "application/pdf")
    assert document_part("data.json", b"{}")[2] == "application/json"
    # Venice reads the extension first, so a known name wins over the bytes.
    assert document_part("x.txt", PDF)[2] == "text/plain"


@pytest.mark.parametrize(
    "content,content_type",
    [
        (PDF, "application/pdf"),
        (b"plain notes\n", "text/plain"),
        ("café ".encode() * 1000 + "é".encode(), "text/plain"),
        (b"text with a \x00 byte", "application/octet-stream"),
        (DOCX, "application/octet-stream"),
        (b"\xff\xfe\xfd", "application/octet-stream"),
    ],
)
def test_unnamed_documents_are_labelled_from_their_bytes(content, content_type):
    assert document_part("document", content)[2] == content_type


def test_a_character_cut_off_by_the_sample_still_counts_as_text():
    # "é" is two bytes; the 4096-byte sample ends between them.
    content = b"a" * 4095 + "é".encode()
    assert document_part("document", content)[2] == "text/plain"


def test_document_files_keep_their_name(tmp_path):
    path = tmp_path / "report.docx"
    path.write_bytes(DOCX)
    assert _document_file(path, None)[0] == "report.docx"
    assert _document_file(str(path), "renamed.pdf")[:2] == ("renamed.pdf", DOCX)
    assert _document_file(DOCX, None)[0] == "document"
    assert _document_file(DOCX, "x.docx")[0] == "x.docx"
    with path.open("rb") as handle:
        handle.read(2)
        # A borrowed file is read from its cursor, named by its basename and left open.
        assert _document_file(handle, None)[:2] == ("report.docx", DOCX[2:])
        assert not handle.closed
    assert _document_file(io.BytesIO(b"abc"), None) == ("document", b"abc", "text/plain")
    with path.open("r", encoding="latin-1") as text_mode:
        with pytest.raises(TypeError, match="binary mode"):
            _document_file(text_mode, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "inputs", [{"response_format": "xml"}, {"filename": ""}, {"filename": 1}, {"other": 1}]
)
def test_options_are_validated(inputs):
    with pytest.raises(ValidationError):
        TextParserInput.model_validate(inputs)


def test_request_and_response_match_dated_fixture():
    resource = json.loads(FIXTURE.read_text())["resources"]["POST /augment/text-parser"]
    request = resource["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert set(request["properties"]) == {"file", "response_format"}
    assert request["required"] == ["file"]
    assert set(request["properties"]["response_format"]["enum"]) == {"json", "text"}
    assert set(TextParserInput.model_fields) == {"filename", "response_format"}
    content = resource["response"]["content"]
    schema = content["application/json"]["schema"]
    assert set(AugmentTextParserResponse.__annotations__) == set(schema["properties"])
    assert AugmentTextParserResponse.__required_keys__ <= set(schema["required"])
    assert content["text/plain"]["schema"]["type"] == "string"


def test_text_responses():
    request = httpx2.Request("POST", URL)
    plain = httpx2.Response(
        200, text="Hello", headers={"content-type": "text/plain"}, request=request
    )
    assert parse_text_parser(plain, "text") == "Hello"
    as_json = httpx2.Response(200, json=PARSED, request=request)
    with pytest.raises(VeniceResponseError, match="expected plain text but received JSON"):
        parse_text_parser(as_json, "text")
    assert parse_text_parser(as_json, "json") == PARSED
    drifted = {"text": "", "tokens": None, "future": 1}
    assert parse_text_parser(httpx2.Response(200, json=drifted, request=request), "json") == drifted
    for body in ({"tokens": 3}, {"text": None}, ["text"]):
        with pytest.raises(VeniceResponseError):
            parse_text_parser(httpx2.Response(200, json=body, request=request), "json")


@pytest.mark.parametrize("is_async", [False, True])
def test_client_sends_the_named_file_and_format(is_async, httpx2_mock, multipart_parts, tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(PDF)
    httpx2_mock.add_response(method="POST", url=URL, json=PARSED)
    assert invoke(is_async, path) == PARSED
    parts = multipart_parts()
    assert parts["file"] == [("report.pdf", "application/pdf", PDF)]
    assert parts["response_format"] == [(None, "text/plain", b"json")]
    assert httpx2_mock.get_request().headers["authorization"] == "Bearer test"


@pytest.mark.parametrize("is_async", [False, True])
def test_client_returns_plain_text(is_async, httpx2_mock, multipart_parts):
    httpx2_mock.add_response(url=URL, text="Slide one", headers={"content-type": "text/plain"})
    result = invoke(is_async, DOCX, filename="deck.pptx", response_format="text")
    assert result == "Slide one"
    parts = multipart_parts()
    assert parts["file"][0][0] == "deck.pptx"
    assert parts["response_format"][0][2] == b"text"


def test_the_format_selects_the_return_type(httpx2_mock):
    httpx2_mock.add_response(url=URL, json=PARSED)
    httpx2_mock.add_response(url=URL, text="Hello", headers={"content-type": "text/plain"})
    with VeniceClient(key="test") as client:
        assert_type(client.augment_parse_text(PDF), AugmentTextParserResponse)
        assert_type(client.augment_parse_text(PDF, response_format="text"), str)


@pytest.mark.parametrize("is_async", [False, True])
def test_client_errors(is_async, httpx2_mock):
    with pytest.raises(ValidationError):
        invoke(is_async, PDF, response_format="xml")
    assert not httpx2_mock.get_requests()
    httpx2_mock.add_response(
        url=URL, status_code=400, json={"error": 'Unsupported file type "image/png".'}
    )
    with pytest.raises(VeniceAPIError, match="Parsing document failed: .*400.*Unsupported"):
        invoke(is_async, b"\x89PNG\r\n\x1a\n", filename="x.png")


def test_cli_prints_the_text(
    cli_runner, httpx2_mock, mock_venice_api_key, multipart_parts, tmp_path
):
    path = tmp_path / "report.pdf"
    path.write_bytes(PDF)
    httpx2_mock.add_response(
        url=URL, text="Page one\x0c\nPage two\x1b[31m\n", headers={"content-type": "text/plain"}
    )
    result = cli_runner.invoke(cli, ["venice", "parse", str(path)])
    assert result.exit_code == 0, result.output
    assert result.stdout == "Page one\nPage two[31m\n"
    assert multipart_parts()["response_format"][0][2] == b"text"


def test_cli_json_output(cli_runner, httpx2_mock, mock_venice_api_key, tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n")
    httpx2_mock.add_response(url=URL, json=PARSED)
    result = cli_runner.invoke(cli, ["venice", "parse", "--format", "json", str(path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == PARSED


def test_cli_errors(cli_runner, httpx2_mock, mock_venice_api_key, tmp_path):
    result = cli_runner.invoke(cli, ["venice", "parse", str(tmp_path / "missing.pdf")])
    assert result.exit_code == 2
    assert "does not exist" in result.output
    path = tmp_path / "x.png"
    path.write_bytes(b"\x89PNG")
    httpx2_mock.add_response(
        url=URL, status_code=400, json={"error": 'Unsupported file type "image/png".'}
    )
    result = cli_runner.invoke(cli, ["venice", "parse", str(path)])
    assert result.exit_code == 1
    assert result.output == (
        "Error: Parsing document failed: Venice API returned 400 Bad Request"
        ' - Unsupported file type "image/png".\n'
    )
