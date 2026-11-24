"""Background removal through /image/background-remove: contract, clients and CLI."""

import asyncio
import io
import json
from pathlib import Path

import pytest
from llm.cli import cli
from pydantic import ValidationError
from typing_extensions import assert_type

from llm_venice import GeneratedImage
from llm_venice._endpoints import BackgroundRemoveInput
from llm_venice.client import AsyncVeniceClient, VeniceClient
from llm_venice.errors import VeniceAPIError, VeniceResponseError

URL = "https://api.venice.ai/api/v1/image/background-remove"
FIXTURE = Path(__file__).parent / "fixtures/openapi-2026-11-22.json"
PNG = b"\x89PNG\r\n\x1a\nsource"
RESULT = b"\x89PNG\r\n\x1a\nno background"
IMAGE_URL = "https://example.com/photos/cat%20one.jpg?size=large"


def invoke(is_async, *args, **kwargs):
    if is_async:

        async def run():
            async with AsyncVeniceClient(key="test") as client:
                return await client.remove_image_background(*args, **kwargs)

        return asyncio.run(run())
    with VeniceClient(key="test") as client:
        return client.remove_image_background(*args, **kwargs)


def respond(httpx2_mock, content=RESULT, content_type="image/png"):
    httpx2_mock.add_response(
        method="POST", url=URL, content=content, headers={"content-type": content_type}
    )


def test_requests_match_dated_fixture():
    fixture = json.loads(FIXTURE.read_text())
    upload, by_url = fixture["schemas"]["BackgroundRemoveImageRequest"]["anyOf"]
    multipart = fixture["schemas"]["BackgroundRemoveImageMultipartRequest"]
    url_body = BackgroundRemoveInput(image_url="https://a.b/c.png", anon_user_id="u").model_dump()
    assert set(url_body) == set(by_url["properties"])
    assert set(multipart["properties"]) == {"image", "anon_user_id"} == set(upload["properties"])
    declared = BackgroundRemoveInput.model_json_schema()["properties"]["anon_user_id"]["anyOf"][0]
    for key in ("minLength", "maxLength", "pattern"):
        assert declared[key] == multipart["properties"]["anon_user_id"][key]
    content = fixture["resources"]["POST /image/background-remove"]["response"]["content"]
    assert set(content) == {"image/png"}


@pytest.mark.parametrize("is_async", [False, True])
def test_local_images_are_uploaded(is_async, httpx2_mock, multipart_parts, tmp_path):
    path = tmp_path / "photo.png"
    path.write_bytes(PNG)
    respond(httpx2_mock)
    result = invoke(is_async, path, anon_user_id="end-user")
    assert result == GeneratedImage(RESULT, "png", "image/png")
    parts = multipart_parts()
    assert parts["image"] == [("photo.png", "image/png", PNG)]
    assert parts["anon_user_id"] == [(None, "text/plain", b"end-user")]
    assert httpx2_mock.get_request().headers["authorization"] == "Bearer test"


@pytest.mark.parametrize("is_async", [False, True])
def test_bytes_and_borrowed_files_are_uploaded(is_async, httpx2_mock, multipart_parts):
    respond(httpx2_mock)
    respond(httpx2_mock)
    invoke(is_async, PNG)
    handle = io.BytesIO(b"xx" + PNG)
    handle.read(2)
    invoke(is_async, handle)
    assert not handle.closed
    for index in (0, 1):
        assert multipart_parts(index)["image"] == [("image.png", "image/png", PNG)]
        assert "anon_user_id" not in multipart_parts(index)


@pytest.mark.parametrize("is_async", [False, True])
def test_urls_are_sent_for_venice_to_fetch(is_async, httpx2_mock, request_json):
    respond(httpx2_mock)
    invoke(is_async, IMAGE_URL)
    request = httpx2_mock.get_request()
    assert request.headers["content-type"] == "application/json"
    assert request_json() == {"image_url": IMAGE_URL}
    respond(httpx2_mock)
    invoke(is_async, "HTTP://example.com/a.png", anon_user_id="u")
    assert request_json() == {"image_url": "HTTP://example.com/a.png", "anon_user_id": "u"}


def test_a_path_named_like_a_url_is_a_file(httpx2_mock, multipart_parts, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("https:odd.png").write_bytes(PNG)
    respond(httpx2_mock)
    invoke(False, Path("https:odd.png"))
    assert multipart_parts()["image"][0][2] == PNG


@pytest.mark.parametrize("is_async", [False, True])
def test_validation_precedes_reading_and_http(is_async, httpx2_mock, tmp_path):
    with pytest.raises(ValidationError):
        invoke(is_async, tmp_path / "missing.png", anon_user_id="a||b")
    with pytest.raises(ValidationError):
        invoke(is_async, "https://")
    assert not httpx2_mock.get_requests()


@pytest.mark.parametrize("is_async", [False, True])
def test_responses_and_errors(is_async, httpx2_mock):
    respond(httpx2_mock, content=b"")
    with pytest.raises(VeniceResponseError, match="Removing image background failed: .*image"):
        invoke(is_async, PNG)
    respond(httpx2_mock, content=b'{"ok": true}', content_type="application/json")
    with pytest.raises(VeniceResponseError, match="unknown output format"):
        invoke(is_async, PNG)
    httpx2_mock.add_response(url=URL, status_code=400, json={"error": "Upstream returned 404"})
    with pytest.raises(VeniceAPIError, match="Removing image background failed: .*400.*404"):
        invoke(is_async, IMAGE_URL)


def test_result_type(httpx2_mock):
    respond(httpx2_mock)
    with VeniceClient(key="test") as client:
        assert_type(client.remove_image_background(PNG), GeneratedImage)


def test_cli_saves_next_to_a_local_image(cli_runner, httpx2_mock, mock_venice_api_key, tmp_path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(PNG)
    respond(httpx2_mock)
    result = cli_runner.invoke(cli, ["venice", "background-remove", str(path)])
    assert result.exit_code == 0, result.output
    assert result.output == f"Image saved to {tmp_path / 'photo_nobg.png'}\n"
    assert (tmp_path / "photo_nobg.png").read_bytes() == RESULT
    # An existing file is kept unless --overwrite is given.
    respond(httpx2_mock)
    result = cli_runner.invoke(cli, ["venice", "background-remove", str(path)])
    assert result.exit_code == 0, result.output
    assert len(list(tmp_path.glob("photo_nobg*.png"))) == 2
    respond(httpx2_mock, content=RESULT + b"!")
    result = cli_runner.invoke(
        cli, ["venice", "background-remove", str(path), "--overwrite", "--anon-user-id", "u"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "photo_nobg.png").read_bytes() == RESULT + b"!"


def test_cli_saves_a_url_image_in_the_current_directory(
    cli_runner, httpx2_mock, mock_venice_api_key, tmp_path, monkeypatch, request_json
):
    monkeypatch.chdir(tmp_path)
    respond(httpx2_mock)
    result = cli_runner.invoke(cli, ["venice", "background-remove", IMAGE_URL])
    assert result.exit_code == 0, result.output
    assert request_json() == {"image_url": IMAGE_URL}
    assert (tmp_path / "cat one_nobg.png").read_bytes() == RESULT
    respond(httpx2_mock)
    out = tmp_path / "out"
    out.mkdir()
    result = cli_runner.invoke(cli, ["venice", "background-remove", IMAGE_URL, "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "cat one_nobg.png").exists()
    respond(httpx2_mock)
    result = cli_runner.invoke(
        cli, ["venice", "background-remove", "https://example.com/", "-o", "exact.png"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "exact.png").exists()


def test_cli_errors(cli_runner, httpx2_mock, mock_venice_api_key, tmp_path):
    result = cli_runner.invoke(cli, ["venice", "background-remove", str(tmp_path / "no.png")])
    assert result.exit_code == 2
    assert "does not exist or is not a file" in result.output
    path = tmp_path / "photo.png"
    path.write_bytes(PNG)
    httpx2_mock.add_response(url=URL, status_code=415, json={"error": "Invalid content-type"})
    result = cli_runner.invoke(cli, ["venice", "background-remove", str(path)])
    assert result.exit_code == 1
    assert result.output.startswith("Error: Removing image background failed: ")
    assert not (tmp_path / "photo_nobg.png").exists()
