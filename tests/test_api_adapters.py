"""The new clients preserve adapter policy, including atomic saves and metadata."""

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import MagicMock
from typing import Any

import httpx2
import llm
import pytest
from llm.cli import cli

from llm_venice import AsyncVeniceImage, VeniceImage
from llm_venice import _files

PNG = b"\x89PNG\r\n\x1a\nbytes"


def execute_image(is_async, tmp_path, **options):
    model: Any = (AsyncVeniceImage if is_async else VeniceImage)("model")
    prompt = MagicMock(prompt="hello", options=model.Options(output_dir=tmp_path, **options))
    response = MagicMock(response_json=None)

    async def run():
        return [chunk async for chunk in model.execute(prompt, False, response, key="provided-key")]

    output = (
        asyncio.run(run())
        if is_async
        else list(model.execute(prompt, False, response, key="provided-key"))
    )
    return output, response.response_json


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("whole_violation", [False, True])
def test_image_partial_and_moderated_output(is_async, whole_violation, tmp_path, httpx2_mock):
    httpx2_mock.add_response(
        json={
            "images": [base64.b64encode(PNG).decode()],
            "content_policy_violation_count": 1,
            "request": {"model": "model"},
        },
        headers={
            "x-venice-is-content-violation": str(whole_violation).lower(),
            "x-venice-is-blurred": "true",
        },
    )
    output, metadata = execute_image(
        is_async, tmp_path, extra_body={"future": {"nested": None}}, format="webp"
    )
    assert metadata["content_policy_violation_count"] == 1
    assert metadata["is_blurred"] is True
    assert metadata["content_violation"] == whole_violation
    assert "omitted" in output[0] and "blurred" in output[0]
    if whole_violation:
        assert metadata["output_paths"] == []
        assert list(tmp_path.iterdir()) == []
    else:
        assert Path(metadata["output_paths"][0]).read_bytes() == PNG
        assert metadata["output_paths"][0].endswith(".png")
    request = httpx2_mock.get_request()
    assert request.headers["authorization"] == "Bearer provided-key"
    payload = json.loads(request.content)
    assert payload["future"] == {"nested": None}
    assert payload["format"] == "webp" and payload["return_binary"] is False
    assert "output_dir" not in payload and "extra_body" not in payload


@pytest.mark.parametrize("is_async", [False, True])
def test_image_late_collision_reuses_generated_bytes(is_async, tmp_path, httpx2_mock, monkeypatch):
    httpx2_mock.add_response(json={"images": [base64.b64encode(PNG).decode()]})
    original = _files.try_rename_no_replace
    calls = []

    def race(source, destination):
        calls.append(destination)
        if len(calls) == 1:
            destination.write_bytes(b"other writer")
        return original(source, destination)

    monkeypatch.setattr(_files, "try_rename_no_replace", race)
    _, metadata = execute_image(is_async, tmp_path, output_filename="image.png")
    assert (tmp_path / "image.png").read_bytes() == b"other writer"
    assert Path(metadata["output_paths"][0]).read_bytes() == PNG
    assert len(httpx2_mock.get_requests()) == 1


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("failure", ["service", "decode", "file"])
def test_image_errors_map_to_model_error(is_async, failure, tmp_path, httpx2_mock):
    if failure == "service":
        httpx2_mock.add_exception(httpx2.ReadError("connection lost"))
    elif failure == "decode":
        httpx2_mock.add_response(json={"images": ["invalid base64"]})
    else:
        httpx2_mock.add_response(json={"images": [base64.b64encode(PNG).decode()]})
        (tmp_path / "directory").mkdir()
    with pytest.raises(llm.ModelError):
        execute_image(is_async, tmp_path, output_filename="directory", overwrite_files=True)


def test_return_binary_cli_rejected(
    hermetic_venice_model, cli_runner, tmp_path, monkeypatch, httpx2_mock
):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "venice_models.json").write_text(json.dumps([{"id": "image", "type": "image"}]))
    result = cli_runner.invoke(cli, ["-m", "venice/image", "hello", "-o", "return_binary", "true"])
    assert result.exit_code != 0
    assert "return_binary" in result.output and "Extra inputs" in result.output
    assert not httpx2_mock.get_requests()
