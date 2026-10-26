"""End-to-end CLI tests for image generation and speech file output."""

import base64
import json
import pathlib

import pytest
from llm.cli import cli

from llm_venice.constants import ENDPOINT_AUDIO_SPEECH, ENDPOINT_IMAGE_GENERATE

PNG = b"\x89PNG\r\n\x1a\n"
AUDIO = b"ID3audio"
IMAGE_JSON = {
    "images": [base64.b64encode(PNG).decode()],
    "request": {"model": "img-model"},
    "timing": {},
}


@pytest.fixture
def catalog(hermetic_venice_catalog):
    """A catalog with one image model, one tts model and one text model."""
    return hermetic_venice_catalog(
        [
            {
                "id": "img-model",
                "type": "image",
                "model_spec": {"capabilities": {}},
            },
            {"id": "tts-model", "type": "tts"},
            {"id": "text-model", "type": "text", "model_spec": {"capabilities": {}}},
        ]
    )


def test_cli_image_generation_writes_file(
    cli_runner, httpx2_mock, mock_venice_api_key, catalog, tmp_path
):
    """llm -m venice/<image-model> writes an image and reports the path."""
    httpx2_mock.add_response(method="POST", url=ENDPOINT_IMAGE_GENERATE, json=IMAGE_JSON)

    result = cli_runner.invoke(
        cli,
        ["-m", "venice/img-model", "-o", "output_dir", str(tmp_path), "A cat", "--no-log"],
    )

    assert result.exit_code == 0, result.output
    (saved,) = list(tmp_path.glob("*.png"))
    assert saved.read_bytes() == PNG
    assert str(saved) in result.output

    payload = json.loads(httpx2_mock.get_requests()[0].read())
    assert payload["model"] == "img-model"
    assert payload["prompt"] == "A cat"


def test_cli_image_generation_is_logged(
    cli_runner, httpx2_mock, mock_venice_api_key, catalog, tmp_path
):
    """The image path is recorded in logs.db, so it can be found again."""
    httpx2_mock.add_response(method="POST", url=ENDPOINT_IMAGE_GENERATE, json=IMAGE_JSON)

    result = cli_runner.invoke(
        cli,
        [
            "-m",
            "venice/img-model",
            "-o",
            "output_dir",
            str(tmp_path),
            "A cat",
            "--log",
        ],
    )

    assert result.exit_code == 0, result.output
    logs = cli_runner.invoke(cli, ["logs", "--json"]).output
    entries = json.loads(logs)
    assert len(entries) == 1
    # The output path is not part of response_json yet; adapters add it in a
    # later layer, so assert on the echoed message and the recorded model.
    (saved,) = list(tmp_path.glob("*.png"))
    assert str(saved) in result.output
    assert entries[0]["response_json"]["request"]["model"] == "img-model"


def test_cli_speech_saves_file_and_logs(
    cli_runner, httpx2_mock, mock_venice_api_key, catalog, tmp_path
):
    """llm -m venice/<tts-model> still writes a file and logs its path."""
    httpx2_mock.add_response(
        method="POST",
        url=ENDPOINT_AUDIO_SPEECH,
        content=AUDIO,
        headers={"Content-Type": "audio/mpeg"},
    )

    result = cli_runner.invoke(
        cli,
        [
            "-m",
            "venice/tts-model",
            "-o",
            "output_dir",
            str(tmp_path),
            "-o",
            "output_filename",
            "hello.mp3",
            "Hello.",
            "--log",
            "--no-stream",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "hello.mp3").read_bytes() == AUDIO
    assert str(tmp_path / "hello.mp3") in result.output

    payload = json.loads(httpx2_mock.get_requests()[0].read())
    assert payload["model"] == "tts-model"
    assert payload["input"] == "Hello."
    assert payload["streaming"] is False

    entries = json.loads(cli_runner.invoke(cli, ["logs", "--json"]).output)
    assert len(entries) == 1
    assert entries[0]["response_json"]["output_path"] == str(tmp_path / "hello.mp3")


def test_cli_speech_streaming_writes_file(
    cli_runner, httpx2_mock, mock_venice_api_key, catalog, tmp_path
):
    """The default streaming path writes the same file as --no-stream."""
    httpx2_mock.add_response(
        method="POST",
        url=ENDPOINT_AUDIO_SPEECH,
        content=AUDIO,
        headers={"Content-Type": "audio/mpeg"},
    )

    result = cli_runner.invoke(
        cli,
        [
            "-m",
            "venice/tts-model",
            "-o",
            "output_dir",
            str(tmp_path),
            "-o",
            "output_filename",
            "hello.mp3",
            "Hello.",
            "--no-log",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "hello.mp3").read_bytes() == AUDIO
    assert not (tmp_path / "hello.mp3.part").exists()
    assert json.loads(httpx2_mock.get_requests()[0].read())["streaming"] is True


def test_cli_speech_api_error_is_clean(
    cli_runner, httpx2_mock, mock_venice_api_key, catalog, tmp_path
):
    """A model-path API error is reported without a traceback and writes no file."""
    httpx2_mock.add_response(
        method="POST", url=ENDPOINT_AUDIO_SPEECH, status_code=400, json={"error": "Invalid voice"}
    )

    result = cli_runner.invoke(
        cli,
        [
            "-m",
            "venice/tts-model",
            "-o",
            "output_dir",
            str(tmp_path),
            "Hello.",
            "--no-log",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid voice" in result.output
    assert "Traceback" not in result.output
    assert list(tmp_path.iterdir()) == []


def test_tests_do_not_patch_httpx2_modules():
    """Transport mocking goes through httpx2_mock, not patch() on httpx2 itself."""
    needle = 'patch("' + "httpx2"  # built at runtime so this file does not match itself
    offenders = [
        path.name
        for path in pathlib.Path(__file__).parent.glob("test_*.py")
        if needle in path.read_text()
    ]
    assert offenders == []
