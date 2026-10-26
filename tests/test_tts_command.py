"""Tests for `llm venice tts`."""

import io
import json
import pathlib
from unittest.mock import patch

import pytest
from llm.cli import cli

SPEECH_URL = "https://api.venice.ai/api/v1/audio/speech"
MODELS_URL = "https://api.venice.ai/api/v1/models?type=all"
AUDIO = b"ID3abc"

CATALOG = [
    {"id": "text-model", "type": "text", "model_spec": {"capabilities": {}}},
    {"id": "tts-first", "type": "tts"},
    {"id": "tts-kokoro", "type": "tts"},
]


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    """Isolated LLM user dir with a model catalog containing tts models."""
    (tmp_path / "venice_models.json").write_text(json.dumps(CATALOG))
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture
def speech_response(httpx2_mock):
    httpx2_mock.add_response(
        method="POST",
        url=SPEECH_URL,
        content=AUDIO,
        headers={"Content-Type": "audio/mpeg"},
    )
    return httpx2_mock


def tts(cli_runner, *args, input=None):
    return cli_runner.invoke(cli, ["venice", "tts", *args], input=input)


def request_payload(httpx2_mock):
    (request,) = httpx2_mock.get_requests()
    return json.loads(request.read())


# stdout mode


def test_stdout_writes_exact_bytes(user_dir, speech_response, mock_venice_api_key, cli_runner):
    result = tts(cli_runner, "Hello", "-o", "-")

    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == AUDIO
    assert result.stderr == ""
    assert not (user_dir / "audio").exists()

    payload = request_payload(speech_response)
    assert payload["input"] == "Hello"
    assert payload["streaming"] is True
    assert "progress" not in payload


def test_stdout_progress_goes_to_stderr(user_dir, speech_response, mock_venice_api_key, cli_runner):
    result = tts(cli_runner, "Hello", "-o", "-", "--progress")

    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == AUDIO
    assert "Downloaded" in result.stderr


def test_stdout_to_tty_is_refused(user_dir, httpx2_mock, mock_venice_api_key, cli_runner):
    with patch("llm_venice.cli.tts._stdout_is_tty", return_value=True):
        result = tts(cli_runner, "Hello", "-o", "-")

    assert result.exit_code == 2
    assert "Refusing to write binary audio to a terminal" in result.stderr
    assert httpx2_mock.get_requests() == []


def test_stdout_broken_pipe_exits_cleanly(
    user_dir, speech_response, mock_venice_api_key, cli_runner
):
    class BrokenPipeBuffer(io.BytesIO):
        def write(self, b):
            raise BrokenPipeError

    with patch("llm_venice.cli.tts._binary_stdout", return_value=BrokenPipeBuffer()):
        result = tts(cli_runner, "Hello", "-o", "-")

    assert result.exit_code == 1
    assert result.stderr == ""
    assert result.exception is None or isinstance(result.exception, SystemExit)


# text input


def test_text_from_stdin(user_dir, speech_response, mock_venice_api_key, cli_runner):
    result = tts(cli_runner, "-o", "-", input="Hello from stdin\n")

    assert result.exit_code == 0, result.output
    assert request_payload(speech_response)["input"] == "Hello from stdin"


def test_text_dash_reads_stdin(user_dir, speech_response, mock_venice_api_key, cli_runner):
    result = tts(cli_runner, "-", "-o", "-", input="Dash input")

    assert result.exit_code == 0, result.output
    assert request_payload(speech_response)["input"] == "Dash input"


@pytest.mark.parametrize("args, stdin", [(["   "], None), ([], ""), (["-"], "  \n")])
def test_empty_text_errors(user_dir, httpx2_mock, mock_venice_api_key, cli_runner, args, stdin):
    result = tts(cli_runner, *args, "-o", "-", input=stdin)

    assert result.exit_code == 2
    assert "No text to speak" in result.stderr
    assert httpx2_mock.get_requests() == []


def test_missing_text_with_tty_stdin_errors(user_dir, httpx2_mock, mock_venice_api_key, cli_runner):
    with patch("llm_venice.cli.tts._stdin_is_tty", return_value=True):
        result = tts(cli_runner, "-o", "-")

    assert result.exit_code == 2
    assert "Provide TEXT, or pipe text on stdin" in result.stderr


# file mode


def test_output_file(user_dir, speech_response, mock_venice_api_key, cli_runner, tmp_path):
    target = tmp_path / "out" / "hello.mp3"

    result = tts(cli_runner, "Hello", "-o", str(target))

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{target}\n"
    assert target.read_bytes() == AUDIO


def test_output_directory_gets_generated_name(
    user_dir, speech_response, mock_venice_api_key, cli_runner, tmp_path
):
    out_dir = tmp_path / "clips"
    out_dir.mkdir()

    result = tts(cli_runner, "Hello", "-m", "tts-kokoro", "--format", "wav", "-o", str(out_dir))

    assert result.exit_code == 0, result.output
    saved = pathlib.Path(result.stdout.strip())
    assert saved.parent == out_dir
    assert saved.name.endswith("_venice_tts-kokoro.wav")
    assert saved.read_bytes() == AUDIO


def test_default_output_under_llm_user_dir(
    user_dir, speech_response, mock_venice_api_key, cli_runner
):
    result = tts(cli_runner, "Hello")

    assert result.exit_code == 0, result.output
    saved = pathlib.Path(result.stdout.strip())
    assert saved.parent == user_dir / "audio"
    assert saved.read_bytes() == AUDIO


@pytest.mark.parametrize("overwrite", [False, True])
def test_overwrite_versus_unique_name(
    user_dir, speech_response, mock_venice_api_key, cli_runner, tmp_path, overwrite
):
    target = tmp_path / "hello.mp3"
    target.write_bytes(b"old")

    args = ["Hello", "-o", str(target)] + (["--overwrite"] if overwrite else [])
    result = tts(cli_runner, *args)

    assert result.exit_code == 0, result.output
    saved = pathlib.Path(result.stdout.strip())
    if overwrite:
        assert saved == target
        assert target.read_bytes() == AUDIO
    else:
        assert saved != target
        assert saved.name.startswith("hello_")
        assert target.read_bytes() == b"old"
        assert saved.read_bytes() == AUDIO


# option validation


@pytest.mark.parametrize(
    "args, flag",
    [
        (["--speed", "5"], "--speed"),
        (["--speed", "0.1"], "--speed"),
        (["--format", "ogg"], "--format"),
    ],
)
def test_invalid_options_rejected_before_request(
    user_dir, httpx2_mock, mock_venice_api_key, cli_runner, args, flag
):
    result = tts(cli_runner, "Hello", "-o", "-", *args)

    assert result.exit_code == 2
    assert f"Invalid value for {flag}" in result.stderr
    assert httpx2_mock.get_requests() == []


def test_options_reach_payload(user_dir, speech_response, mock_venice_api_key, cli_runner):
    result = tts(
        cli_runner, "Hello", "-o", "-", "--voice", "af_sky", "--format", "flac", "--speed", "1.5"
    )

    assert result.exit_code == 0, result.output
    payload = request_payload(speech_response)
    assert payload["voice"] == "af_sky"
    assert payload["response_format"] == "flac"
    assert payload["speed"] == 1.5


# model resolution


@pytest.mark.parametrize("model", ["tts-kokoro", "venice/tts-kokoro"])
def test_model_with_or_without_prefix(
    user_dir, speech_response, mock_venice_api_key, cli_runner, model
):
    result = tts(cli_runner, "Hello", "-m", model, "-o", "-")

    assert result.exit_code == 0, result.output
    assert request_payload(speech_response)["model"] == "tts-kokoro"


def test_default_model_is_first_catalog_tts_model(
    user_dir, speech_response, mock_venice_api_key, cli_runner
):
    result = tts(cli_runner, "Hello", "-o", "-")

    assert result.exit_code == 0, result.output
    assert request_payload(speech_response)["model"] == "tts-first"


def test_non_tts_model_errors(user_dir, httpx2_mock, mock_venice_api_key, cli_runner):
    result = tts(cli_runner, "Hello", "-m", "venice/text-model", "-o", "-")

    assert result.exit_code == 2
    assert "'text-model' is a text model, not a text-to-speech model" in result.stderr
    assert httpx2_mock.get_requests() == []


def test_default_model_fetches_catalog_when_missing(
    tmp_path, monkeypatch, httpx2_mock, mock_venice_api_key, cli_runner
):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    httpx2_mock.add_response(method="GET", url=MODELS_URL, json={"data": CATALOG})
    httpx2_mock.add_response(method="POST", url=SPEECH_URL, content=AUDIO)

    result = tts(cli_runner, "Hello", "-o", "-")

    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == AUDIO
    assert (tmp_path / "venice_models.json").exists()
    speech_request = httpx2_mock.get_requests()[-1]
    assert json.loads(speech_request.read())["model"] == "tts-first"


def test_default_model_without_catalog_errors(
    tmp_path, monkeypatch, httpx2_mock, mock_venice_api_key, cli_runner
):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    httpx2_mock.add_response(method="GET", url=MODELS_URL, status_code=500, is_reusable=True)

    result = tts(cli_runner, "Hello", "-o", "-")

    assert result.exit_code == 1
    assert "No Venice model catalog available" in result.stderr
    assert "Traceback" not in result.output
    assert all(r.method == "GET" for r in httpx2_mock.get_requests())


# errors


def test_http_error_is_clean(user_dir, httpx2_mock, mock_venice_api_key, cli_runner, tmp_path):
    httpx2_mock.add_response(
        method="POST",
        url=SPEECH_URL,
        status_code=400,
        json={"error": "Invalid voice"},
    )
    target = tmp_path / "hello.mp3"

    result = tts(cli_runner, "Hello", "-o", str(target))

    assert result.exit_code == 1
    assert result.stderr.startswith("Error: Generating speech failed")
    assert "Invalid voice" in result.stderr
    assert "Traceback" not in result.output
    assert not target.exists()


def test_missing_key_is_clean(user_dir, httpx2_mock, cli_runner):
    with patch("llm.get_key", return_value=None):
        result = tts(cli_runner, "Hello", "-o", "-")

    assert result.exit_code == 1
    assert "No key found for Venice" in result.stderr
    assert httpx2_mock.get_requests() == []


# removed -o stdout option


def test_model_path_rejects_removed_stdout_option(
    user_dir, httpx2_mock, mock_venice_api_key, cli_runner
):
    result = cli_runner.invoke(cli, ["-m", "venice/tts-kokoro", "Hi", "-o", "stdout", "true"])

    assert result.exit_code != 0
    assert "stdout" in result.output
    assert "Extra inputs are not permitted" in result.output
    assert httpx2_mock.get_requests() == []
