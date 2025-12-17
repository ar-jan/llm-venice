from contextlib import contextmanager
import io
import pathlib
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from llm_venice import VeniceSpeech


def test_venice_speech_payload_includes_options(mock_venice_api_key, monkeypatch, tmp_path):
    """Ensure /audio/speech requests include expected parameters."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

    model = VeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Hello from Venice"

    options = Mock()
    options.model_dump.return_value = {
        "voice": "af_sky",
        "response_format": "mp3",
        "speed": 1.0,
        "output_dir": None,
        "output_filename": None,
        "overwrite_files": False,
    }
    prompt.options = options

    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b"fake-audio-bytes"
        mock_response.headers = {"Content-Type": "audio/mpeg"}
        mock_post.return_value = mock_response

        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            with patch("pathlib.Path.write_bytes") as mock_write:
                list(model.execute(prompt, False, MagicMock(), None))

                assert options.model_dump.call_args_list == [
                    call(by_alias=True),
                    call(by_alias=True),
                ]
                mock_post.assert_called_once()

                payload = mock_post.call_args[1]["json"]
                assert payload["model"] == "tts-kokoro"
                assert payload["input"] == "Hello from Venice"
                assert payload["voice"] == "af_sky"
                assert payload["response_format"] == "mp3"
                assert payload["speed"] == 1.0
                assert payload["streaming"] is False

                mock_write.assert_called_once()


def test_venice_speech_streaming_writes_file(mock_venice_api_key, monkeypatch, tmp_path):
    """stream=True should download audio via httpx.stream and write a file."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

    model = VeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Stream me"
    prompt.options = VeniceSpeech.Options(response_format="mp3")

    chunks = [b"chunk-1", b"chunk-2"]

    @contextmanager
    def fake_stream(*_args, **_kwargs):
        response = Mock()
        response.raise_for_status.return_value = None
        response.headers = {"Content-Type": "audio/mpeg"}
        response.iter_bytes.return_value = chunks
        yield response

    def stream_side_effect(*args, **kwargs):
        return fake_stream(*args, **kwargs)

    with patch("httpx.stream", side_effect=stream_side_effect) as mock_stream:
        with patch("httpx.post") as mock_post:
            with patch.object(model, "get_key", return_value=mock_venice_api_key):
                response = MagicMock()
                results = list(model.execute(prompt, True, response, None))

    mock_stream.assert_called_once()
    mock_post.assert_not_called()

    saved_lines = [line for line in results if line.startswith("Audio saved to ")]
    assert len(saved_lines) == 1
    output_path = saved_lines[0].replace("Audio saved to ", "", 1)

    resolved_path = pathlib.Path(output_path)
    assert tmp_path in resolved_path.parents
    assert resolved_path.read_bytes() == b"".join(chunks)


def test_venice_speech_stdout_writes_bytes(mock_venice_api_key, monkeypatch, tmp_path):
    """stdout=True should write binary audio to stdout and not save a file by default."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

    model = VeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options = VeniceSpeech.Options(stdout=True, response_format="mp3")

    chunks = [b"chunk-1", b"chunk-2"]

    @contextmanager
    def fake_stream(*_args, **_kwargs):
        response = Mock()
        response.raise_for_status.return_value = None
        response.headers = {"Content-Type": "audio/mpeg"}
        response.iter_bytes.return_value = chunks
        yield response

    class DummyStdout:
        def __init__(self):
            self.buffer = io.BytesIO()

    dummy_stdout = DummyStdout()

    with patch("httpx.stream", side_effect=lambda *a, **k: fake_stream(*a, **k)) as mock_stream:
        with patch("httpx.post") as mock_post:
            with patch("sys.stdout", dummy_stdout):
                with patch.object(model, "get_key", return_value=mock_venice_api_key):
                    response = MagicMock()
                    results = list(model.execute(prompt, False, response, None))

    mock_stream.assert_called_once()
    mock_post.assert_not_called()
    assert results == []

    payload = mock_stream.call_args[1]["json"]
    assert "stdout" not in payload
    assert "progress" not in payload

    assert dummy_stdout.buffer.getvalue() == b"".join(chunks)
    assert not (tmp_path / "audio").exists()


def test_venice_speech_stdout_broken_pipe_still_saves_file(
    mock_venice_api_key, monkeypatch, tmp_path
):
    """stdout=True should keep writing the file even if stdout breaks."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

    model = VeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options = VeniceSpeech.Options(
        stdout=True,
        output_dir=tmp_path,
        output_filename="audio.mp3",
        response_format="mp3",
    )

    chunks = [b"chunk-1", b"chunk-2", b"chunk-3"]

    @contextmanager
    def fake_stream(*_args, **_kwargs):
        response = Mock()
        response.raise_for_status.return_value = None
        response.headers = {"Content-Type": "audio/mpeg"}
        response.iter_bytes.return_value = chunks
        yield response

    class BrokenPipeBuffer(io.BytesIO):
        def __init__(self):
            super().__init__()
            self.write_calls = 0

        def write(self, b):
            self.write_calls += 1
            if self.write_calls > 1:
                raise BrokenPipeError
            return super().write(b)

    class DummyStdout:
        def __init__(self):
            self.buffer = BrokenPipeBuffer()

    dummy_stdout = DummyStdout()

    with patch("httpx.stream", side_effect=lambda *a, **k: fake_stream(*a, **k)) as mock_stream:
        with patch("httpx.post") as mock_post:
            with patch("sys.stdout", dummy_stdout):
                with patch.object(model, "get_key", return_value=mock_venice_api_key):
                    response = MagicMock()
                    results = list(model.execute(prompt, False, response, None))

    mock_stream.assert_called_once()
    mock_post.assert_not_called()
    assert results == []

    assert (tmp_path / "audio.mp3").read_bytes() == b"".join(chunks)
    assert dummy_stdout.buffer.getvalue() == chunks[0]
    assert dummy_stdout.buffer.write_calls == 2


def test_venice_speech_speed_validation():
    """Options should validate speed range."""
    with pytest.raises(ValueError):
        VeniceSpeech.Options(speed=0.24)
