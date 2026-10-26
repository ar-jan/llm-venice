from contextlib import contextmanager
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

    with patch("httpx2.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b"fake-audio-bytes"
        mock_response.headers = {"Content-Type": "audio/mpeg"}
        mock_post.return_value = mock_response

        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            results = list(model.execute(prompt, False, MagicMock(), None))

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

    saved = pathlib.Path(results[-1].removeprefix("Audio saved to "))
    assert saved.read_bytes() == b"fake-audio-bytes"
    assert not saved.with_name(saved.name + ".part").exists()


def test_venice_speech_streaming_writes_file(mock_venice_api_key, monkeypatch, tmp_path):
    """stream=True should download audio via httpx2.stream and write a file."""
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

    with patch("httpx2.stream", side_effect=stream_side_effect) as mock_stream:
        with patch("httpx2.post") as mock_post:
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


def test_venice_speech_speed_validation():
    """Options should validate speed range."""
    with pytest.raises(ValueError):
        VeniceSpeech.Options(speed=0.24)


def test_venice_speech_streaming_http_error_has_detail(httpx2_mock, mock_venice_api_key, tmp_path):
    """A streamed error response is read, so the API's error detail reaches the user."""
    import llm

    httpx2_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        status_code=400,
        json={"error": "Invalid voice"},
    )
    model = VeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options = VeniceSpeech.Options(output_dir=tmp_path)

    with pytest.raises(llm.ModelError, match="Invalid voice"):
        list(model.execute(prompt, True, MagicMock(), None))


def _chunks_then_fail(chunks, exc):
    yield from chunks
    raise exc


class TestWriteAudioFile:
    """write_audio_file writes via <name>.part and renames on success."""

    def test_success_renames_part_file(self, tmp_path):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "sub" / "out.mp3"
        write_audio_file([b"a", b"", b"b"], target, DownloadProgress(False))

        assert target.read_bytes() == b"ab"
        assert not (tmp_path / "sub" / "out.mp3.part").exists()

    def test_failure_keeps_partial_and_warns(self, tmp_path, capsys):
        import httpx2

        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"
        error = httpx2.ReadError("connection lost")

        with pytest.raises(httpx2.ReadError) as excinfo:
            write_audio_file(
                _chunks_then_fail([b"a" * 2048], error), target, DownloadProgress(False)
            )

        assert excinfo.value is error
        assert not target.exists()
        assert (tmp_path / "out.mp3.part").read_bytes() == b"a" * 2048
        stderr = capsys.readouterr().err
        assert "Warning: download interrupted; partial audio kept at" in stderr
        assert "out.mp3.part (2KB)" in stderr

    def test_failure_before_any_bytes_removes_part(self, tmp_path, capsys):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"

        with pytest.raises(OSError):
            write_audio_file(
                _chunks_then_fail([], OSError("boom")), target, DownloadProgress(False)
            )

        assert not target.exists()
        assert not (tmp_path / "out.mp3.part").exists()
        assert capsys.readouterr().err == ""

    def test_failure_keeps_existing_file(self, tmp_path):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"
        target.write_bytes(b"old audio")

        with pytest.raises(OSError):
            write_audio_file(
                _chunks_then_fail([b"new"], OSError("disk full")), target, DownloadProgress(False)
            )

        assert target.read_bytes() == b"old audio"
        assert (tmp_path / "out.mp3.part").read_bytes() == b"new"

    def test_keyboard_interrupt_is_not_converted(self, tmp_path):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"

        with pytest.raises(KeyboardInterrupt):
            write_audio_file(
                _chunks_then_fail([b"x"], KeyboardInterrupt()), target, DownloadProgress(False)
            )

        assert (tmp_path / "out.mp3.part").read_bytes() == b"x"

    def test_stale_part_file_is_replaced(self, tmp_path):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"
        (tmp_path / "out.mp3.part").write_bytes(b"stale leftover")

        write_audio_file([b"fresh"], target, DownloadProgress(False))

        assert target.read_bytes() == b"fresh"
        assert not (tmp_path / "out.mp3.part").exists()


def test_venice_speech_streaming_failure_keeps_partial(
    mock_venice_api_key, monkeypatch, tmp_path, capsys
):
    """The model path keeps a partial download as .part, not under the final name."""
    import httpx2
    import llm

    model = VeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Stream me"
    prompt.options = VeniceSpeech.Options(output_dir=tmp_path, output_filename="out.mp3")

    @contextmanager
    def fake_stream(*_args, **_kwargs):
        response = Mock()
        response.raise_for_status.return_value = None
        response.headers = {"Content-Type": "audio/mpeg"}
        response.iter_bytes.return_value = _chunks_then_fail(
            [b"chunk-1"], httpx2.ReadError("connection lost")
        )
        yield response

    with patch("httpx2.stream", side_effect=lambda *a, **k: fake_stream(*a, **k)):
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            with pytest.raises(llm.ModelError, match="connection lost"):
                list(model.execute(prompt, True, MagicMock(), None))

    assert not (tmp_path / "out.mp3").exists()
    assert (tmp_path / "out.mp3.part").read_bytes() == b"chunk-1"
    assert "partial audio kept at" in capsys.readouterr().err
