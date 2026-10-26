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

    def test_existing_file_is_never_touched_without_overwrite(self, tmp_path):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"
        target.write_bytes(b"old audio")

        with pytest.raises(OSError):
            write_audio_file(
                _chunks_then_fail([b"new"], OSError("disk full")), target, DownloadProgress(False)
            )

        # The taken name is left alone: a timestamped name was used instead.
        assert target.read_bytes() == b"old audio"
        (part,) = tmp_path.glob("out_*.mp3.part")
        assert part.read_bytes() == b"new"
        assert list(tmp_path.glob("out_*.mp3")) == []

    def test_overwrite_replaces_the_target(self, tmp_path):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"
        target.write_bytes(b"old audio")

        write_audio_file([b"new audio"], target, DownloadProgress(False), overwrite=True)

        assert target.read_bytes() == b"new audio"
        assert list(tmp_path.glob("*.part")) == []

    def test_overwrite_failure_keeps_the_old_target(self, tmp_path):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"
        target.write_bytes(b"old audio")

        with pytest.raises(OSError):
            write_audio_file(
                _chunks_then_fail([b"new"], OSError("disk full")),
                target,
                DownloadProgress(False),
                overwrite=True,
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

    def test_stale_part_file_is_left_alone(self, tmp_path):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"
        stale = tmp_path / "out.mp3.part"
        stale.write_bytes(b"stale leftover")

        write_audio_file([b"fresh"], target, DownloadProgress(False))

        # A taken staging name is skipped rather than reused or deleted: it may
        # belong to another writer.
        assert target.read_bytes() == b"fresh"
        assert stale.read_bytes() == b"stale leftover"
        assert [p for p in tmp_path.glob("*.part") if p != stale] == []
        assert len(list(tmp_path.glob(".*.part"))) == 0

    def test_planted_symlink_is_not_followed(self, tmp_path):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        victim = tmp_path / "victim.txt"
        victim.write_text("IMPORTANT")
        (tmp_path / "out.mp3.part").symlink_to(victim)

        write_audio_file([b"audio"], tmp_path / "out.mp3", DownloadProgress(False))

        assert victim.read_text() == "IMPORTANT"
        assert (tmp_path / "out.mp3").read_bytes() == b"audio"

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_overlapping_downloads(self, tmp_path, overwrite):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"
        inner: list[pathlib.Path] = []

        def first_writer():
            yield b"AAAA"
            # A concurrent run starts while this download is in flight.
            saved = write_audio_file(
                [b"BBBB"], target, DownloadProgress(False), overwrite=overwrite
            )
            inner.append(saved)
            assert saved.read_bytes() == b"BBBB"
            yield b"AAAA"

        saved = write_audio_file(
            first_writer(), target, DownloadProgress(False), overwrite=overwrite
        )

        assert saved == target
        assert target.read_bytes() == b"AAAA" + b"AAAA", "published file must hold one download"
        others = [p for p in tmp_path.iterdir() if p != target]
        if overwrite:
            assert inner == [target]
            assert others == []
        else:
            assert others == inner
            assert others[0].read_bytes() == b"BBBB"
        assert not list(tmp_path.glob("*.part"))

    @pytest.mark.parametrize("chunks", [[], [b"partial"]])
    def test_failure_with_stale_part_preserves_only_owned_partial(self, tmp_path, capsys, chunks):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"
        stale = tmp_path / "out.mp3.part"
        stale.write_bytes(b"stale")
        error = OSError("connection lost")

        with pytest.raises(OSError) as excinfo:
            write_audio_file(_chunks_then_fail(chunks, error), target, DownloadProgress(False))

        assert excinfo.value is error
        assert not target.exists()
        assert stale.read_bytes() == b"stale"
        stderr = capsys.readouterr().err
        if chunks:
            (partial,) = tmp_path.glob(".out.mp3.*.part")
            assert partial.read_bytes() == b"partial"
            assert str(partial) in stderr
        else:
            assert list(tmp_path.iterdir()) == [stale]
            assert stderr == ""

    def test_staging_open_failure_releases_reservation(self, tmp_path, monkeypatch):
        from llm_venice.models import audio

        def fail_open(output_path):
            raise OSError("cannot create staging file")

        monkeypatch.setattr(audio, "_open_staging_file", fail_open)
        with pytest.raises(OSError, match="cannot create staging file"):
            audio.write_audio_file([b"audio"], tmp_path / "out.mp3", audio.DownloadProgress(False))
        assert list(tmp_path.iterdir()) == []

    def test_dangling_destination_symlink_is_left_alone(self, tmp_path):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"
        missing = tmp_path / "missing.mp3"
        target.symlink_to(missing)

        saved = write_audio_file([b"audio"], target, DownloadProgress(False))

        assert saved != target
        assert saved.read_bytes() == b"audio"
        assert target.is_symlink()
        assert not missing.exists()


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
