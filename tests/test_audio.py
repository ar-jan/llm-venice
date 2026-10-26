import asyncio
import json
import pathlib
from unittest.mock import MagicMock, Mock, patch, call

import httpx2
import llm
import pytest

from llm_venice import AsyncVeniceSpeech, VeniceSpeech
from llm_venice.constants import ENDPOINT_AUDIO_SPEECH


def add_speech_response(httpx2_mock, *, content=b"", status_code=200, is_reusable=False):
    """Register a mocked POST to the speech endpoint."""
    httpx2_mock.add_response(
        method="POST",
        url=ENDPOINT_AUDIO_SPEECH,
        status_code=status_code,
        content=content,
        headers={"Content-Type": "audio/mpeg"},
        is_reusable=is_reusable,
    )


def test_venice_speech_payload_includes_options(
    mock_venice_api_key, monkeypatch, tmp_path, httpx2_mock, request_json
):
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

    add_speech_response(httpx2_mock, content=b"fake-audio-bytes")
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        results = list(model.execute(prompt, False, MagicMock(), None))

    assert options.model_dump.call_args_list == [
        call(by_alias=True),
        call(by_alias=True),
    ]
    assert len(httpx2_mock.get_requests()) == 1

    payload = request_json()
    assert payload["model"] == "tts-kokoro"
    assert payload["input"] == "Hello from Venice"
    assert payload["voice"] == "af_sky"
    assert payload["response_format"] == "mp3"
    assert payload["speed"] == 1.0
    assert payload["streaming"] is False

    saved = pathlib.Path(results[-1].removeprefix("Audio saved to "))
    assert saved.read_bytes() == b"fake-audio-bytes"
    assert not saved.with_name(saved.name + ".part").exists()


def test_venice_speech_streaming_writes_file(
    mock_venice_api_key, monkeypatch, tmp_path, httpx2_mock
):
    """stream=True should download audio via httpx2.stream and write a file."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

    model = VeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Stream me"
    prompt.options = VeniceSpeech.Options(response_format="mp3")

    chunks = [b"chunk-1", b"chunk-2"]

    add_speech_response(httpx2_mock, content=b"".join(chunks))
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        response = MagicMock()
        results = list(model.execute(prompt, True, response, None))

    assert len(httpx2_mock.get_requests()) == 1
    assert json.loads(httpx2_mock.get_requests()[0].read())["streaming"] is True

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

    # The request was rejected before any audio arrived, so nothing was written
    assert list(tmp_path.iterdir()) == []


def _chunks_then_fail(chunks, exc):
    yield from chunks
    raise exc


class TestWriteAudioFile:
    """write_audio_file stages audio and publishes complete files atomically."""

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

        # Publication was never reached, so only our staging file remains.
        assert target.read_bytes() == b"old audio"
        assert (tmp_path / "out.mp3.part").read_bytes() == b"new"
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
    @pytest.mark.parametrize("inner_overwrite", [False, True])
    def test_overlapping_downloads(self, tmp_path, overwrite, inner_overwrite):
        from llm_venice.models.audio import DownloadProgress, write_audio_file

        target = tmp_path / "out.mp3"
        inner: list[pathlib.Path] = []

        def first_writer():
            yield b"AAAA"
            assert not target.exists(), "no destination should be visible before publication"
            # A concurrent run starts while this download is in flight.
            saved = write_audio_file(
                [b"BBBB"], target, DownloadProgress(False), overwrite=inner_overwrite
            )
            inner.append(saved)
            assert saved.read_bytes() == b"BBBB"
            yield b"AAAA"

        saved = write_audio_file(
            first_writer(), target, DownloadProgress(False), overwrite=overwrite
        )

        assert saved.read_bytes() == b"AAAA" + b"AAAA", "published file must hold one download"
        others = [p for p in tmp_path.iterdir() if p != target]
        assert inner == [target]
        if overwrite:
            assert saved == target
            assert others == []
        else:
            assert saved != target
            assert others == [saved]
            assert target.read_bytes() == b"BBBB"
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

    def test_staging_open_failure_leaves_no_output(self, tmp_path, monkeypatch):
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
    mock_venice_api_key, monkeypatch, tmp_path, capsys, httpx2_mock
):
    """The model path keeps a partial download as .part, not under the final name."""
    import llm

    model = VeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Stream me"
    prompt.options = VeniceSpeech.Options(output_dir=tmp_path, output_filename="out.mp3")

    class FailingStream(httpx2.SyncByteStream):
        def __iter__(self):
            yield b"chunk-1"
            raise httpx2.ReadError("connection lost")

    httpx2_mock.add_response(method="POST", url=ENDPOINT_AUDIO_SPEECH, stream=FailingStream())

    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        with pytest.raises(llm.ModelError, match="connection lost"):
            list(model.execute(prompt, True, MagicMock(), None))

    assert not (tmp_path / "out.mp3").exists()
    assert (tmp_path / "out.mp3.part").read_bytes() == b"chunk-1"
    assert "partial audio kept at" in capsys.readouterr().err


def _run(coro):
    return asyncio.run(coro)


def _async_speech_options(**kwargs):
    return VeniceSpeech.Options(**kwargs)


def test_async_speech_payload_and_file_output(httpx2_mock, mock_venice_api_key, request_json):
    """ "AsyncVeniceSpeech should send the same payload and save the same file as the sync model."""
    add_speech_response(httpx2_mock, content=b"async-audio")
    model = AsyncVeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Hello async"
    prompt.options = _async_speech_options(voice="af_sky", speed=1.25)

    async def run():
        return [chunk async for chunk in model.execute(prompt, False, MagicMock(), None)]

    results = _run(run())

    payload = request_json()
    assert payload == {
        "model": "tts-kokoro",
        "input": "Hello async",
        "voice": "af_sky",
        "response_format": "mp3",
        "speed": 1.25,
        "streaming": False,
    }
    saved = pathlib.Path(results[-1].removeprefix("Audio saved to "))
    assert saved.read_bytes() == b"async-audio"
    assert not saved.with_name(saved.name + ".part").exists()


def test_async_speech_streaming_writes_file(httpx2_mock, mock_venice_api_key, tmp_path):
    """The async model streams to a file through the same writer as the sync model."""
    add_speech_response(httpx2_mock, content=b"async-chunked")
    model = AsyncVeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Stream me"
    prompt.options = _async_speech_options(output_dir=tmp_path, output_filename="out.mp3")

    async def run():
        return [chunk async for chunk in model.execute(prompt, True, MagicMock(), None)]

    _run(run())

    assert (tmp_path / "out.mp3").read_bytes() == b"async-chunked"
    assert not (tmp_path / "out.mp3.part").exists()


def test_async_speech_streaming_failure_keeps_partial(
    httpx2_mock, mock_venice_api_key, tmp_path, capsys
):
    """The async model keeps a partial download as .part, like the sync model."""

    class FailingStream(httpx2.SyncByteStream):
        def __iter__(self):
            yield b"chunk-1"
            raise httpx2.ReadError("connection lost")

    httpx2_mock.add_response(method="POST", url=ENDPOINT_AUDIO_SPEECH, stream=FailingStream())
    model = AsyncVeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Stream me"
    prompt.options = _async_speech_options(output_dir=tmp_path, output_filename="out.mp3")

    async def run():
        async for _ in model.execute(prompt, True, MagicMock(), None):
            pass

    with pytest.raises(llm.ModelError, match="connection lost"):
        _run(run())

    assert not (tmp_path / "out.mp3").exists()
    assert (tmp_path / "out.mp3.part").read_bytes() == b"chunk-1"
    assert "partial audio kept at" in capsys.readouterr().err


def test_async_speech_http_error_is_model_error(httpx2_mock, mock_venice_api_key, tmp_path):
    """API errors become ModelError with the API's detail, sync or async."""
    httpx2_mock.add_response(
        method="POST", url=ENDPOINT_AUDIO_SPEECH, status_code=400, json={"error": "Invalid voice"}
    )
    model = AsyncVeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options = _async_speech_options(output_dir=tmp_path)

    async def run():
        async for _ in model.execute(prompt, False, MagicMock(), None):
            pass

    with pytest.raises(llm.ModelError, match="Invalid voice"):
        _run(run())


def test_async_speech_sync_and_async_payloads_match(httpx2_mock, mock_venice_api_key, request_json):
    """Sync and async speech models build identical API payloads."""
    add_speech_response(httpx2_mock, content=b"a", is_reusable=True)

    def options():
        return VeniceSpeech.Options(voice="af_sky", response_format="flac", speed=0.75)

    sync_prompt = MagicMock()
    sync_prompt.prompt = "Same"
    sync_prompt.options = options()
    sync_model = VeniceSpeech("tts-kokoro")
    with patch.object(sync_model, "get_key", return_value=mock_venice_api_key):
        list(sync_model.execute(sync_prompt, False, MagicMock(), None))
    sync_payload = request_json(-1)

    async_prompt = MagicMock()
    async_prompt.prompt = "Same"
    async_prompt.options = options()
    async_model = AsyncVeniceSpeech("tts-kokoro")

    async def run():
        return [c async for c in async_model.execute(async_prompt, False, MagicMock(), None)]

    with patch.object(async_model, "get_key", return_value=mock_venice_api_key):
        _run(run())
    async_payload = request_json(-1)

    assert sync_payload == async_payload
