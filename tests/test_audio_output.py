"""Output collisions must preserve audio and report the path actually written."""

import asyncio
import errno
import os
import pathlib
import stat
from unittest.mock import MagicMock

import httpx2
import pytest

from llm_venice.models.audio import (
    AsyncVeniceSpeech,
    DownloadProgress,
    VeniceSpeech,
    write_audio_file,
)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_model_output_collision(
    httpx2_mock, mock_venice_api_key, tmp_path, asynchronous, stream, overwrite
):
    target = tmp_path / "out.mp3"

    def racing_response(request):
        # The name was available during request preparation, but another writer
        # finishes before this response arrives.
        target.write_bytes(b"other download")
        return httpx2.Response(200, content=b"new audio", headers={"Content-Type": "audio/mpeg"})

    httpx2_mock.add_callback(
        racing_response, method="POST", url="https://api.venice.ai/api/v1/audio/speech"
    )
    model = AsyncVeniceSpeech("tts-kokoro") if asynchronous else VeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options = model.Options(
        output_dir=tmp_path, output_filename=target.name, overwrite_files=overwrite
    )
    response = MagicMock()

    if isinstance(model, AsyncVeniceSpeech):

        async def collect():
            return [message async for message in model.execute(prompt, stream, response)]

        messages = asyncio.run(collect())
    else:
        messages = list(model.execute(prompt, stream, response))

    saved = pathlib.Path(response.response_json["output_path"])
    assert messages[-1] == f"Audio saved to {saved}"
    assert saved.read_bytes() == b"new audio"
    assert response.response_json["bytes_written"] == len(b"new audio")
    if overwrite:
        assert saved == target
    else:
        assert saved != target
        assert target.read_bytes() == b"other download"
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("partial", [b"", b"unfinished"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_failed_writer_preserves_concurrent_overwrite(tmp_path, capsys, partial, error_type):
    target = tmp_path / "out.mp3"
    error = error_type("first download interrupted")

    def first_writer():
        yield partial
        assert not target.exists()
        write_audio_file([b"completed"], target, DownloadProgress(False), overwrite=True)
        assert target.read_bytes() == b"completed"
        raise error

    with pytest.raises(error_type) as caught:
        write_audio_file(first_writer(), target, DownloadProgress(False))

    assert caught.value is error
    assert target.read_bytes() == b"completed"
    parts = list(tmp_path.glob("*.part"))
    if partial:
        assert parts == [tmp_path / "out.mp3.part"]
        assert parts[0].read_bytes() == partial
        assert str(parts[0]) in capsys.readouterr().err
    else:
        assert parts == []
        assert capsys.readouterr().err == ""


@pytest.fixture
def umask_022():
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644])
def test_overwrite_preserves_access_mode_with_private_staging(tmp_path, umask_022, fallback, mode):
    target = tmp_path / "out.mp3"
    target.write_bytes(b"original")
    target.chmod(mode)
    stale = tmp_path / "out.mp3.part"
    if fallback:
        stale.write_bytes(b"stale")

    def chunks():
        (part,) = tmp_path.glob(".out.mp3.*.part") if fallback else [stale]
        assert stat.S_IMODE(part.stat().st_mode) == 0o600
        assert target.read_bytes() == b"original"
        yield b"replacement"

    saved = write_audio_file(chunks(), target, DownloadProgress(False), overwrite=True)

    assert saved == target
    assert target.read_bytes() == b"replacement"
    assert stat.S_IMODE(target.stat().st_mode) == mode
    if fallback:
        assert stale.read_bytes() == b"stale"
    assert list(tmp_path.glob("*.part")) == ([stale] if fallback else [])


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_new_output_is_private(tmp_path, umask_022, fallback, overwrite):
    target = tmp_path / "out.mp3"
    stale = tmp_path / "out.mp3.part"
    if fallback:
        stale.write_bytes(b"stale")

    saved = write_audio_file([b"audio"], target, DownloadProgress(False), overwrite=overwrite)

    assert saved == target
    assert target.read_bytes() == b"audio"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_overwrite_replaces_symlink_without_copying_or_changing_its_permissions(
    tmp_path, umask_022
):
    target = tmp_path / "out.mp3"
    victim = tmp_path / "victim.mp3"
    victim.write_bytes(b"original")
    victim.chmod(0o640)
    target.symlink_to(victim)

    saved = write_audio_file([b"audio"], target, DownloadProgress(False), overwrite=True)

    assert saved == target
    assert not target.is_symlink()
    assert target.read_bytes() == b"audio"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert victim.read_bytes() == b"original"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o640


@pytest.mark.parametrize("fallback", [False, True])
def test_interrupted_overwrite_keeps_partial_private(tmp_path, umask_022, fallback):
    target = tmp_path / "out.mp3"
    target.write_bytes(b"original")
    target.chmod(0o644)
    preferred = tmp_path / "out.mp3.part"
    if fallback:
        preferred.write_bytes(b"stale")

    def chunks():
        yield b"partial"
        raise OSError("interrupted")

    with pytest.raises(OSError, match="interrupted"):
        write_audio_file(chunks(), target, DownloadProgress(False), overwrite=True)

    (part,) = tmp_path.glob(".out.mp3.*.part") if fallback else [preferred]
    assert part.read_bytes() == b"partial"
    assert stat.S_IMODE(part.stat().st_mode) == 0o600
    assert target.read_bytes() == b"original"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_unsupported_publication_keeps_audio_without_overwriting(tmp_path, monkeypatch, capsys):
    target = tmp_path / "out.mp3"
    target.write_bytes(b"original")
    error = OSError(errno.EOPNOTSUPP, "hard links unavailable")

    def fail_link(*args, **kwargs):
        raise error

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError) as caught:
        write_audio_file([b"complete audio"], target, DownloadProgress(False))

    assert caught.value is error
    assert target.read_bytes() == b"original"
    part = tmp_path / "out.mp3.part"
    assert part.read_bytes() == b"complete audio"
    assert str(part) in capsys.readouterr().err


@pytest.mark.parametrize("fallback", [False, True])
def test_failed_overwrite_publication_restores_private_staging(
    tmp_path, monkeypatch, umask_022, fallback
):
    target = tmp_path / "out.mp3"
    target.write_bytes(b"original")
    target.chmod(0o644)
    preferred = tmp_path / "out.mp3.part"
    if fallback:
        preferred.write_bytes(b"stale")
    error = OSError("cannot replace destination")

    def fail_replace(*args, **kwargs):
        raise error

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError) as caught:
        write_audio_file([b"complete audio"], target, DownloadProgress(False), overwrite=True)

    assert caught.value is error
    assert target.read_bytes() == b"original"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    (part,) = tmp_path.glob(".out.mp3.*.part") if fallback else [preferred]
    assert part.read_bytes() == b"complete audio"
    assert stat.S_IMODE(part.stat().st_mode) == 0o600


def test_staging_cleanup_failure_still_reports_success(tmp_path, monkeypatch, capsys):
    target = tmp_path / "out.mp3"
    part = tmp_path / "out.mp3.part"
    real_unlink = pathlib.Path.unlink

    def fail_staging_unlink(path, *args, **kwargs):
        if path == part:
            raise OSError("cannot remove staging link")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", fail_staging_unlink)
    saved = write_audio_file([b"audio"], target, DownloadProgress(False))

    assert saved == target
    assert target.read_bytes() == b"audio"
    assert part.read_bytes() == b"audio"
    assert f"audio saved to {target}" in capsys.readouterr().err
