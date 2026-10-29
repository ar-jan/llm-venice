"""Retain actual staging bytes and the original error when file I/O fails."""

import io
import json
import os
import pathlib
import subprocess
import sys

import pytest

from llm_venice.models import audio
from llm_venice import _files


@pytest.mark.skipif(os.name == "nt", reason="RLIMIT_FSIZE and SIGXFSZ require Unix")
def test_real_flush_failure_reports_actual_retained_size(tmp_path):
    # Limit only the child process. The buffered writer can accept more bytes
    # than the OS allows it to flush, without filling a disk or mocking writes.
    script = """
import errno, json, pathlib, resource, signal, sys
from llm_venice.models.audio import DownloadProgress, write_audio_file
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))
target = pathlib.Path(sys.argv[1])
try:
    write_audio_file([b'x' * 8192], target, DownloadProgress(False))
except OSError as exc:
    assert exc.errno == errno.EFBIG
else:
    raise AssertionError('expected a real file-size failure')
part = target.with_name(target.name + '.part')
print(json.dumps({'size': part.stat().st_size, 'published': target.exists()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "out.mp3")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"size": 4096, "published": False}
    assert "out.mp3.part (4KB)" in result.stderr
    assert "8KB" not in result.stderr


def test_partial_write_failure_keeps_bytes_even_without_progress(tmp_path, monkeypatch, capsys):
    target = tmp_path / "out.mp3"
    open_staging = _files.open_staging_file
    error = OSError("disk full during write")

    class PartialWriter:
        def __init__(self, file):
            self.file = file

        def write(self, chunk):
            self.file.write(chunk[:4])
            raise error

        def close(self):
            self.file.close()

    def open_partial(path):
        file, part = open_staging(path)
        return PartialWriter(file), part

    monkeypatch.setattr(_files, "open_staging_file", open_partial)
    reporter = audio.DownloadProgress(False)
    with pytest.raises(OSError) as caught:
        audio.write_audio_file([b"abcdefgh"], target, reporter)

    assert caught.value is error
    assert reporter.bytes_written == 0
    assert not target.exists()
    assert (tmp_path / "out.mp3.part").read_bytes() == b"abcd"
    assert "out.mp3.part (4B)" in capsys.readouterr().err


def test_empty_file_after_flush_failure_is_removed(tmp_path, monkeypatch, capsys):
    target = tmp_path / "out.mp3"
    open_staging = _files.open_staging_file
    error = OSError("flush failed before any bytes reached disk")

    class UnflushedWriter:
        def __init__(self, file):
            self.file = file

        def write(self, chunk):
            return len(chunk)

        def close(self):
            self.file.close()
            raise error

    def open_unflushed(path):
        file, part = open_staging(path)
        return UnflushedWriter(file), part

    monkeypatch.setattr(_files, "open_staging_file", open_unflushed)
    reporter = audio.DownloadProgress(False)
    with pytest.raises(OSError) as caught:
        audio.write_audio_file([b"buffered"], target, reporter)

    assert caught.value is error
    assert reporter.bytes_written == 8
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup", ["close", "stat", "unlink", "stderr"])
def test_cleanup_cannot_replace_original_failure(
    tmp_path, monkeypatch, capsys, error_type, cleanup
):
    target = tmp_path / "out.mp3"
    part = tmp_path / "out.mp3.part"
    original = error_type("original stream failure")
    cleanup_error = OSError("secondary cleanup failure")
    data = b"" if cleanup == "unlink" else b"partial"

    def chunks():
        yield data
        raise original

    # Keep stdout/stderr and pathlib restored before inspecting the result.
    with monkeypatch.context() as patch:
        if cleanup == "close":
            open_staging = _files.open_staging_file

            class CloseFailure:
                def __init__(self, file):
                    self.file = file

                def write(self, chunk):
                    return self.file.write(chunk)

                def close(self):
                    self.file.close()
                    raise cleanup_error

            def open_failing_close(path):
                file, staging = open_staging(path)
                return CloseFailure(file), staging

            patch.setattr(_files, "open_staging_file", open_failing_close)
        elif cleanup == "stat":
            real_stat = pathlib.Path.stat

            def fail_stat(path, *args, **kwargs):
                if path == part:
                    raise cleanup_error
                return real_stat(path, *args, **kwargs)

            patch.setattr(pathlib.Path, "stat", fail_stat)
        elif cleanup == "unlink":
            real_unlink = pathlib.Path.unlink

            def fail_unlink(path, *args, **kwargs):
                if path == part:
                    raise cleanup_error
                return real_unlink(path, *args, **kwargs)

            patch.setattr(pathlib.Path, "unlink", fail_unlink)
        else:
            closed_stderr = io.StringIO()
            closed_stderr.close()
            patch.setattr(audio.sys, "stderr", closed_stderr)

        with pytest.raises(error_type) as caught:
            audio.write_audio_file(chunks(), target, audio.DownloadProgress(False))

    assert caught.value is original
    assert not target.exists()
    assert part.read_bytes() == data
    if cleanup == "stat":
        assert "size unavailable" in capsys.readouterr().err
