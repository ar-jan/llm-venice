"""Exclusive rename bindings and publication fallback/error behavior."""

import ctypes
import errno
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_venice import _files
from llm_venice.models.audio import DownloadProgress, write_audio_file


@pytest.fixture
def native_rename(tmp_path):
    source, target = tmp_path / "probe.part", tmp_path / "probe"
    source.write_bytes(b"probe")
    if not _files.try_rename_no_replace(source, target):
        pytest.skip("native exclusive rename unavailable on this platform/filesystem")
    target.unlink()
    return _files.try_rename_no_replace


def test_native_rename_moves_unicode_file_without_hard_links(tmp_path, monkeypatch, native_rename):
    def no_links(*args, **kwargs):
        pytest.fail("native publication must not need hard links")

    monkeypatch.setattr(os, "link", no_links)
    target = tmp_path / "héllo-音声.mp3"
    saved = write_audio_file([b"audio"], target, DownloadProgress(False))

    assert saved == target
    assert target.read_bytes() == b"audio"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_symlink"])
def test_native_rename_refuses_every_existing_destination(tmp_path, native_rename, kind):
    source, target = tmp_path / "audio.part", tmp_path / "audio.mp3"
    source.write_bytes(b"new")
    if kind == "file":
        target.write_bytes(b"old")
    elif kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError):
        native_rename(source, target)

    assert source.read_bytes() == b"new"
    if kind == "file":
        assert target.read_bytes() == b"old"
    elif kind == "directory":
        assert target.is_dir()
    else:
        assert target.is_symlink()
        assert not target.exists()


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_libc_binding_signature_and_exclusive_flag(monkeypatch, tmp_path, platform):
    function = Mock(return_value=0)
    libc = SimpleNamespace(renameat2=function, renamex_np=function)
    loader = Mock(return_value=libc)
    monkeypatch.setattr(ctypes, "CDLL", loader)
    loaded = _files._rename_function.__wrapped__(platform)
    monkeypatch.setattr(_files, "_rename_function", lambda _platform: loaded)
    monkeypatch.setattr(_files, "sys", SimpleNamespace(platform=platform))
    source, target = tmp_path / "音声.part", tmp_path / "音声.mp3"

    assert _files.try_rename_no_replace(source, target)
    loader.assert_called_once_with(None, use_errno=True)
    assert function.restype == ctypes.c_int
    if platform == "linux":
        assert function.argtypes == [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.assert_called_once_with(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    else:
        assert function.argtypes == [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.assert_called_once_with(os.fsencode(source), os.fsencode(target), 4)


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_missing_libc_symbol_allows_fallback(monkeypatch, platform):
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace())
    assert _files._rename_function.__wrapped__(platform) is None


@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP])
@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_unsupported_native_operation_allows_fallback(monkeypatch, tmp_path, platform, error):
    def unsupported(*args):
        ctypes.set_errno(error)
        return -1

    monkeypatch.setattr(_files, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(_files, "_rename_function", lambda _platform: unsupported)
    assert not _files.try_rename_no_replace(tmp_path / "a", tmp_path / "b")


@pytest.mark.parametrize(
    "error", [errno.EEXIST, errno.EACCES, errno.EPERM, errno.ENOSPC, errno.EIO]
)
def test_native_errors_are_preserved_without_fallback(monkeypatch, tmp_path, capsys, error):
    def fail(*args):
        ctypes.set_errno(error)
        return -1

    def no_links(*args, **kwargs):
        pytest.fail("collisions and operational errors must not trigger the fallback")

    monkeypatch.setattr(_files, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(_files, "_rename_function", lambda _platform: fail)
    monkeypatch.setattr(os, "link", no_links)
    source, target = tmp_path / "out.mp3.part", tmp_path / "out.mp3"
    if error == errno.EEXIST:
        with pytest.raises(FileExistsError) as caught:
            _files.try_rename_no_replace(source, target)
    else:
        with pytest.raises(OSError) as caught:
            write_audio_file([b"audio"], target, DownloadProgress(False))
        assert source.read_bytes() == b"audio"
        assert not target.exists()
        assert str(source) in capsys.readouterr().err
    assert caught.value.errno == error
    assert caught.value.filename == str(source)
    assert caught.value.filename2 == str(target)


def test_native_rename_retries_interrupted_syscall(monkeypatch, tmp_path):
    calls = []

    def interrupted_once(*args):
        calls.append(args)
        ctypes.set_errno(errno.EINTR)
        return -1 if len(calls) == 1 else 0

    monkeypatch.setattr(_files, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(_files, "_rename_function", lambda _platform: interrupted_once)
    assert _files.try_rename_no_replace(tmp_path / "a", tmp_path / "b")
    assert len(calls) == 2


@pytest.mark.parametrize("invalid_source", [False, True])
def test_native_binding_rejects_nul_in_paths(monkeypatch, tmp_path, invalid_source):
    function = Mock(return_value=0)
    monkeypatch.setattr(_files, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(_files, "_rename_function", lambda _platform: function)
    source = tmp_path / ("a\0suffix" if invalid_source else "a")
    target = tmp_path / ("b" if invalid_source else "b\0suffix")
    with pytest.raises(ValueError, match="null byte"):
        _files.try_rename_no_replace(source, target)
    function.assert_not_called()


@pytest.mark.parametrize("error", [None, errno.EEXIST, errno.EACCES, errno.ENOTSUP])
def test_windows_uses_python_rename(monkeypatch, tmp_path, error):
    source, target = tmp_path / "a", tmp_path / "b"
    exception = OSError(error, "rename failed") if error is not None else None
    rename = Mock(side_effect=exception)
    monkeypatch.setattr(_files, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(os, "rename", rename)
    if error in (errno.EEXIST, errno.EACCES):
        with pytest.raises(OSError) as caught:
            _files.try_rename_no_replace(source, target)
        assert caught.value is exception
    else:
        assert _files.try_rename_no_replace(source, target) is (error is None)
    rename.assert_called_once_with(source, target)


@pytest.mark.parametrize("unavailable", ["symbol", "platform", "filesystem"])
def test_hard_link_fallback_keeps_collision_protection(monkeypatch, tmp_path, unavailable):
    if unavailable == "symbol":
        monkeypatch.setattr(_files, "_rename_function", lambda _platform: None)
    elif unavailable == "platform":
        monkeypatch.setattr(_files, "sys", SimpleNamespace(platform="other"))
    else:

        def unsupported(*args):
            ctypes.set_errno(errno.EOPNOTSUPP)
            return -1

        monkeypatch.setattr(_files, "sys", SimpleNamespace(platform="linux"))
        monkeypatch.setattr(_files, "_rename_function", lambda _platform: unsupported)
    target = tmp_path / "out.mp3"
    target.write_bytes(b"existing")
    saved = write_audio_file([b"new"], target, DownloadProgress(False))
    assert saved != target
    assert saved.read_bytes() == b"new"
    assert target.read_bytes() == b"existing"
    assert not list(tmp_path.glob("*.part"))
