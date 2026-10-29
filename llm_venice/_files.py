"""Platform bindings for publishing files without replacing an existing path."""

from collections.abc import AsyncIterable, Iterable
from contextlib import suppress
from dataclasses import dataclass
import asyncio
import ctypes
import datetime
import errno
from functools import lru_cache
import os
import pathlib
import sys
import stat
import tempfile
from typing import Any, BinaryIO, Callable, Optional, TypeVar


_UNSUPPORTED = {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}


@lru_cache(maxsize=2)
def _rename_function(platform: str) -> Optional[Callable[..., int]]:
    """Load libc lazily; older libcs may not export the exclusive rename API."""
    if platform not in {"linux", "darwin"}:
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if platform == "linux":
            function = libc.renameat2
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
        else:
            function = libc.renamex_np
            function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        return function
    except (AttributeError, OSError):
        return None


def try_rename_no_replace(source: pathlib.Path, destination: pathlib.Path) -> bool:
    """Rename atomically, returning False only when exclusive rename is unsupported.

    Collisions and operational failures are raised as normal OSError subclasses.
    Never use ordinary Unix rename, which can silently replace the destination.
    """
    if sys.platform == "win32":
        # Python's Windows rename refuses to replace an existing destination.
        try:
            os.rename(source, destination)
        except OSError as exc:
            if exc.errno in _UNSUPPORTED:
                return False
            raise
        return True

    function = _rename_function(sys.platform)
    if function is None:
        return False
    source_bytes, destination_bytes = os.fsencode(source), os.fsencode(destination)
    # c_char_p would silently truncate at NUL, unlike Python's filesystem APIs.
    if b"\0" in source_bytes or b"\0" in destination_bytes:
        raise ValueError("embedded null byte")
    if sys.platform == "linux":
        # Linux: AT_FDCWD = -100, RENAME_NOREPLACE = 1 (renameat2(2)).
        args = (-100, source_bytes, -100, destination_bytes, 1)
    else:
        # Darwin: RENAME_EXCL = 0x4 (sys/stdio.h, renamex_np(2)).
        args = (source_bytes, destination_bytes, 0x4)
    while True:
        if function(*args) == 0:
            return True
        error = ctypes.get_errno()
        if error == errno.EINTR:
            continue
        if error in _UNSUPPORTED:
            return False
        raise OSError(error, os.strerror(error), os.fspath(source), None, os.fspath(destination))


@dataclass(frozen=True)
class Publication:
    """Completed publication, possibly with a harmless extra staging link."""

    path: pathlib.Path
    cleanup_error: OSError | None = None
    staging_path: pathlib.Path | None = None


def open_staging_file(output_path: pathlib.Path) -> tuple[BinaryIO, pathlib.Path]:
    """Claim private staging in the destination directory, never following links."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = output_path.with_name(f"{output_path.name}.part")
    try:
        fd = os.open(preferred, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        fd, name = tempfile.mkstemp(
            dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".part"
        )
        preferred = pathlib.Path(name)
    return os.fdopen(fd, "wb"), preferred


def publish_file(
    staging: pathlib.Path, destination: pathlib.Path, *, overwrite: bool = False
) -> Publication:
    """Publish once, atomically. Filename retries belong to the adapter caller."""
    if overwrite:
        try:
            existing = destination.lstat()
        except FileNotFoundError:
            existing = None
        try:
            if existing is not None and stat.S_ISREG(existing.st_mode):
                staging.chmod(stat.S_IMODE(existing.st_mode) & 0o777)
            os.replace(staging, destination)
        except BaseException:
            with suppress(OSError):
                staging.chmod(0o600)
            raise
        return Publication(destination)
    if try_rename_no_replace(staging, destination):
        return Publication(destination)
    os.link(staging, destination)
    try:
        staging.unlink()
    except OSError as exc:
        return Publication(destination, exc, staging)
    return Publication(destination)


def publish_media_file(
    staging: pathlib.Path, destination: pathlib.Path, *, overwrite: bool = False
) -> Publication:
    """Publish generated media, picking a timestamped name if another writer won the path."""
    candidate = destination
    for _ in range(5):
        try:
            outcome = publish_file(staging, candidate, overwrite=overwrite)
        except FileExistsError:
            if overwrite:
                raise
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            candidate = destination.with_name(f"{destination.stem}_{timestamp}{destination.suffix}")
            continue
        if outcome.cleanup_error is not None:
            with suppress(OSError, ValueError):
                print(
                    f"Warning: file saved to {outcome.path}; could not remove staging file "
                    f"{outcome.staging_path}: {outcome.cleanup_error}",
                    file=sys.stderr,
                )
        return outcome
    raise FileExistsError(f"could not publish file without overwriting a file at {destination}")


def retain_partial(staging: pathlib.Path, error: BaseException) -> None:
    """Remove empty staging; attach nonempty/unknown-size staging to the error."""
    try:
        if staging.stat().st_size == 0:
            staging.unlink()
            return
    except FileNotFoundError:
        return
    except OSError:
        pass
    error.partial_path = staging  # type: ignore[attr-defined]


def write_chunks(
    chunks: Iterable[bytes],
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    publish: Callable[..., Publication] = publish_file,
) -> Publication:
    """Write privately, then publish. Preserve the original error and retained bytes."""
    destination = pathlib.Path(path)
    if destination.is_dir():
        raise IsADirectoryError(str(destination))
    handle, staging = open_staging_file(destination)
    try:
        for chunk in chunks:
            handle.write(chunk)
        handle.close()
        return publish(staging, destination, overwrite=overwrite)
    except BaseException as exc:
        with suppress(OSError, ValueError):
            handle.close()
        retain_partial(staging, exc)
        raise


_R = TypeVar("_R")


async def async_file_call(function: Callable[..., _R], *args: Any, **kwargs: Any) -> _R:
    """Wait for a running file operation before propagating cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except BaseException:
        # Repeated cancellation must not race close/unlink against a running write.
        while not task.done():
            try:
                await asyncio.shield(task)
            except BaseException:
                pass
        # Retrieve any exception, keeping the original cancellation/failure.
        if not task.cancelled():
            task.exception()
        raise


async def write_async_chunks(
    chunks: AsyncIterable[bytes],
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    publish: Callable[..., Publication] = publish_file,
) -> Publication:
    """Async network consumption with cancellation-safe file writes and shared publication."""
    destination = pathlib.Path(path)
    if destination.is_dir():
        raise IsADirectoryError(str(destination))
    handle, staging = open_staging_file(destination)
    try:
        async for chunk in chunks:
            await async_file_call(handle.write, chunk)
        await async_file_call(handle.close)
        return await async_file_call(publish, staging, destination, overwrite=overwrite)
    except BaseException as exc:
        with suppress(OSError, ValueError):
            handle.close()
        retain_partial(staging, exc)
        raise
