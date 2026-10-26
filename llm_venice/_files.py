"""Platform bindings for publishing files without replacing an existing path."""

import ctypes
import errno
from functools import lru_cache
import os
import pathlib
import sys
from typing import Callable, Optional


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
