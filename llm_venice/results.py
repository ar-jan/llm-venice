"""Buffered media values. Saves are explicit and atomic per file, not per batch."""

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import httpx2

from ._files import write_chunks, write_async_chunks
from .errors import VeniceConnectionError


@dataclass(frozen=True)
class GeneratedImage:
    """A generated or upscaled image. Saving never transcodes its bytes."""

    data: bytes
    format: str
    content_type: str | None

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Save to an exact path; collisions raise unless overwrite is requested."""
        return write_chunks((self.data,), path, overwrite=overwrite).path


@dataclass(frozen=True)
class ImageResult:
    """Returned images and independent moderation/omitted-variant facts.

    Images remain available even when content_violation is true. A failed batch
    save leaves earlier files published; errors may identify a retained partial_path.
    """

    model: str
    images: tuple[GeneratedImage, ...]
    content_violation: bool = False
    content_policy_violation_count: int = 0
    blurred: bool = False
    timing: Mapping[str, Any] | None = None
    raw: Mapping[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def save(
        self,
        directory: str | os.PathLike[str],
        *,
        filename: str = "image",
        overwrite: bool = False,
    ) -> list[Path]:
        """Save into a directory, numbering batches before each filename suffix."""
        if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise ValueError("filename must be a basename, not a path")
        paths: list[Path] = []
        for index, image in enumerate(self.images, 1):
            name = Path(filename)
            if not name.suffix:
                name = name.with_suffix(f".{image.format}")
            if len(self.images) > 1:
                name = name.with_name(f"{name.stem}_{index}{name.suffix}")
            paths.append(image.save(Path(directory) / name, overwrite=overwrite))
        return paths


@dataclass(frozen=True)
class SpeechResult:
    """Buffered speech with its actual output format; no implicit file writes."""

    model: str
    data: bytes
    content_type: str | None
    format: str

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Save to an exact path, creating parents without prompting."""
        return write_chunks((self.data,), path, overwrite=overwrite).path


class SpeechStream:
    """Context-bound, single-use speech. Consume only inside client.stream_speech()."""

    def __init__(self, response: httpx2.Response, format: str) -> None:
        self.content_type = response.headers.get("content-type")
        self.format = format
        self._response = response
        self._claimed = False
        self._active = True

    def _claim(self) -> None:
        if not self._active:
            raise ValueError("speech stream context has exited")
        if self._claimed:
            raise ValueError("speech stream has already been consumed")
        self._claimed = True

    def iter_bytes(self) -> Iterator[bytes]:
        """Claim the stream once; later body failures become VeniceConnectionError."""
        self._claim()
        return self._iterate()

    def close(self) -> None:
        """End the stream lifetime and close its response (idempotent)."""
        self._active = False
        self._response.close()

    def _iterate(self) -> Iterator[bytes]:
        if not self._active:
            raise ValueError("speech stream context has exited")
        try:
            iterator = self._response.iter_bytes()
            while True:
                if not self._active:
                    raise ValueError("speech stream context has exited")
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                yield chunk
        except httpx2.RequestError as exc:
            raise VeniceConnectionError(f"Generating speech failed: {exc}") from exc
        finally:
            self._response.close()

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Consume into an exact file path with atomic publication and partial retention."""
        return write_chunks(self.iter_bytes(), path, overwrite=overwrite).path


class AsyncSpeechStream:
    """Async counterpart of SpeechStream; network I/O never uses a worker thread."""

    def __init__(self, response: httpx2.Response, format: str) -> None:
        self.content_type = response.headers.get("content-type")
        self.format = format
        self._response = response
        self._claimed = False
        self._active = True

    def _claim(self) -> None:
        if not self._active:
            raise ValueError("speech stream context has exited")
        if self._claimed:
            raise ValueError("speech stream has already been consumed")
        self._claimed = True

    def aiter_bytes(self) -> AsyncIterator[bytes]:
        """Claim the byte iterator once; consume it inside the stream context."""
        self._claim()
        return self._iterate()

    async def aclose(self) -> None:
        """End the stream lifetime and close its response (idempotent)."""
        self._active = False
        await self._response.aclose()

    async def _iterate(self) -> AsyncIterator[bytes]:
        if not self._active:
            raise ValueError("speech stream context has exited")
        try:
            iterator = self._response.aiter_bytes()
            while True:
                if not self._active:
                    raise ValueError("speech stream context has exited")
                try:
                    chunk = await anext(iterator)
                except StopAsyncIteration:
                    break
                yield chunk
        except httpx2.RequestError as exc:
            raise VeniceConnectionError(f"Generating speech failed: {exc}") from exc
        finally:
            await self._response.aclose()

    async def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Consume into an exact path, awaiting outstanding file writes on cancellation."""
        return (await write_async_chunks(self.aiter_bytes(), path, overwrite=overwrite)).path
