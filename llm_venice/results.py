"""Buffered media values. Saves are explicit and atomic per file, not per batch."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from ._files import write_chunks


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
