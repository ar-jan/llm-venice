"""Venice audio/text-to-speech model implementation."""

import asyncio
from contextlib import contextmanager, suppress
import datetime
import os
import pathlib
from dataclasses import dataclass
import stat
import sys
import tempfile
import time
from typing import AsyncGenerator, BinaryIO, Iterable, Iterator, Literal, Optional, Union

import httpx2
import llm
from llm.utils import logging_client
from pydantic import Field

from llm_venice._files import try_rename_no_replace
from llm_venice.api.client import get_auth_headers_with_content_type
from llm_venice.api.errors import VeniceAPIError, raise_api_error
from llm_venice.constants import ENDPOINT_AUDIO_SPEECH
from llm_venice.notices import VeniceNotice, render_notice
from llm_venice.utils import (
    generate_timestamp_filename,
    get_unique_filepath,
    require_model_key,
    validate_output_directory,
)


AudioResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class VeniceSpeechOptions(llm.Options):
    """Options for Venice text-to-speech models."""

    voice: Optional[str] = Field(
        description="The voice to use when generating the audio",
        default=None,
    )
    response_format: Optional[AudioResponseFormat] = Field(
        description="The format in which the generated audio is returned",
        default="mp3",
    )
    speed: Optional[float] = Field(
        description="The speed of the generated audio. "
        "Select a value from 0.25 to 4.0. 1.0 is the default.",
        default=1.0,
        ge=0.25,
        le=4.0,
    )
    output_dir: Optional[Union[pathlib.Path, str]] = Field(
        description="Directory to save generated audio",
        default=None,
    )
    output_filename: Optional[str] = Field(
        description="Custom filename for saved audio",
        default=None,
    )
    overwrite_files: Optional[bool] = Field(
        description="Option to overwrite existing output files",
        default=False,
    )
    progress: Optional[bool] = Field(
        description="Show download progress on stderr",
        default=False,
    )


@dataclass
class SpeechGenerationResult:
    audio_bytes: Optional[bytes]
    output_path: Optional[pathlib.Path]
    content_type: Optional[str] = None
    request_json: Optional[dict] = None


@dataclass
class SpeechStreamResult:
    chunks: Iterable[bytes]
    output_path: Optional[pathlib.Path]
    content_type: Optional[str] = None
    request_json: Optional[dict] = None


@dataclass(frozen=True)
class _SpeechCLIRequest:
    payload: dict
    output_path: pathlib.Path
    progress: bool
    overwrite: bool
    use_streaming_download: bool


def _normalize_output_dir(
    output_dir: Optional[Union[pathlib.Path, str]],
) -> Optional[Union[pathlib.Path, str]]:
    if isinstance(output_dir, str) and not output_dir.strip():
        return None
    return output_dir


def _normalize_output_filename(output_filename: Optional[str]) -> Optional[str]:
    if isinstance(output_filename, str) and not output_filename.strip():
        return None
    return output_filename


def _resolve_speech_output_path(
    *,
    model_name: str,
    response_format: str,
    output_dir: Optional[Union[pathlib.Path, str]],
    output_filename: Optional[str],
    overwrite_files: bool,
) -> pathlib.Path:
    resolved_output_dir = validate_output_directory(_normalize_output_dir(output_dir))
    target_dir = resolved_output_dir or (llm.user_dir() / "audio")
    normalized_filename = _normalize_output_filename(output_filename)
    if not normalized_filename:
        normalized_filename = generate_timestamp_filename("venice", model_name, response_format)
    return get_unique_filepath(target_dir, normalized_filename, overwrite_files)


def _build_speech_payload(*, input_text: str, options_dict: dict, model_name: str) -> dict:
    return {
        "model": model_name,
        "input": input_text,
        **{k: v for k, v in options_dict.items() if v is not None},
    }


def _prepare_speech_generation(
    *,
    input_text: str,
    options_dict: dict,
    model_name: str,
) -> tuple[dict, pathlib.Path]:
    """
    Prepare a speech request payload and a resolved output path.

    Removes the CLI-only progress option and output-path options from the payload.
    Does not write anything to disk.
    """
    output_dir = options_dict.pop("output_dir", None)
    output_filename = options_dict.pop("output_filename", None)
    overwrite_files = options_dict.pop("overwrite_files", False)
    options_dict.pop("progress", None)

    response_format = options_dict.get("response_format") or "mp3"
    output_path = _resolve_speech_output_path(
        model_name=model_name,
        response_format=response_format,
        output_dir=output_dir,
        output_filename=output_filename,
        overwrite_files=overwrite_files,
    )
    payload = _build_speech_payload(
        input_text=input_text, options_dict=options_dict, model_name=model_name
    )
    return payload, output_path


def _prepare_speech_cli_request(
    *,
    input_text: str,
    options: llm.Options,
    model_name: str,
    stream: bool,
) -> _SpeechCLIRequest:
    options_dict = options.model_dump(by_alias=True)
    progress = bool(options_dict.get("progress", False))

    overwrite = bool(options_dict.get("overwrite_files", False))
    payload, resolved_output_path = _prepare_speech_generation(
        input_text=input_text, options_dict=options_dict, model_name=model_name
    )

    payload["streaming"] = stream
    use_streaming_download = progress or stream
    return _SpeechCLIRequest(
        payload=payload,
        output_path=resolved_output_path,
        progress=progress,
        overwrite=overwrite,
        use_streaming_download=use_streaming_download,
    )


def _set_speech_response_json(
    response: Union[llm.Response, llm.AsyncResponse],
    *,
    payload: dict,
    content_type: Optional[str],
    bytes_written: int,
    output_path: pathlib.Path,
) -> None:
    response.response_json = {
        "request": payload,
        "content_type": content_type,
        "bytes_written": bytes_written,
        "output_path": str(output_path),
    }


def generate_speech_result(
    *,
    input_text: str,
    options: llm.Options,
    model_name: str,
    api_key: str,
) -> SpeechGenerationResult:
    """
    Generate audio via the Venice /audio/speech API.

    Returns audio bytes in-memory along with a resolved default output path.

    Use save_speech_result(result) to save the bytes to disk.
    """
    options_dict = options.model_dump(by_alias=True)
    payload, output_path = _prepare_speech_generation(
        input_text=input_text, options_dict=options_dict, model_name=model_name
    )
    payload["streaming"] = False
    audio_bytes, content_type = _post_speech_bytes(api_key=api_key, payload=payload)
    return SpeechGenerationResult(
        audio_bytes=audio_bytes,
        output_path=output_path,
        content_type=content_type,
        request_json=payload,
    )


def save_speech_result(result: SpeechGenerationResult) -> pathlib.Path:
    """Persist a SpeechGenerationResult to disk."""
    if result.output_path is None:
        raise ValueError("No output path available to save audio")

    result.output_path.parent.mkdir(parents=True, exist_ok=True)

    if result.audio_bytes is None:
        raise ValueError("No audio bytes available to save")

    result.output_path.write_bytes(result.audio_bytes)
    return result.output_path


def _format_bytes(num_bytes: int) -> str:
    """Format bytes as a human-friendly string."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f}{unit}"
        value /= 1024.0
    return f"{value:.0f}TB"


@contextmanager
def _stream_speech_request(*, api_key: str, payload: dict, headers: dict):
    """
    Stream a /audio/speech request response.

    Uses llm's logging_client when LLM_VENICE_SHOW_RESPONSES is set.
    """
    if os.environ.get("LLM_VENICE_SHOW_RESPONSES"):
        with logging_client() as client:
            with client.stream(
                "POST",
                ENDPOINT_AUDIO_SPEECH,
                headers=headers,
                json=payload,
                timeout=120,
            ) as r:
                yield r
    else:
        with httpx2.stream(
            "POST",
            ENDPOINT_AUDIO_SPEECH,
            headers=headers,
            json=payload,
            timeout=120,
        ) as r:
            yield r


@contextmanager
def stream_speech_result(
    *,
    input_text: str,
    options: llm.Options,
    model_name: str,
    api_key: str,
) -> Iterator[SpeechStreamResult]:
    """
    Stream audio bytes via the Venice /audio/speech API without writing to disk.

    Returns a SpeechStreamResult containing an iterator of audio chunks and a resolved default output path.
    """
    options_dict = options.model_dump(by_alias=True)
    payload, output_path = _prepare_speech_generation(
        input_text=input_text, options_dict=options_dict, model_name=model_name
    )
    payload["streaming"] = True

    with _stream_speech_bytes(api_key=api_key, payload=payload) as (content_type, chunks):
        yield SpeechStreamResult(
            chunks=chunks,
            output_path=output_path,
            content_type=content_type,
            request_json=payload,
        )


@contextmanager
def _stream_speech_bytes(
    *, api_key: str, payload: dict
) -> Iterator[tuple[Optional[str], Iterable[bytes]]]:
    """Stream speech audio response bytes for a prepared payload."""
    headers = get_auth_headers_with_content_type(api_key)
    with _stream_speech_request(api_key=api_key, payload=payload, headers=headers) as r:
        try:
            r.raise_for_status()
        except httpx2.HTTPStatusError as exc:
            # A streamed body is unread; read it so the error detail can be extracted.
            r.read()
            raise_api_error("Generating speech", exc)
        yield r.headers.get("Content-Type"), r.iter_bytes()


class DownloadProgress:
    """Report download progress on stderr, throttled to one update per 0.5 s."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.bytes_written = 0
        self._start = time.monotonic()
        self._last_update = self._start

    def advance(self, num_bytes: int) -> None:
        self.bytes_written += num_bytes
        self._report()

    def finish(self) -> None:
        self._report(force=True)
        if self.enabled and sys.stderr.isatty():
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _report(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and (now - self._last_update) < 0.5:
            return
        elapsed = max(now - self._start, 0.001)
        rate = self.bytes_written / elapsed
        message = f"Downloaded {_format_bytes(self.bytes_written)} ({_format_bytes(int(rate))}/s)"
        if sys.stderr.isatty():
            sys.stderr.write(f"\r{message}")
        else:
            sys.stderr.write(f"{message}\n")
        sys.stderr.flush()
        self._last_update = now


def partial_path_for(output_path: pathlib.Path) -> pathlib.Path:
    """Return the preferred path audio is staged at before it is complete."""
    return output_path.with_name(f"{output_path.name}.part")


def _publish_audio_file(
    part_path: pathlib.Path, output_path: pathlib.Path, *, overwrite: bool
) -> pathlib.Path:
    """
    Publish complete audio without replacing another writer's file by default.

    Prefer native exclusive rename, falling back to a hard link only when that
    operation is unsupported. Both refuse existing paths, including dangling
    symlinks. Staging lives in the same directory/filesystem. If neither method
    is supported, fail safely and leave the completed staging file.
    Explicit overwrite uses replace and preserves regular-file access modes.
    """
    if overwrite:
        try:
            destination = output_path.lstat()
        except FileNotFoundError:
            destination = None
        try:
            if destination is not None and stat.S_ISREG(destination.st_mode):
                # Copy access bits, not special bits such as setuid/setgid. Do not
                # follow a destination symlink or copy its typically broad mode.
                part_path.chmod(stat.S_IMODE(destination.st_mode) & 0o777)
            os.replace(part_path, output_path)
        except BaseException:
            # If publication fails after copying a shared destination's mode,
            # keep the retained staging audio private without masking the error.
            with suppress(OSError):
                part_path.chmod(0o600)
            raise
        return output_path

    candidate = output_path
    for _ in range(5):
        try:
            if try_rename_no_replace(part_path, candidate):
                return candidate
            os.link(part_path, candidate)
        except FileExistsError:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            candidate = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
            continue
        # Publication succeeded. Failure to remove the extra link must not
        # report a failed download or delete the already published audio.
        try:
            part_path.unlink()
        except OSError as exc:
            message = (
                f"audio saved to {candidate}; could not remove staging file {part_path}: {exc}"
            )
            print(render_notice(VeniceNotice("warning", message)), file=sys.stderr)
        return candidate
    raise FileExistsError(f"could not publish audio without overwriting a file at {output_path}")


def _open_staging_file(output_path: pathlib.Path) -> tuple[BinaryIO, pathlib.Path]:
    """
    Open a private staging file next to output_path.

    The preferred ``<name>.part`` name is claimed with exclusive creation, which
    also refuses to follow a symlink planted at that path. If it is already
    taken, a unique temporary name is used, so concurrent downloads never share
    a staging file and no other writer's file is ever truncated or removed.
    """
    preferred = partial_path_for(output_path)
    try:
        fd = os.open(preferred, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        return os.fdopen(fd, "wb"), preferred
    except FileExistsError:
        pass

    fd, temp_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".part"
    )
    return os.fdopen(fd, "wb"), pathlib.Path(temp_name)


def write_audio_file(
    chunks: Iterable[bytes],
    output_path: pathlib.Path,
    reporter: DownloadProgress,
    *,
    overwrite: bool = False,
) -> pathlib.Path:
    """
    Write audio chunks to output_path via a staging file.

    Staging happens in ``<name>.part`` (or a unique ``.<name>.<random>.part`` when
    that name is taken), with private permissions. The complete file is then
    published atomically. An overwrite preserves existing regular-file access
    permissions; new files remain private.

    Unless overwrite is set, publication refuses to replace an existing path;
    collisions retry with a timestamped name. No destination is reserved or
    removed during download or failure cleanup. On failure, a non-empty staging
    file is kept and a warning names it; an empty one is removed. The original
    exception is re-raised unchanged. Return the actual destination, which may
    differ from output_path after a collision.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path: Optional[pathlib.Path] = None
    try:
        file_handle, part_path = _open_staging_file(output_path)
        with file_handle:
            for chunk in chunks:
                if not chunk:
                    continue
                file_handle.write(chunk)
                reporter.advance(len(chunk))
        saved_path = _publish_audio_file(part_path, output_path, overwrite=overwrite)
    except BaseException:
        if reporter.bytes_written and part_path is not None and part_path.exists():
            message = (
                "download interrupted; partial audio kept at "
                f"{part_path} ({_format_bytes(reporter.bytes_written)})"
            )
            # Stay click-free here: library modules must not depend on the CLI layer.
            print(render_notice(VeniceNotice("warning", message)), file=sys.stderr)
        elif part_path is not None:
            part_path.unlink(missing_ok=True)
        raise
    reporter.finish()
    return saved_path


def _stream_speech_to_file(
    *,
    api_key: str,
    payload: dict,
    output_path: pathlib.Path,
    progress: bool,
    overwrite: bool = False,
) -> tuple[Optional[str], int, pathlib.Path]:
    """Download speech audio and write it to output_path."""
    reporter = DownloadProgress(progress)
    with _stream_speech_bytes(api_key=api_key, payload=payload) as (content_type, chunks):
        saved_path = write_audio_file(chunks, output_path, reporter, overwrite=overwrite)
    return content_type, reporter.bytes_written, saved_path


def _post_speech_bytes(*, api_key: str, payload: dict) -> tuple[bytes, Optional[str]]:
    """POST /audio/speech and return the full audio response bytes."""
    headers = get_auth_headers_with_content_type(api_key)

    if os.environ.get("LLM_VENICE_SHOW_RESPONSES"):
        with logging_client() as client:
            r = client.post(ENDPOINT_AUDIO_SPEECH, headers=headers, json=payload, timeout=120)
            try:
                r.raise_for_status()
            except httpx2.HTTPStatusError as exc:
                raise_api_error("Generating speech", exc)
            return r.content, r.headers.get("Content-Type")

    r = httpx2.post(ENDPOINT_AUDIO_SPEECH, headers=headers, json=payload, timeout=120)
    try:
        r.raise_for_status()
    except httpx2.HTTPStatusError as exc:
        raise_api_error("Generating speech", exc)
    return r.content, r.headers.get("Content-Type")


def _write_audio_bytes(
    *, output_path: pathlib.Path, audio_bytes: bytes, overwrite: bool = False
) -> pathlib.Path:
    """Write audio bytes to a path the caller chose, creating the directory if needed."""
    return write_audio_file(
        [audio_bytes], output_path, DownloadProgress(False), overwrite=overwrite
    )


class VeniceSpeech(llm.KeyModel):
    """Venice AI text-to-speech model."""

    can_stream = True
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"

    def __init__(self, model_id, model_name=None):
        self.model_id = f"venice/{model_id}"
        self.model_name = model_name or model_id

    def __str__(self):
        return f"Venice Speech: {self.model_id}"

    class Options(VeniceSpeechOptions):  # type: ignore[override]
        pass

    def execute(self, prompt, stream, response, conversation=None, key=None):
        """Execute text-to-speech request."""
        api_key = require_model_key(self, key)

        try:
            options_dict = prompt.options.model_dump(by_alias=True)
            validate_output_directory(
                _normalize_output_dir(options_dict.get("output_dir")),
                create_if_missing=True,
            )
            request = _prepare_speech_cli_request(
                input_text=prompt.prompt,
                options=prompt.options,
                model_name=self.model_name,
                stream=stream,
            )

            if request.use_streaming_download:
                yield f"Writing audio to {request.output_path}\n"

                content_type, bytes_written, saved_path = _stream_speech_to_file(
                    api_key=api_key,
                    payload=request.payload,
                    output_path=request.output_path,
                    progress=request.progress,
                    overwrite=request.overwrite,
                )

                _set_speech_response_json(
                    response,
                    payload=request.payload,
                    content_type=content_type,
                    bytes_written=bytes_written,
                    output_path=saved_path,
                )
                yield f"Audio saved to {saved_path}"
                return

            audio_bytes, content_type = _post_speech_bytes(api_key=api_key, payload=request.payload)
            saved_path = _write_audio_bytes(
                output_path=request.output_path,
                audio_bytes=audio_bytes,
                overwrite=request.overwrite,
            )

            _set_speech_response_json(
                response,
                payload=request.payload,
                content_type=content_type,
                bytes_written=len(audio_bytes),
                output_path=saved_path,
            )

            yield f"Audio saved to {saved_path}"
        except (OSError, ValueError, VeniceAPIError, httpx2.RequestError) as exc:
            raise llm.ModelError(str(exc)) from exc


class AsyncVeniceSpeech(llm.AsyncKeyModel):
    """Asynchronous Venice AI text-to-speech model."""

    can_stream = True
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"

    def __init__(self, model_id, model_name=None):
        self.model_id = f"venice/{model_id}"
        self.model_name = model_name or model_id

    def __str__(self):
        return f"Venice Speech: {self.model_id}"

    class Options(VeniceSpeechOptions):  # type: ignore[override]
        pass

    async def execute(
        self, prompt, stream, response, conversation=None, key=None
    ) -> AsyncGenerator[str, None]:
        """Execute text-to-speech request asynchronously."""
        api_key = require_model_key(self, key)

        try:
            options_dict = prompt.options.model_dump(by_alias=True)
            validate_output_directory(
                _normalize_output_dir(options_dict.get("output_dir")),
                create_if_missing=True,
            )
            request = _prepare_speech_cli_request(
                input_text=prompt.prompt,
                options=prompt.options,
                model_name=self.model_name,
                stream=stream,
            )

            if request.use_streaming_download:
                yield f"Writing audio to {request.output_path}\n"

                content_type, bytes_written, saved_path = await asyncio.to_thread(
                    _stream_speech_to_file,
                    api_key=api_key,
                    payload=request.payload,
                    output_path=request.output_path,
                    progress=request.progress,
                    overwrite=request.overwrite,
                )

                _set_speech_response_json(
                    response,
                    payload=request.payload,
                    content_type=content_type,
                    bytes_written=bytes_written,
                    output_path=saved_path,
                )
                yield f"Audio saved to {saved_path}"
                return

            audio_bytes, content_type = await asyncio.to_thread(
                _post_speech_bytes, api_key=api_key, payload=request.payload
            )
            saved_path = await asyncio.to_thread(
                _write_audio_bytes,
                output_path=request.output_path,
                audio_bytes=audio_bytes,
                overwrite=request.overwrite,
            )

            _set_speech_response_json(
                response,
                payload=request.payload,
                content_type=content_type,
                bytes_written=len(audio_bytes),
                output_path=saved_path,
            )

            yield f"Audio saved to {saved_path}"
        except (OSError, ValueError, VeniceAPIError, httpx2.RequestError) as exc:
            raise llm.ModelError(str(exc)) from exc
