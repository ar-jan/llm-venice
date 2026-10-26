"""Venice audio/text-to-speech model implementation."""

from contextlib import contextmanager, suppress
import datetime
import os
import pathlib
from dataclasses import dataclass
import sys
import time
from typing import Iterable, Iterator, Literal, Optional, Union

import httpx2
import llm
from pydantic import Field
from typing import cast
from ._common import initialize_model, warn_deprecated, model_error_boundary
from llm_venice.client import VeniceClient, AsyncVeniceClient
from llm_venice.params import SpeechParams, copy_params
from llm_venice._endpoints import speech_body
from llm_venice import _files
from llm.utils import logging_client

from llm_venice.api.client import get_auth_headers_with_content_type
from llm_venice.api.errors import raise_api_error
from llm_venice.constants import ENDPOINT_AUDIO_SPEECH
from llm_venice.utils import (
    generate_timestamp_filename,
    get_unique_filepath,
    require_model_key,
    validate_output_directory,
)


AudioResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class SpeechOutputOptions(llm.Options):
    output_dir: pathlib.Path | str | None = None
    output_filename: str | None = None
    overwrite_files: bool = False
    progress: bool = False


class VeniceSpeechOptions(SpeechParams, SpeechOutputOptions):
    """Speech endpoint parameters plus plugin output policy and defaults."""

    response_format: Optional[AudioResponseFormat] = "mp3"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)


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
        "output_paths": [str(output_path)],
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


def _publish_media_file(part_path, output_path, *, overwrite):
    candidate = output_path
    for _ in range(5):
        try:
            outcome = _files.publish_file(part_path, candidate, overwrite=overwrite)
        except FileExistsError:
            if overwrite:
                raise
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            candidate = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
            continue
        if outcome.cleanup_error is not None:
            with suppress(OSError, ValueError):
                print(
                    f"Warning: audio saved to {outcome.path}; could not remove staging file "
                    f"{outcome.staging_path}: {outcome.cleanup_error}",
                    file=sys.stderr,
                )
        return outcome
    raise FileExistsError(f"could not publish audio without overwriting a file at {output_path}")


def _partial_notice(error):
    path = getattr(error, "partial_path", None)
    if path is None:
        return
    with suppress(OSError, ValueError):
        try:
            size = f" ({_format_bytes(path.stat().st_size)})"
        except OSError:
            size = " (size unavailable)"
        print(f"Warning: download interrupted; partial audio kept at {path}{size}", file=sys.stderr)


def write_audio_file(chunks, output_path, reporter, *, overwrite=False):
    """Apply adapter collision policy to the shared atomic writer."""

    def progress_chunks():
        for chunk in chunks:
            yield chunk
            reporter.advance(len(chunk))

    try:
        outcome = _files.write_chunks(
            progress_chunks(), output_path, overwrite=overwrite, publish=_publish_media_file
        )
    except BaseException as exc:
        _partial_notice(exc)
        raise
    reporter.finish()
    return outcome.path


async def write_async_audio_file(chunks, output_path, reporter, *, overwrite=False):
    async def progress_chunks():
        async for chunk in chunks:
            yield chunk
            reporter.advance(len(chunk))

    try:
        outcome = await _files.write_async_chunks(
            progress_chunks(), output_path, overwrite=overwrite, publish=_publish_media_file
        )
    except BaseException as exc:
        _partial_notice(exc)
        raise
    reporter.finish()
    return outcome.path


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


def _speech_options(model, prompt):
    warn_deprecated(model)
    opts = cast(VeniceSpeechOptions, model.Options.model_validate(prompt.options.model_dump()))
    validate_output_directory(opts.output_dir)
    params = copy_params(SpeechParams, opts)
    return opts, params


def speech_output_path(model_name, opts, format):
    return _resolve_speech_output_path(
        model_name=model_name,
        response_format=format,
        output_dir=opts.output_dir,
        output_filename=opts.output_filename,
        overwrite_files=opts.overwrite_files,
    )


class VeniceSpeech(llm.KeyModel):
    """Venice speech adapter, saving generated bytes under llm's output policy."""

    can_stream = True
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    model_name: str
    model_spec: dict

    def __init__(self, model_id, model_name=None, model_spec=None):
        initialize_model(self, model_id, model_name, model_spec)

    def __str__(self):
        return f"Venice Speech: {self.model_id}"

    class Options(VeniceSpeechOptions):  # type: ignore[reportIncompatibleVariableOverride]
        pass

    def execute(self, prompt, stream, response, conversation=None, key=None):
        with model_error_boundary():
            opts, params = _speech_options(self, prompt)
            use_stream = stream or opts.progress
            reporter = DownloadProgress(opts.progress)
            with VeniceClient(key=require_model_key(self, key)) as client:
                if use_stream:
                    with client.stream_speech(self.model_name, prompt.prompt, params) as result:
                        path = speech_output_path(self.model_name, opts, result.format)
                        yield f"Writing audio to {path}\n"
                        saved = write_audio_file(
                            result.iter_bytes(), path, reporter, overwrite=opts.overwrite_files
                        )
                        content_type = result.content_type
                        size = reporter.bytes_written
                else:
                    buffered = client.generate_speech(self.model_name, prompt.prompt, params)
                    path = speech_output_path(self.model_name, opts, buffered.format)
                    saved = write_audio_file(
                        [buffered.data], path, reporter, overwrite=opts.overwrite_files
                    )
                    content_type, size = buffered.content_type, len(buffered.data)
            _set_speech_response_json(
                response,
                payload=speech_body(self.model_name, prompt.prompt, params, streaming=use_stream),
                content_type=content_type,
                bytes_written=size,
                output_path=saved,
            )
            yield f"Audio saved to {saved}"


class AsyncVeniceSpeech(llm.AsyncKeyModel):
    """Venice speech adapter using native async network I/O."""

    can_stream = True
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    model_name: str
    model_spec: dict

    def __init__(self, model_id, model_name=None, model_spec=None):
        initialize_model(self, model_id, model_name, model_spec)

    def __str__(self):
        return f"Venice Speech: {self.model_id}"

    class Options(VeniceSpeechOptions):  # type: ignore[reportIncompatibleVariableOverride]
        pass

    async def execute(self, prompt, stream, response, conversation=None, key=None):
        with model_error_boundary():
            opts, params = _speech_options(self, prompt)
            use_stream = stream or opts.progress
            reporter = DownloadProgress(opts.progress)
            async with AsyncVeniceClient(key=require_model_key(self, key)) as client:
                if use_stream:
                    async with client.stream_speech(
                        self.model_name, prompt.prompt, params
                    ) as result:
                        path = speech_output_path(self.model_name, opts, result.format)
                        yield f"Writing audio to {path}\n"
                        saved = await write_async_audio_file(
                            result.aiter_bytes(), path, reporter, overwrite=opts.overwrite_files
                        )
                        content_type = result.content_type
                        size = reporter.bytes_written
                else:
                    buffered = await client.generate_speech(self.model_name, prompt.prompt, params)
                    path = speech_output_path(self.model_name, opts, buffered.format)
                    saved = await _files.async_file_call(
                        write_audio_file,
                        [buffered.data],
                        path,
                        reporter,
                        overwrite=opts.overwrite_files,
                    )
                    content_type, size = buffered.content_type, len(buffered.data)
            _set_speech_response_json(
                response,
                payload=speech_body(self.model_name, prompt.prompt, params, streaming=use_stream),
                content_type=content_type,
                bytes_written=size,
                output_path=saved,
            )
            yield f"Audio saved to {saved}"
