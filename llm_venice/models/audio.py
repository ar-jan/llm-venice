"""Venice audio/text-to-speech model implementation."""

from contextlib import suppress
import pathlib
import sys
import time
from typing import Literal, Optional, Union

import llm
from llm_venice._options import CatalogOptionsMixin, normalize_policy, validate_request
from pydantic import Field
from typing import cast
from ._common import initialize_model, warn_deprecated, model_error_boundary
from llm_venice.client import VeniceClient, AsyncVeniceClient
from llm_venice.params import SpeechParams, copy_params
from llm_venice._endpoints import speech_body
from llm_venice import _files

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


class _SpeechOptions(SpeechParams, SpeechOutputOptions):
    """Speech endpoint parameters plus plugin output policy and defaults."""

    response_format: Optional[AudioResponseFormat] = "mp3"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)


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


def _format_bytes(num_bytes: int) -> str:
    """Format bytes as a human-friendly string."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f}{unit}"
        value /= 1024.0
    return f"{value:.0f}TB"


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


def _partial_notice(error):
    published = getattr(error, "published_path", None)
    if published is not None:
        with suppress(OSError, ValueError):
            print(f"Warning: audio saved to {published} before cancellation", file=sys.stderr)
        return
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
            progress_chunks(), output_path, overwrite=overwrite, publish=_files.publish_media_file
        )
    except BaseException as exc:
        _partial_notice(exc)
        raise
    reporter.finish()
    return outcome.path


async def _single_chunk(data):
    yield data


async def write_async_audio_file(chunks, output_path, reporter, *, overwrite=False):
    async def progress_chunks():
        async for chunk in chunks:
            yield chunk
            reporter.advance(len(chunk))

    try:
        outcome = await _files.write_async_chunks(
            progress_chunks(), output_path, overwrite=overwrite, publish=_files.publish_media_file
        )
    except BaseException as exc:
        _partial_notice(exc)
        raise
    reporter.finish()
    return outcome.path


def _speech_options(model, prompt):
    warn_deprecated(model)
    opts = cast(_SpeechOptions, model.Options.model_validate(prompt.options.model_dump()))
    validate_output_directory(opts.output_dir)
    validate_request(model.catalog_policy, prompt.prompt, opts)
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


class VeniceSpeech(CatalogOptionsMixin, llm.KeyModel):
    """Venice speech adapter, saving generated bytes under llm's output policy."""

    can_stream = True
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    model_name: str
    model_spec: dict

    def __init__(self, model_id, model_name=None, model_spec=None):
        initialize_model(self, model_id, model_name, model_spec)
        self.catalog_policy = normalize_policy("tts", model_spec)

    def __str__(self):
        return f"Venice Speech: {self.model_id}"

    class Options(_SpeechOptions):  # type: ignore[reportIncompatibleVariableOverride]
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


class AsyncVeniceSpeech(CatalogOptionsMixin, llm.AsyncKeyModel):
    """Venice speech adapter using native async network I/O."""

    can_stream = True
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    model_name: str
    model_spec: dict

    def __init__(self, model_id, model_name=None, model_spec=None):
        initialize_model(self, model_id, model_name, model_spec)
        self.catalog_policy = normalize_policy("tts", model_spec)

    def __str__(self):
        return f"Venice Speech: {self.model_id}"

    Options: type[VeniceSpeech.Options] = VeniceSpeech.Options  # type: ignore[reportIncompatibleVariableOverride]

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
                    saved = await write_async_audio_file(
                        _single_chunk(buffered.data),
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
