"""Venice audio/text-to-speech model implementation."""

import asyncio
from contextlib import contextmanager
import os
import pathlib
from dataclasses import dataclass
import sys
import time
from typing import AsyncGenerator, Iterable, Iterator, Literal, Optional, Union

import httpx
import llm
from llm.utils import logging_client
from pydantic import Field

from llm_venice.api.client import get_auth_headers_with_content_type
from llm_venice.api.errors import VeniceAPIError, raise_api_error
from llm_venice.constants import ENDPOINT_AUDIO_SPEECH
from llm_venice.utils import (
    generate_timestamp_filename,
    get_unique_filepath,
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
    stdout: Optional[bool] = Field(
        description=(
            "Write audio bytes to stdout. When set, progress and status messages are written to stderr."
        ),
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
    output_path: Optional[pathlib.Path]
    write_stdout: bool
    progress: bool
    use_streaming_download: bool


def _clean_output_dir(
    output_dir: Optional[Union[pathlib.Path, str]],
) -> Optional[Union[pathlib.Path, str]]:
    if isinstance(output_dir, str) and not output_dir.strip():
        return None
    return output_dir


def _clean_output_filename(output_filename: Optional[str]) -> Optional[str]:
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
    resolved_output_dir = validate_output_directory(_clean_output_dir(output_dir))
    target_dir = resolved_output_dir or (llm.user_dir() / "audio")
    cleaned_filename = _clean_output_filename(output_filename)
    if not cleaned_filename:
        cleaned_filename = generate_timestamp_filename("venice", model_name, response_format)
    return get_unique_filepath(target_dir, cleaned_filename, overwrite_files)


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

    Removes CLI-only options (stdout/progress) and output-path options from the payload.
    Does not write anything to disk.
    """
    output_dir = options_dict.pop("output_dir", None)
    output_filename = options_dict.pop("output_filename", None)
    overwrite_files = options_dict.pop("overwrite_files", False)
    options_dict.pop("progress", None)
    options_dict.pop("stdout", None)

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

    write_stdout = bool(options_dict.get("stdout", False))
    progress = bool(options_dict.get("progress", False))

    requested_file_output = (
        _clean_output_dir(options_dict.get("output_dir")) is not None
        or _clean_output_filename(options_dict.get("output_filename")) is not None
    )
    save_to_file = (not write_stdout) or requested_file_output

    payload, resolved_output_path = _prepare_speech_generation(
        input_text=input_text, options_dict=options_dict, model_name=model_name
    )

    payload["streaming"] = stream
    use_streaming_download = write_stdout or progress or stream
    return _SpeechCLIRequest(
        payload=payload,
        output_path=resolved_output_path if save_to_file else None,
        write_stdout=write_stdout,
        progress=progress,
        use_streaming_download=use_streaming_download,
    )


def _set_speech_response_json(
    response: Union[llm.Response, llm.AsyncResponse],
    *,
    payload: dict,
    content_type: Optional[str],
    bytes_written: int,
    output_path: Optional[pathlib.Path],
    stdout: bool,
) -> None:
    response.response_json = {
        "request": payload,
        "content_type": content_type,
        "bytes_written": bytes_written,
        "output_path": str(output_path) if output_path else None,
        "stdout": stdout,
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
        with httpx.stream(
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
        except httpx.HTTPStatusError as exc:
            raise_api_error("Generating speech", exc)
        yield r.headers.get("Content-Type"), r.iter_bytes()


def _stream_speech_to_outputs(
    *,
    api_key: str,
    payload: dict,
    output_path: Optional[pathlib.Path],
    write_stdout: bool,
    progress: bool,
) -> tuple[Optional[str], int]:
    """Download speech audio and write to file and/or stdout."""
    bytes_written = 0
    start_time = time.monotonic()
    last_update = start_time
    write_stdout_enabled = write_stdout

    def maybe_update_progress(*, force: bool = False) -> None:
        nonlocal last_update
        if not progress:
            return
        now = time.monotonic()
        if not force and (now - last_update) < 0.5:
            return
        elapsed = max(now - start_time, 0.001)
        rate = bytes_written / elapsed
        message = f"Downloaded {_format_bytes(bytes_written)} ({_format_bytes(int(rate))}/s)"
        if sys.stderr.isatty():
            sys.stderr.write(f"\r{message}")
        else:
            sys.stderr.write(f"{message}\n")
        sys.stderr.flush()
        last_update = now

    file_handle = None
    content_type: Optional[str] = None

    try:
        with _stream_speech_bytes(api_key=api_key, payload=payload) as (content_type, chunks):
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                file_handle = open(output_path, "wb")

            for chunk in chunks:
                if not chunk:
                    continue
                bytes_written += len(chunk)
                if file_handle is not None:
                    file_handle.write(chunk)
                if write_stdout_enabled:
                    try:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                    except BrokenPipeError:
                        if output_path is None:
                            break
                        write_stdout_enabled = False
                maybe_update_progress()
    finally:
        if file_handle is not None:
            file_handle.close()

    maybe_update_progress(force=True)
    if progress and sys.stderr.isatty():
        sys.stderr.write("\n")
        sys.stderr.flush()

    return content_type, bytes_written


def _post_speech_bytes(*, api_key: str, payload: dict) -> tuple[bytes, Optional[str]]:
    """POST /audio/speech and return the full audio response bytes."""
    headers = get_auth_headers_with_content_type(api_key)

    if os.environ.get("LLM_VENICE_SHOW_RESPONSES"):
        client = logging_client()
        r = client.post(ENDPOINT_AUDIO_SPEECH, headers=headers, json=payload, timeout=120)
    else:
        r = httpx.post(ENDPOINT_AUDIO_SPEECH, headers=headers, json=payload, timeout=120)

    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise_api_error("Generating speech", exc)

    return r.content, r.headers.get("Content-Type")


def _write_audio_bytes(*, output_path: pathlib.Path, audio_bytes: bytes) -> None:
    """Write audio bytes to disk, ensuring the destination directory exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)


class VeniceSpeech(llm.KeyModel):
    """Venice AI text-to-speech model."""

    can_stream = True
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"

    def __init__(self, model_id, model_name=None):
        self.model_id = f"venice/{model_id}"
        self.model_name = model_id

    def __str__(self):
        return f"Venice Speech: {self.model_id}"

    class Options(VeniceSpeechOptions):  # type: ignore[override]
        pass

    def execute(self, prompt, stream, response, conversation=None, key=None):
        """Execute text-to-speech request."""
        api_key = self.get_key(key)
        if api_key is None:
            raise llm.NeedsKeyException("No key found for Venice")

        try:
            options_dict = prompt.options.model_dump(by_alias=True)
            validate_output_directory(
                _clean_output_dir(options_dict.get("output_dir")),
                create_if_missing=True,
            )
            request = _prepare_speech_cli_request(
                input_text=prompt.prompt,
                options=prompt.options,
                model_name=self.model_name,
                stream=stream,
            )

            if request.use_streaming_download:
                if request.output_path is not None and not request.write_stdout:
                    yield f"Writing audio to {request.output_path}\n"
                elif request.output_path is not None and request.write_stdout:
                    sys.stderr.write(f"Saving a copy to {request.output_path}\n")
                    sys.stderr.flush()

                content_type, bytes_written = _stream_speech_to_outputs(
                    api_key=api_key,
                    payload=request.payload,
                    output_path=request.output_path,
                    write_stdout=request.write_stdout,
                    progress=request.progress,
                )

                _set_speech_response_json(
                    response,
                    payload=request.payload,
                    content_type=content_type,
                    bytes_written=bytes_written,
                    output_path=request.output_path,
                    stdout=request.write_stdout,
                )

                if request.write_stdout:
                    if request.output_path is not None:
                        sys.stderr.write(f"Audio saved to {request.output_path}\n")
                        sys.stderr.flush()
                    return

                if request.output_path is None:
                    raise llm.ModelError("No output path available to save audio")
                yield f"Audio saved to {request.output_path}"
                return

            if request.output_path is None:
                raise llm.ModelError("No output path available to save audio")

            audio_bytes, content_type = _post_speech_bytes(api_key=api_key, payload=request.payload)
            _write_audio_bytes(output_path=request.output_path, audio_bytes=audio_bytes)

            _set_speech_response_json(
                response,
                payload=request.payload,
                content_type=content_type,
                bytes_written=len(audio_bytes),
                output_path=request.output_path,
                stdout=False,
            )

            yield f"Audio saved to {request.output_path}"
        except (OSError, ValueError, VeniceAPIError) as exc:
            raise llm.ModelError(str(exc)) from exc


class AsyncVeniceSpeech(llm.AsyncKeyModel):
    """Asynchronous Venice AI text-to-speech model."""

    can_stream = True
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"

    def __init__(self, model_id, model_name=None):
        self.model_id = f"venice/{model_id}"
        self.model_name = model_id

    def __str__(self):
        return f"Venice Speech: {self.model_id}"

    class Options(VeniceSpeechOptions):  # type: ignore[override]
        pass

    async def execute(
        self, prompt, stream, response, conversation=None, key=None
    ) -> AsyncGenerator[str, None]:
        """Execute text-to-speech request asynchronously."""
        api_key = self.get_key(key)
        if api_key is None:
            raise llm.NeedsKeyException("No key found for Venice")

        try:
            options_dict = prompt.options.model_dump(by_alias=True)
            validate_output_directory(
                _clean_output_dir(options_dict.get("output_dir")),
                create_if_missing=True,
            )
            request = _prepare_speech_cli_request(
                input_text=prompt.prompt,
                options=prompt.options,
                model_name=self.model_name,
                stream=stream,
            )

            if request.use_streaming_download:
                if request.output_path is not None and not request.write_stdout:
                    yield f"Writing audio to {request.output_path}\n"
                elif request.output_path is not None and request.write_stdout:
                    sys.stderr.write(f"Saving a copy to {request.output_path}\n")
                    sys.stderr.flush()

                content_type, bytes_written = await asyncio.to_thread(
                    _stream_speech_to_outputs,
                    api_key=api_key,
                    payload=request.payload,
                    output_path=request.output_path,
                    write_stdout=request.write_stdout,
                    progress=request.progress,
                )

                _set_speech_response_json(
                    response,
                    payload=request.payload,
                    content_type=content_type,
                    bytes_written=bytes_written,
                    output_path=request.output_path,
                    stdout=request.write_stdout,
                )

                if request.write_stdout:
                    if request.output_path is not None:
                        sys.stderr.write(f"Audio saved to {request.output_path}\n")
                        sys.stderr.flush()
                    return

                if request.output_path is None:
                    raise llm.ModelError("No output path available to save audio")
                yield f"Audio saved to {request.output_path}"
                return

            if request.output_path is None:
                raise llm.ModelError("No output path available to save audio")

            audio_bytes, content_type = await asyncio.to_thread(
                _post_speech_bytes, api_key=api_key, payload=request.payload
            )
            await asyncio.to_thread(
                _write_audio_bytes, output_path=request.output_path, audio_bytes=audio_bytes
            )

            _set_speech_response_json(
                response,
                payload=request.payload,
                content_type=content_type,
                bytes_written=len(audio_bytes),
                output_path=request.output_path,
                stdout=False,
            )

            yield f"Audio saved to {request.output_path}"
        except (OSError, ValueError, VeniceAPIError) as exc:
            raise llm.ModelError(str(exc)) from exc
