"""Text-to-speech command for Venice CLI."""

import os
import pathlib
import sys
from typing import BinaryIO, Iterable, Optional, get_args

import click
import httpx2
from pydantic import ValidationError

from llm_venice.api.errors import VeniceAPIError
from llm_venice.api.refresh import default_models_path
from llm_venice.cli.errors import handle_cli_error
from llm_venice.models import _load_or_fetch_models
from llm_venice.models.audio import (
    AudioResponseFormat,
    DownloadProgress,
    VeniceSpeech,
    stream_speech_result,
)
from llm_venice.utils import get_venice_key

# Maps VeniceSpeech.Options fields to the flags that set them, for error messages.
_OPTION_FLAGS = {
    "voice": "--voice",
    "response_format": "--format",
    "speed": "--speed",
}


def _stdout_is_tty() -> bool:
    return sys.stdout.isatty()


def _stdin_is_tty() -> bool:
    return sys.stdin.isatty()


def _binary_stdout() -> BinaryIO:
    # click.get_binary_stream() is deprecated since click 8.5.
    return sys.stdout.buffer


def _read_text(text: Optional[str]) -> str:
    """Return the text to speak, from the argument or from stdin."""
    if text is None and _stdin_is_tty():
        raise click.UsageError("Provide TEXT, or pipe text on stdin.")
    raw = sys.stdin.read() if text is None or text == "-" else text
    stripped = raw.strip()
    if not stripped:
        raise click.UsageError("No text to speak.")
    return stripped


def _build_options(
    *,
    voice: Optional[str],
    response_format: Optional[str],
    speed: Optional[float],
    output_dir: Optional[pathlib.Path],
    output_filename: Optional[str],
    overwrite: bool,
) -> VeniceSpeech.Options:
    """Validate the speech options with the same model as `llm -m venice/<tts-model>`."""
    values = {
        "voice": voice,
        "response_format": response_format,
        "speed": speed,
        "output_dir": output_dir,
        "output_filename": output_filename,
        "overwrite_files": overwrite,
    }
    try:
        return VeniceSpeech.Options(**{k: v for k, v in values.items() if v is not None})
    except ValidationError as exc:
        messages = []
        for error in exc.errors():
            field = str(error["loc"][0]) if error["loc"] else ""
            flag = _OPTION_FLAGS.get(field, field)
            messages.append(f"Invalid value for {flag}: {error['msg']}")
        raise click.UsageError("\n".join(messages)) from exc


def _split_output_path(
    output: Optional[str],
) -> tuple[Optional[pathlib.Path], Optional[str]]:
    """Split -o PATH into (output_dir, output_filename)."""
    if output is None:
        return None, None
    path = pathlib.Path(output).expanduser()
    if path.is_dir():
        return path, None
    return path.parent, path.name


def _resolve_model(requested: Optional[str]) -> str:
    """Return the API model ID, defaulting to the first tts model in the catalog."""
    catalog = [
        m for m in (_load_or_fetch_models(default_models_path()) or []) if isinstance(m, dict)
    ]

    if requested:
        model_id = requested.removeprefix("venice/")
        entry = next((m for m in catalog if m.get("id") == model_id), None)
        if entry is not None and entry.get("type") != "tts":
            raise click.UsageError(
                f"Model '{model_id}' is a {entry.get('type')} model, not a text-to-speech model."
            )
        return model_id

    for entry in catalog:
        if entry.get("type") == "tts" and isinstance(entry.get("id"), str):
            return entry["id"]

    if not catalog:
        raise click.ClickException(
            "No Venice model catalog available to pick a default model. "
            "Run 'llm venice refresh', or pass -m MODEL."
        )
    raise click.ClickException(
        "The Venice model catalog lists no text-to-speech models. "
        "Run 'llm venice refresh', or pass -m MODEL."
    )


def _silence_stdout() -> None:
    """Point stdout at devnull so the interpreter's final flush cannot fail again."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError):
        # Not a real file descriptor (e.g. under click's CliRunner).
        pass


def _write_to_stdout(chunks: Iterable[bytes], reporter: DownloadProgress) -> None:
    """Write raw audio bytes to stdout; stop quietly if the reader goes away."""
    out = _binary_stdout()
    try:
        for chunk in chunks:
            if not chunk:
                continue
            out.write(chunk)
            out.flush()
            reporter.advance(len(chunk))
    except BrokenPipeError:
        _silence_stdout()
        raise click.exceptions.Exit(1)
    reporter.finish()


def _write_to_file(
    chunks: Iterable[bytes], output_path: pathlib.Path, reporter: DownloadProgress
) -> None:
    """Write audio bytes to output_path, removing a partial file on failure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "wb") as file_handle:
            for chunk in chunks:
                if not chunk:
                    continue
                file_handle.write(chunk)
                reporter.advance(len(chunk))
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    reporter.finish()


def create_tts_command():
    """
    Create the tts command.

    Returns:
        Click command for text-to-speech
    """

    @click.command(name="tts")
    @click.argument("text", required=False)
    @click.option(
        "-m",
        "--model",
        help=(
            "Text-to-speech model ID, with or without the 'venice/' prefix. "
            "Defaults to the first tts model in the Venice model catalog."
        ),
    )
    @click.option("--voice", help="Voice to use, e.g. af_sky")
    @click.option(
        "--format",
        "response_format",
        metavar="FORMAT",
        help=f"Audio format: {', '.join(get_args(AudioResponseFormat))} (default: mp3)",
    )
    @click.option("--speed", type=float, help="Speech speed from 0.25 to 4.0 (default: 1.0)")
    @click.option(
        "-o",
        "--output",
        metavar="PATH",
        help=(
            "Output file or directory, or '-' to write the audio to stdout. "
            "Unrelated to llm's '-o/--option KEY VALUE'."
        ),
    )
    @click.option("--overwrite", is_flag=True, help="Overwrite an existing output file")
    @click.option("--progress", is_flag=True, help="Show download progress on stderr")
    def tts(text, model, voice, response_format, speed, output, overwrite, progress):
        """Generate speech from TEXT, or from stdin when TEXT is omitted or '-'.

        Without -o the audio is saved under the llm user directory in audio/.
        In file mode the saved path is printed on stdout. With '-o -' only the
        raw audio bytes are written to stdout. This command does not log to
        llm's logs database.

        \b
        Examples:
          llm venice tts "Hello." -o hello.mp3
          llm venice tts "Hello." --format wav -o - > hello.wav
          echo "Hello." | llm venice tts --voice af_sky -o - | mpv -
        """
        to_stdout = output == "-"
        output_dir, output_filename = _split_output_path(None if to_stdout else output)
        options = _build_options(
            voice=voice,
            response_format=response_format,
            speed=speed,
            output_dir=output_dir,
            output_filename=output_filename,
            overwrite=overwrite,
        )
        if to_stdout and _stdout_is_tty():
            raise click.UsageError(
                "Refusing to write binary audio to a terminal. "
                "Redirect stdout (e.g. '> out.mp3') or use -o FILE."
            )
        input_text = _read_text(text)
        key = get_venice_key(click_exceptions=True)
        model_id = _resolve_model(model)

        reporter = DownloadProgress(progress)
        try:
            with stream_speech_result(
                input_text=input_text, options=options, model_name=model_id, api_key=key
            ) as result:
                if to_stdout:
                    _write_to_stdout(result.chunks, reporter)
                    return
                if result.output_path is None:
                    raise ValueError("No output path available to save audio")
                _write_to_file(result.chunks, result.output_path, reporter)
        except (VeniceAPIError, ValueError, OSError, httpx2.RequestError) as e:
            handle_cli_error(e)
        click.echo(str(result.output_path))

    return tts
