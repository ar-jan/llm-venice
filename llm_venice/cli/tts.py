"""Text-to-speech command for Venice CLI."""

import os
import pathlib
import sys
from typing import BinaryIO, Iterable, Optional, get_args

import click
from pydantic import ValidationError

from llm_venice.errors import VeniceError
from llm_venice.client import VeniceClient
from llm_venice.params import SpeechParams, copy_params
from llm_venice._options import validate_request
from llm_venice.catalog import catalog_path
from llm_venice.cli.errors import handle_cli_error
from llm_venice.models import _load_or_fetch_models
from llm_venice.models.audio import (
    AudioResponseFormat,
    DownloadProgress,
    VeniceSpeech,
    speech_output_path,
    write_audio_file,
)

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
    model: VeniceSpeech,
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
        return model.Options(**{k: v for k, v in values.items() if v is not None})
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
    """Split -o PATH into (output_dir, output_filename).

    A trailing separator means the path is a directory even when it does not
    exist yet, since Path() drops it and is_dir() would then say file.
    """
    if output is None:
        return None, None
    directory_intent = output.endswith(("/", os.sep))
    path = pathlib.Path(output).expanduser()
    if directory_intent or path.is_dir():
        return path, None
    return path.parent, path.name


def _resolve_model(requested: Optional[str]) -> VeniceSpeech:
    """Return the API model ID, defaulting to the first tts model in the catalog."""
    catalog = [m for m in (_load_or_fetch_models(catalog_path()) or []) if isinstance(m, dict)]

    if requested:
        model_id = requested.removeprefix("venice/")
        entry = next((m for m in catalog if m.get("id") == model_id), None)
        if entry is not None and entry.get("type") != "tts":
            raise click.UsageError(
                f"Model '{model_id}' is a {entry.get('type')} model, not a text-to-speech model."
            )
        return VeniceSpeech(model_id, model_spec=entry.get("model_spec") if entry else None)

    for entry in catalog:
        if entry.get("type") == "tts" and isinstance(entry.get("id"), str):
            if entry.get("model_spec", {}).get("offline") is True:
                continue
            return VeniceSpeech(entry["id"], model_spec=entry.get("model_spec"))

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
        resolved_model = _resolve_model(model)
        options = _build_options(
            model=resolved_model,
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
        model_id = resolved_model.model_name

        reporter = DownloadProgress(progress)
        try:
            validate_request(resolved_model.catalog_policy, input_text, options)
            with (
                VeniceClient() as client,
                client.stream_speech(
                    model_id, input_text, copy_params(SpeechParams, options)
                ) as result,
            ):
                if to_stdout:
                    _write_to_stdout(result.iter_bytes(), reporter)
                    return
                path = speech_output_path(model_id, options, result.format)
                saved_path = write_audio_file(
                    result.iter_bytes(), path, reporter, overwrite=overwrite
                )
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        click.echo(str(saved_path))

    return tts
