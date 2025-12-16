"""Utility functions for the LLM Venice plugin."""

import datetime
import os
import pathlib
import sys
from typing import Optional, Union

import click
import llm
from llm import NeedsKeyException


def get_venice_key(explicit_key: Optional[str] = None, *, click_exceptions: bool = False) -> str:
    """
    Get the Venice API key from LLM's key management.

    Raises:
        NeedsKeyException by default when no key is found; set click_exceptions=True
        for CLI use to raise click.ClickException instead.

    Returns:
        The Venice API key.
    """
    key = llm.get_key(explicit_key, "venice", "LLM_VENICE_KEY")
    if not key:
        message = "No key found for Venice"
        if click_exceptions:
            raise click.ClickException(message)
        raise NeedsKeyException(message)
    return key


def generate_timestamp_filename(prefix: str, model_name: str, extension: str) -> str:
    """
    Generate a timestamped filename.

    Args:
        prefix: Prefix for the filename
        model_name: Name of the model
        extension: File extension (without dot)

    Returns:
        Formatted filename with timestamp
    """
    datestring = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return f"{datestring}_{prefix}_{model_name}.{extension}"


def get_unique_filepath(
    directory: pathlib.Path, filename: str, overwrite: bool = False
) -> pathlib.Path:
    """
    Get a unique filepath, adding timestamp if file exists and overwrite is False.

    Args:
        directory: Directory for the file
        filename: Desired filename
        overwrite: Whether to allow overwriting existing files

    Returns:
        A unique filepath
    """
    filepath = directory / filename

    if not filepath.exists() or overwrite:
        return filepath

    # Add timestamp to make unique
    stem = filepath.stem
    suffix = filepath.suffix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    new_filename = f"{stem}_{timestamp}{suffix}"
    return directory / new_filename


def validate_output_directory(
    output_dir: Optional[Union[pathlib.Path, str]],
    *,
    create_if_missing: bool = False,
) -> Optional[pathlib.Path]:
    """
    Validate that an output directory is writable.

    If the directory does not exist, this validates that it can be created.
    When running under an interactive click CLI, set create_if_missing=True to
    prompt the user to create the directory.

    Args:
        output_dir: Directory path to validate, or None

    Returns:
        Resolved path if valid, None if input was None

    Raises:
        ValueError: If directory is not writable or cannot be created
    """
    if output_dir is None:
        return None

    if isinstance(output_dir, str) and not output_dir.strip():
        # Treat empty string the same as no directory provided
        return None

    resolved_dir = pathlib.Path(output_dir).expanduser()
    if resolved_dir.exists():
        if not resolved_dir.is_dir() or not os.access(resolved_dir, os.W_OK):
            raise ValueError(f"output_dir {resolved_dir} is not a writable directory")
        return resolved_dir

    def can_create_directory(path: pathlib.Path) -> bool:
        parent = path
        while not parent.exists():
            if parent.parent == parent:
                return False
            parent = parent.parent

        if not parent.is_dir():
            return False

        # On POSIX, directory creation requires write + execute.
        return os.access(parent, os.W_OK | os.X_OK)

    if (
        create_if_missing
        and click.get_current_context(silent=True) is not None
        and sys.stdin.isatty()
    ):
        should_create = click.confirm(
            f"Output directory {resolved_dir} does not exist. Create it?",
            default=True,
            err=True,
        )
        if not should_create:
            raise ValueError(f"output_dir {resolved_dir} is not a writable directory")

        try:
            resolved_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"output_dir {resolved_dir} is not a writable directory") from exc

        if not resolved_dir.is_dir() or not os.access(resolved_dir, os.W_OK):
            raise ValueError(f"output_dir {resolved_dir} is not a writable directory")
        return resolved_dir

    if not can_create_directory(resolved_dir):
        raise ValueError(f"output_dir {resolved_dir} is not a writable directory")

    return resolved_dir
