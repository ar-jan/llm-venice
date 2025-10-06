"""Utility functions for the LLM Venice plugin."""

import datetime
import os
import pathlib
from typing import Optional, Union

import click
import llm

from llm_venice.constants import VENICE_KEY_NAME, VENICE_KEY_ENV_VAR


def get_venice_key() -> str:
    """
    Get the Venice API key from LLM's key management.

    Raises:
        click.ClickException: If no key is found.

    Returns:
        The Venice API key.
    """
    key = llm.get_key("", VENICE_KEY_NAME, VENICE_KEY_ENV_VAR)
    if not key:
        raise click.ClickException("No key found for Venice")
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
) -> Optional[pathlib.Path]:
    """
    Validate that a directory exists and is writable.

    Args:
        output_dir: Directory path to validate, or None

    Returns:
        Resolved path if valid, None if input was None

    Raises:
        ValueError: If directory is not writable
    """
    if output_dir is None:
        return None

    if isinstance(output_dir, str) and not output_dir.strip():
        # Treat empty string the same as no directory provided
        return None

    resolved_dir = pathlib.Path(output_dir)
    if (
        not resolved_dir.exists()
        or not resolved_dir.is_dir()
        or not os.access(resolved_dir, os.W_OK)
    ):
        raise ValueError(f"output_dir {resolved_dir} is not a writable directory")

    return resolved_dir
