"""Shared CLI error helpers."""

from typing import NoReturn

import click


def handle_cli_error(error: Exception) -> NoReturn:
    """Convert known errors into a ClickException with a clean message."""
    raise click.ClickException(str(error)) from error
