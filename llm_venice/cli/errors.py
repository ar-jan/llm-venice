"""Shared CLI error helpers."""

import click


def handle_cli_error(error: Exception):
    """Convert known errors into a ClickException with a clean message."""
    raise click.ClickException(str(error)) from error
