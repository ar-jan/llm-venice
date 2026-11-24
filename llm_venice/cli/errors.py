"""Shared CLI error helpers."""

from collections.abc import Mapping
import re
from typing import NoReturn

import click
from pydantic import ValidationError


def handle_cli_error(error: Exception) -> NoReturn:
    """Convert known errors into a ClickException with a clean message."""
    raise click.ClickException(str(error)) from error


def usage_error(exc: ValidationError, flags: Mapping[str, str] = {}) -> click.UsageError:
    """Name each invalid field by its flag (``flags``, else ``--field-name``).

    A check across several options has no field, so its message is shown with the
    field names in ``flags`` replaced by their flags.
    """
    messages = []
    for error in exc.errors():
        if not error["loc"]:
            message = str(error.get("ctx", {}).get("error") or error["msg"])
            for field, flag in flags.items():
                message = re.sub(rf"\b{re.escape(field)}\b", flag, message)
            messages.append(message)
            continue
        field = str(error["loc"][0])
        flag = flags.get(field, f"--{field.replace('_', '-')}")
        messages.append(f"Invalid value for {flag}: {error['msg']}")
    return click.UsageError("\n".join(messages))
