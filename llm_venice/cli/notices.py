"""CLI helpers for non-fatal notices."""

from typing import Sequence

import click

from llm_venice.notices import VeniceNotice, render_notice


def emit_cli_notices(notices: Sequence[VeniceNotice]) -> None:
    """Emit non-fatal notices to stderr via Click."""
    for notice in notices:
        click.echo(render_notice(notice), err=True)
