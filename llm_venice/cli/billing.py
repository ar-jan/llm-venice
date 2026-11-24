"""Billing commands for Venice CLI."""

from collections.abc import Iterable
from datetime import timezone
import json
import re
from typing import Any, get_args

import click
from pydantic import ValidationError

from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceError
from llm_venice._notices import VeniceNotice
from llm_venice.cli.errors import handle_cli_error, usage_error
from llm_venice.cli.notices import emit_cli_notices
from llm_venice.types import BillingCurrency

TIMESTAMP_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"]
# Usage history reports no total, so stop after this many pages even if more remain.
MAX_USAGE_PAGES = 1000
_FLAGS = {
    "start_timestamp": "--start",
    "end_timestamp": "--end",
    "page_size": "--page-size",
    "start_date": "--start-date",
    "end_date": "--end-date",
    "lookback": "--lookback",
}


def _utc(value):
    """Read a naive command-line time as UTC, which Venice's billing uses."""
    return value.replace(tzinfo=timezone.utc) if value is not None else None


def _walk(fetch, first: dict[str, Any], *, all_pages: bool):
    """Yield pages from ``fetch(**query)``, following cursors when ``all_pages``.

    Stops with a warning when a cursor repeats or after MAX_USAGE_PAGES pages.
    """
    query = first
    seen: set[str] = set()
    for _ in range(MAX_USAGE_PAGES):
        page, cursor = fetch(**query)
        yield page
        if not all_pages or cursor is None:
            if cursor is not None:
                click.echo(f"More entries remain; continue with --cursor {cursor}", err=True)
            return
        if cursor in seen:
            emit_cli_notices(
                [VeniceNotice("warning", "Venice repeated a cursor; the history may be incomplete")]
            )
            return
        seen.add(cursor)
        query = {"cursor": cursor}
    emit_cli_notices(
        [VeniceNotice("warning", f"stopped after {MAX_USAGE_PAGES} pages; more entries remain")]
    )


def _write_csv_pages(pages: Iterable[str]) -> None:
    """Keep one header and separate records even when a page omits its final newline."""
    header = None
    separator = "\n"
    needs_separator = False
    for text in pages:
        if not text:
            continue
        first, *remainder = re.split(r"(\r\n|\r|\n)", text, maxsplit=1)
        if header is None:
            header = first
            if remainder:
                separator = remainder[0]
        elif first == header:
            text = remainder[1] if remainder else ""
        if text:
            if needs_separator:
                click.echo(separator, nl=False)
            click.echo(text, nl=False)
            needs_separator = not text.endswith(("\r", "\n"))


def create_billing_group():
    """
    Create the billing command group.

    Returns:
        Click group for balance, usage history and usage analytics
    """

    @click.group(name="billing")
    def billing():
        """Show balances, usage history and usage analytics (requires an ADMIN key)"""

    @billing.command(name="balance")
    def balance():
        """Show whether the account can make requests, and its balances."""
        try:
            with VeniceClient() as client:
                response = client.get_billing_balance()
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        click.echo(json.dumps(response, indent=2))

    @billing.command(name="history")
    @click.option(
        "--currency", type=click.Choice(get_args(BillingCurrency)), help="Only this currency"
    )
    @click.option(
        "--start",
        type=click.DateTime(formats=TIMESTAMP_FORMATS),
        help="Earliest entry time, inclusive (UTC)",
    )
    @click.option(
        "--end",
        type=click.DateTime(formats=TIMESTAMP_FORMATS),
        help="Latest entry time, exclusive (UTC)",
    )
    @click.option(
        "--page-size",
        type=click.IntRange(10, 1000),
        help="Entries per page (Venice's default: 1000)",
    )
    @click.option("--cursor", help="Continue from a previous page's cursor, without filters")
    @click.option("--all-pages", is_flag=True, help="Follow cursors to the end of the history")
    @click.option("--csv", "as_csv", is_flag=True, help="Output CSV instead of JSON")
    def history(currency, start, end, page_size, cursor, all_pages, as_csv):
        """Show usage history, oldest first, one page at a time unless --all-pages.

        Filters apply to the first page; a cursor carries them to the next, so
        --cursor cannot be combined with them. When more entries remain, the cursor
        to continue with is printed to stderr. With --all-pages, JSON output merges
        the entries of every page, and CSV output keeps only the first header row.
        """
        first = {
            "currency": currency,
            "start_timestamp": _utc(start),
            "end_timestamp": _utc(end),
            "page_size": page_size,
            "cursor": cursor,
        }
        try:
            with VeniceClient() as client:
                if as_csv:

                    def fetch_csv(**query):
                        page = client.export_usage_history(**query)
                        return page.text, page.next_cursor

                    _write_csv_pages(_walk(fetch_csv, first, all_pages=all_pages))
                    return

                def fetch_json(**query):
                    page = client.list_usage_history(**query)
                    return page, page["nextCursor"]

                pages = list(_walk(fetch_json, first, all_pages=all_pages))
        except ValidationError as exc:
            raise usage_error(exc, _FLAGS) from exc
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        if all_pages:
            entries = [entry for page in pages for entry in page["data"]]
            output = {"data": entries, "nextCursor": pages[-1]["nextCursor"]}
        else:
            output = pages[0]
        click.echo(json.dumps(output, indent=2))

    @billing.command(name="analytics")
    @click.option("--lookback", help="Days to look back, such as 30d (at most 90d; default 7d)")
    @click.option(
        "--start-date", type=click.DateTime(formats=["%Y-%m-%d"]), help="First day, with --end-date"
    )
    @click.option(
        "--end-date", type=click.DateTime(formats=["%Y-%m-%d"]), help="Last day, with --start-date"
    )
    def analytics(lookback, start_date, end_date):
        """Show usage totals by day, model and API key (a Venice beta endpoint)."""
        try:
            with VeniceClient() as client:
                response = client.get_usage_analytics(
                    lookback=lookback,
                    start_date=start_date.date() if start_date else None,
                    end_date=end_date.date() if end_date else None,
                )
        except ValidationError as exc:
            raise usage_error(exc, _FLAGS) from exc
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        click.echo(json.dumps(response, indent=2))

    return billing
