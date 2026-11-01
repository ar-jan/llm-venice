"""Search command for Venice CLI."""

import json
import textwrap
import unicodedata
from typing import get_args

import click
from pydantic import ValidationError

from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceError
from llm_venice.cli.errors import handle_cli_error
from llm_venice.types import AugmentSearchProvider, AugmentSearchResponse

_OPTION_FLAGS = {"query": "QUERY", "limit": "--limit", "provider": "--provider"}


def _clean(value: object, *, single_line: bool = False) -> str:
    """Drop terminal control characters from untrusted result text."""
    if not isinstance(value, str):
        return ""
    if single_line:
        value = " ".join(value.split())
    return "".join(
        char for char in value if char in "\n\t" or unicodedata.category(char) != "Cc"
    ).strip()


def _echo_results(response: AugmentSearchResponse) -> None:
    results = response["results"]
    if not results:
        click.echo("No results found.", err=True)
        return
    for index, result in enumerate(results, 1):
        if index > 1:
            click.echo()
        title = _clean(result.get("title"), single_line=True) or "(untitled)"
        click.echo(click.style(f"{index}. {title}", bold=True))
        click.echo(f"   {_clean(result.get('url'), single_line=True)}")
        content = _clean(result.get("content"))
        if content:
            click.echo(textwrap.indent(content, "   "))
        date = _clean(result.get("date"), single_line=True)
        if date:
            click.echo(click.style(f"   {date}", dim=True))


def create_search_command():
    """
    Create the search command.

    Returns:
        Click command for web search without a model
    """

    @click.command(name="search")
    @click.argument("query", nargs=-1, required=True)
    @click.option(
        "-p",
        "--provider",
        type=click.Choice(get_args(AugmentSearchProvider)),
        help="Search provider. Venice uses brave when omitted.",
    )
    @click.option(
        "-n",
        "--limit",
        type=click.IntRange(1, 20),
        help="Maximum number of results. Venice returns up to 10 when omitted.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output the JSON response")
    def search(query, provider, limit, as_json):
        """Search the web with Venice, without a model.

        Uses Venice's /augment/search endpoint. Multiple QUERY words are joined
        with spaces. Each search is billed.

        \b
        Example:
          llm venice search -p exa -n 5 "privacy-preserving AI inference"
        """
        try:
            with VeniceClient() as client:
                response = client.augment_search(" ".join(query), limit=limit, provider=provider)
        except ValidationError as exc:
            messages = []
            for error in exc.errors():
                field = str(error["loc"][0]) if error["loc"] else ""
                messages.append(
                    f"Invalid value for {_OPTION_FLAGS.get(field, field)}: {error['msg']}"
                )
            raise click.UsageError("\n".join(messages)) from exc
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        if as_json:
            click.echo(json.dumps(response, indent=2))
        else:
            _echo_results(response)

    return search
