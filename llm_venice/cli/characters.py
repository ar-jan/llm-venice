"""Characters command for Venice CLI."""

from typing import Any

import click
import llm

from llm_venice._endpoints import CHARACTER_PAGE_LIMIT
from llm_venice.client import VeniceClient
from llm_venice.utils import write_json_atomic
from llm_venice.errors import VeniceError
from llm_venice._notices import VeniceNotice
from llm_venice.cli.errors import handle_cli_error
from llm_venice.cli.notices import emit_cli_notices

# Venice reports no total, so stop after this many pages even if each is full.
MAX_CHARACTER_PAGES = 100


def fetch_all_characters(client: VeniceClient, **filters: Any) -> tuple[dict[str, Any], bool]:
    """Page through every character, returning the merged envelope and whether it was cut short.

    Paging stops at a short or empty page, at a page with no new IDs (a server
    that ignores ``offset``), or after MAX_CHARACTER_PAGES pages.
    """
    envelope: dict[str, Any] | None = None
    characters: list[Any] = []
    seen: set[Any] = set()
    for page in range(MAX_CHARACTER_PAGES):
        response = client._json_operation(
            "characters",
            limit=CHARACTER_PAGE_LIMIT,
            offset=page * CHARACTER_PAGE_LIMIT,
            **filters,
        )
        if envelope is None:
            envelope = response
        batch = response.get("data", [])
        new = [item for item in batch if item.get("id") not in seen]
        seen.update(item.get("id") for item in new)
        characters.extend(new)
        if len(batch) < CHARACTER_PAGE_LIMIT or not new:
            return {**envelope, "data": characters}, False
    assert envelope is not None
    return {**envelope, "data": characters}, True


def create_characters_command():
    """
    Create the characters command.

    Returns:
        Click command for listing characters
    """

    @click.command(name="characters")
    @click.option(
        "--web-enabled",
        type=click.Choice(["true", "false"]),
        help="Filter by web-enabled status",
    )
    @click.option("--adult", type=click.Choice(["true", "false"]), help="Filter by adult category")
    def characters(web_enabled, adult):
        """List public characters, fetching every page."""
        try:
            with VeniceClient() as client:
                characters_data, truncated = fetch_all_characters(
                    client,
                    web_enabled=None if web_enabled is None else web_enabled == "true",
                    adult=None if adult is None else adult == "true",
                )
            path = llm.user_dir() / "venice_characters.json"
            write_json_atomic(path, characters_data)
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        characters_count = len(characters_data["data"])
        if truncated:
            message = f"stopped after {MAX_CHARACTER_PAGES} pages; the saved list may be incomplete"
            emit_cli_notices([VeniceNotice("warning", message)])
        click.echo(f"{characters_count} characters saved to {path}", err=True)

    return characters
