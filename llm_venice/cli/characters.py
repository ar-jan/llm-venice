"""Characters commands for Venice CLI."""

import json
from typing import Any, get_args

import click
from click.core import ParameterSource
import llm

from llm_venice._endpoints import CHARACTER_PAGE_LIMIT
from llm_venice.models._common import normalize_model_id
from llm_venice.types import CharacterSortBy, CharacterSortOrder
from llm_venice.client import VeniceClient
from llm_venice.utils import write_json_atomic
from llm_venice.errors import VeniceError
from llm_venice._notices import VeniceNotice
from llm_venice.cli.errors import handle_cli_error
from llm_venice.cli.notices import emit_cli_notices

# Venice reports no total, so stop after this many pages even if each is full.
MAX_CHARACTER_PAGES = 100


def fetch_all_characters(client: VeniceClient, **filters: Any) -> tuple[dict[str, Any], str | None]:
    """Page through every character, returning the merged envelope and why it may be incomplete.

    Paging ends at a short or empty page. It also stops, reporting the reason, at
    a full page that adds no new IDs (a server that ignores ``offset``) and after
    MAX_CHARACTER_PAGES pages.
    """
    envelope: dict[str, Any] | None = None
    characters: list[Any] = []
    seen: set[Any] = set()
    for page in range(MAX_CHARACTER_PAGES):
        offset = page * CHARACTER_PAGE_LIMIT
        response = client._json_operation(
            "characters", limit=CHARACTER_PAGE_LIMIT, offset=offset, **filters
        )
        if envelope is None:
            envelope = response
        batch = response.get("data", [])
        added = 0
        for item in batch:
            if item.get("id") in seen:
                continue
            seen.add(item.get("id"))
            characters.append(item)
            added += 1
        if len(batch) < CHARACTER_PAGE_LIMIT:
            return {**envelope, "data": characters}, None
        if not added:
            return {**envelope, "data": characters}, (
                f"Venice returned a full page with no new characters at offset {offset}"
            )
    assert envelope is not None
    return {**envelope, "data": characters}, f"stopped after {MAX_CHARACTER_PAGES} pages"


def create_characters_command():
    """
    Create the characters command.

    Returns:
        Click group that lists characters without a subcommand, as before
    """

    # A group whose list options stay on the group keeps every earlier
    # `llm venice characters [OPTIONS]` invocation working unchanged.
    @click.group(name="characters", invoke_without_command=True)
    @click.option(
        "--web-enabled",
        type=click.Choice(["true", "false"]),
        help="Filter by web-enabled status",
    )
    @click.option("--adult", type=click.Choice(["true", "false"]), help="Filter by adult category")
    @click.option(
        "--pro", type=click.Choice(["true", "false"]), help="Filter by use of a Pro model"
    )
    @click.option("--category", "categories", multiple=True, help="Filter by category (repeatable)")
    @click.option("--tag", "tags", multiple=True, help="Filter by tag (repeatable)")
    @click.option("--model-id", "model_ids", multiple=True, help="Filter by model ID (repeatable)")
    @click.option("--search", help="Search names, descriptions and tags (#tag supported)")
    @click.option("--sort-by", type=click.Choice(get_args(CharacterSortBy)), help="Sort order")
    @click.option(
        "--sort-order",
        type=click.Choice(get_args(CharacterSortOrder)),
        help="Sort direction (Venice's default: desc)",
    )
    @click.pass_context
    def characters(
        ctx, web_enabled, adult, pro, categories, tags, model_ids, search, sort_by, sort_order
    ):
        """List public characters, fetching every page, or show one character or its reviews.

        Without a subcommand, every page of characters matching the options is saved to
        venice_characters.json in the llm user directory.
        """
        if ctx.invoked_subcommand:
            for param in ctx.command.params:
                if (
                    param.name
                    and ctx.get_parameter_source(param.name) is not ParameterSource.DEFAULT
                ):
                    raise click.UsageError(
                        f"{param.opts[0]} applies only to listing characters, "
                        f"not to '{ctx.invoked_subcommand}'"
                    )
            return

        def flag(value):
            return None if value is None else value == "true"

        try:
            with VeniceClient() as client:
                characters_data, incomplete = fetch_all_characters(
                    client,
                    web_enabled=flag(web_enabled),
                    adult=flag(adult),
                    pro=flag(pro),
                    categories=list(categories) or None,
                    tags=list(tags) or None,
                    # llm names models venice/<id>; the filter takes the bare ID.
                    model_ids=[normalize_model_id(model) for model in model_ids] or None,
                    search=search,
                    sort_by=sort_by,
                    sort_order=sort_order,
                )
            path = llm.user_dir() / "venice_characters.json"
            write_json_atomic(path, characters_data)
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        characters_count = len(characters_data["data"])
        if incomplete:
            message = f"{incomplete}; the saved list may be incomplete"
            emit_cli_notices([VeniceNotice("warning", message)])
        click.echo(f"{characters_count} characters saved to {path}", err=True)

    @characters.command(name="get")
    @click.argument("slug")
    def get_character(slug):
        """Show one character's details by its slug."""
        try:
            with VeniceClient() as client:
                response = client._json_operation("character", slug=slug)
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        click.echo(json.dumps(response, indent=2))

    @characters.command(name="reviews")
    @click.argument("slug")
    @click.option("--page", type=click.IntRange(min=1), help="Page number, from 1")
    @click.option(
        "--page-size",
        type=click.IntRange(1, 100),
        help="Reviews per page (Venice's default: 20)",
    )
    def character_reviews(slug, page, page_size):
        """Show a page of a character's public reviews, with totals and rating summary."""
        try:
            with VeniceClient() as client:
                response = client._json_operation(
                    "character_reviews", slug=slug, page=page, page_size=page_size
                )
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        click.echo(json.dumps(response, indent=2))

    return characters
