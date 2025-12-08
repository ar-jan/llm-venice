"""API keys management commands for Venice CLI."""

import json

import click

from llm_venice.api import keys as api_keys
from llm_venice.api.errors import VeniceAPIError
from llm_venice.cli.errors import handle_cli_error
from llm_venice.utils import get_venice_key


def create_api_keys_group():
    """
    Create the API keys command group.

    Returns:
        Click group for API key management commands
    """

    @click.group(name="api-keys", invoke_without_command=True)
    @click.pass_context
    def api_keys_group(ctx):
        """Manage API keys - list, create, delete, rate-limits"""
        # Retrieve the API key once and store it in context
        key = get_venice_key(click_exceptions=True)
        ctx.obj = {"headers": {"Authorization": f"Bearer {key}"}}

        # Default to listing API keys if no subcommand is provided
        if not ctx.invoked_subcommand:
            ctx.invoke(list_keys)

    @api_keys_group.command(name="list")
    @click.pass_context
    def list_keys(ctx):
        """List all API keys."""
        try:
            response = api_keys.list_api_keys(ctx.obj["headers"])
        except VeniceAPIError as e:
            handle_cli_error(e)
        else:
            click.echo(json.dumps(response, indent=2))

    @api_keys_group.command(name="rate-limits")
    @click.pass_context
    def rate_limits(ctx):
        """Show current rate limits for your API key"""
        try:
            response = api_keys.get_rate_limits(ctx.obj["headers"])
        except VeniceAPIError as e:
            handle_cli_error(e)
        else:
            click.echo(json.dumps(response, indent=2))

    @api_keys_group.command(name="rate-limits-log")
    @click.pass_context
    def rate_limits_log(ctx):
        """Show the last 50 rate limit logs for the account"""
        try:
            response = api_keys.get_rate_limits_log(ctx.obj["headers"])
        except VeniceAPIError as e:
            handle_cli_error(e)
        else:
            click.echo(json.dumps(response, indent=2))

    @api_keys_group.command(name="create")
    @click.option(
        "--type",
        "key_type",
        type=click.Choice(["ADMIN", "INFERENCE"]),
        required=True,
        help="Type of API key",
    )
    @click.option("--description", default="", help="Description for the new API key")
    @click.option(
        "--expiration-date",
        type=click.DateTime(
            formats=[
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M",
                "%Y-%m-%dT%H:%M:%S",
            ]
        ),
        default=None,
        help="The API Key expiration date",
    )
    @click.option(
        "--limits-vcu",
        type=click.FloatRange(min=0.0),
        default=None,
        help="VCU consumption limit per epoch",
    )
    @click.option(
        "--limits-usd",
        type=click.FloatRange(min=0.0),
        default=None,
        help="USD consumption limit per epoch",
    )
    @click.pass_context
    def create_key(ctx, description, key_type, expiration_date, limits_vcu, limits_usd):
        """Create a new API key."""
        expiration_str = expiration_date.strftime("%Y-%m-%dT%H:%M:%SZ") if expiration_date else None
        try:
            response = api_keys.create_api_key(
                ctx.obj["headers"],
                description,
                key_type,
                expiration_str,
                limits_vcu,
                limits_usd,
            )
        except VeniceAPIError as e:
            handle_cli_error(e)
        else:
            click.echo(json.dumps(response, indent=2))

    @api_keys_group.command(name="delete")
    @click.argument("api_key_id")
    @click.pass_context
    def delete_key(ctx, api_key_id):
        """Delete an API key by ID."""
        try:
            response = api_keys.delete_api_key(ctx.obj["headers"], api_key_id)
        except VeniceAPIError as e:
            handle_cli_error(e)
        else:
            click.echo(json.dumps(response, indent=2))

    return api_keys_group
