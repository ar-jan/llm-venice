"""API keys management commands for Venice CLI."""

from datetime import timezone
import json
from typing import get_args

import click

from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceError
from llm_venice.cli.errors import handle_cli_error
from llm_venice.types import ApiKeyLimitPeriod, ApiKeyModelPrivacy

LIMIT_PERIODS = get_args(ApiKeyLimitPeriod)
MODEL_PRIVACY = get_args(ApiKeyModelPrivacy)


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
        ctx.obj = {"client": ctx.with_resource(VeniceClient())}

        # Default to listing API keys if no subcommand is provided
        if not ctx.invoked_subcommand:
            ctx.invoke(list_keys)

    @api_keys_group.command(name="list")
    @click.pass_context
    def list_keys(ctx):
        """List all API keys."""
        try:
            response = ctx.obj["client"]._json_operation("keys")
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        else:
            click.echo(json.dumps(response, indent=2))

    @api_keys_group.command(name="rate-limits")
    @click.pass_context
    def rate_limits(ctx):
        """Show current rate limits for your API key"""
        try:
            response = ctx.obj["client"]._json_operation("rate_limits")
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        else:
            click.echo(json.dumps(response, indent=2))

    @api_keys_group.command(name="rate-limits-log")
    @click.pass_context
    def rate_limits_log(ctx):
        """Show the last 50 rate limit logs for the account"""
        try:
            response = ctx.obj["client"]._json_operation("rate_limits_log")
        except (VeniceError, ValueError, OSError) as e:
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
        "--limits-usd",
        type=click.FloatRange(min=0.0),
        default=None,
        help="USD consumption limit per limit period",
    )
    @click.option(
        "--limits-diem",
        type=click.FloatRange(min=0.0),
        default=None,
        help="Diem consumption limit per limit period",
    )
    @click.option(
        "--limits-vcu",
        type=click.FloatRange(min=0.0),
        default=None,
        help="VCU consumption limit per limit period (deprecated; use --limits-diem)",
    )
    @click.option(
        "--limit-period",
        type=click.Choice(LIMIT_PERIODS),
        default=None,
        help="When the limits reset: EPOCH (each UTC day, Venice's default), MONTH or LIFETIME",
    )
    @click.option(
        "--model-privacy",
        type=click.Choice(MODEL_PRIVACY),
        default=None,
        help="Models the key may call: ALL, PRIVATE_TEXT or PRIVATE_ONLY",
    )
    @click.option(
        "--ip-allowlist",
        multiple=True,
        metavar="ADDRESS",
        help="IP address or CIDR subnet allowed to use the key (repeatable; default any)",
    )
    @click.pass_context
    def create_key(
        ctx,
        description,
        key_type,
        expiration_date,
        limits_usd,
        limits_diem,
        limits_vcu,
        limit_period,
        model_privacy,
        ip_allowlist,
    ):
        """Create a new API key."""
        expiration = expiration_date.replace(tzinfo=timezone.utc) if expiration_date else None
        try:
            response = ctx.obj["client"]._json_operation(
                "create_key",
                description=description,
                key_type=key_type,
                expires_at=expiration,
                limit_usd=limits_usd,
                limit_diem=limits_diem,
                limit_vcu=limits_vcu,
                limit_period=limit_period,
                model_privacy=model_privacy,
                ip_allowlist=list(ip_allowlist) or None,
            )
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        else:
            click.echo(json.dumps(response, indent=2))

    @api_keys_group.command(name="delete")
    @click.argument("api_key_id")
    @click.pass_context
    def delete_key(ctx, api_key_id):
        """Delete an API key by ID."""
        try:
            response = ctx.obj["client"]._json_operation("delete_key", id=api_key_id)
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        else:
            click.echo(json.dumps(response, indent=2))

    return api_keys_group
