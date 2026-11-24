"""API keys management commands for Venice CLI."""

from datetime import timezone
import json
from typing import get_args

import click
from click.core import ParameterSource

from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceError
from llm_venice.cli.errors import handle_cli_error
from llm_venice.types import ApiKeyLimitPeriod, ApiKeyModelPrivacy

LIMIT_PERIODS = get_args(ApiKeyLimitPeriod)
MODEL_PRIVACY = get_args(ApiKeyModelPrivacy)
EXPIRATION_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"]


class LimitOrNone(click.ParamType):
    """A non-negative limit, or 'none' to remove the limit."""

    name = "FLOAT|none"

    def convert(self, value, param, ctx):
        if isinstance(value, str) and value.strip().lower() == "none":
            return None
        return click.FloatRange(min=0.0)(value, param, ctx)


def create_api_keys_group():
    """
    Create the API keys command group.

    Returns:
        Click group for API key management commands
    """

    @click.group(name="api-keys", invoke_without_command=True)
    @click.pass_context
    def api_keys_group(ctx):
        """Manage API keys - list, get, create, update, delete, rate-limits"""
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
        type=click.DateTime(formats=EXPIRATION_FORMATS),
        default=None,
        help="The API Key expiration date (UTC)",
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

    @api_keys_group.command(name="get")
    @click.argument("api_key_id")
    @click.pass_context
    def get_key(ctx, api_key_id):
        """Show an API key's details by ID."""
        try:
            response = ctx.obj["client"]._json_operation("get_key", id=api_key_id)
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        else:
            click.echo(json.dumps(response, indent=2))

    @api_keys_group.command(name="update")
    @click.argument("api_key_id")
    @click.option("--description", default=None, help="New description ('' clears it)")
    @click.option(
        "--expiration-date",
        type=click.DateTime(formats=EXPIRATION_FORMATS),
        default=None,
        help="New expiration date (UTC)",
    )
    @click.option("--no-expiration", is_flag=True, help="Remove the expiration date")
    @click.option(
        "--limits-usd",
        type=LimitOrNone(),
        default=None,
        help="USD consumption limit per limit period, or 'none' to remove it",
    )
    @click.option(
        "--limits-diem",
        type=LimitOrNone(),
        default=None,
        help="Diem consumption limit per limit period, or 'none' to remove it",
    )
    @click.option(
        "--limits-vcu",
        type=LimitOrNone(),
        default=None,
        help="VCU consumption limit (deprecated; use --limits-diem), or 'none'",
    )
    @click.option(
        "--limit-period",
        type=click.Choice(LIMIT_PERIODS),
        default=None,
        help="When the limits reset: EPOCH (each UTC day), MONTH or LIFETIME",
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
        help="Replace the IP allowlist with these addresses or CIDR subnets (repeatable)",
    )
    @click.option(
        "--clear-ip-allowlist",
        is_flag=True,
        help="Remove the IP allowlist, so any IP address may use the key",
    )
    @click.pass_context
    def update_key(ctx, api_key_id, **options):
        """Change an API key's settings; settings not given are kept."""

        def given(name):
            return ctx.get_parameter_source(name) is not ParameterSource.DEFAULT

        changes = {}
        if given("description"):
            changes["description"] = options["description"]
        if options["expiration_date"] and options["no_expiration"]:
            raise click.UsageError("Use --expiration-date or --no-expiration, not both")
        if options["expiration_date"]:
            changes["expires_at"] = options["expiration_date"].replace(tzinfo=timezone.utc)
        if options["no_expiration"]:
            changes["expires_at"] = None
        for option, field in (
            ("limits_usd", "limit_usd"),
            ("limits_diem", "limit_diem"),
            ("limits_vcu", "limit_vcu"),
        ):
            if given(option):
                changes[field] = options[option]
        for name in ("limit_period", "model_privacy"):
            if options[name] is not None:
                changes[name] = options[name]
        if options["ip_allowlist"] and options["clear_ip_allowlist"]:
            raise click.UsageError("Use --ip-allowlist or --clear-ip-allowlist, not both")
        if options["ip_allowlist"]:
            changes["ip_allowlist"] = list(options["ip_allowlist"])
        if options["clear_ip_allowlist"]:
            changes["ip_allowlist"] = []
        if not changes:
            raise click.UsageError("Give at least one setting to change")
        try:
            response = ctx.obj["client"]._json_operation("update_key", id=api_key_id, **changes)
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
