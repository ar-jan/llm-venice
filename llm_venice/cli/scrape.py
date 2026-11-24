"""Scrape command for Venice CLI."""

import json

import click
from pydantic import ValidationError

from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceError
from llm_venice.cli.errors import handle_cli_error
from llm_venice.cli.output import clean_text


def create_scrape_command():
    """
    Create the scrape command.

    Returns:
        Click command for fetching a web page as markdown
    """

    @click.command(name="scrape")
    @click.argument("url")
    @click.option("--json", "as_json", is_flag=True, help="Output the JSON response")
    def scrape(url, as_json):
        """Fetch a web page through Venice and print it as markdown.

        Uses Venice's /augment/scrape endpoint. Some sites that block automated
        access, such as X/Twitter and Reddit, are refused. Each scrape is billed.

        \b
        Example:
          llm venice scrape https://venice.ai > venice.md
        """
        try:
            with VeniceClient() as client:
                response = client.augment_scrape(url)
        except ValidationError as exc:
            error = exc.errors()[0]
            # Show the check's own message, without pydantic's "Value error, " prefix.
            cause = error.get("ctx", {}).get("error")
            raise click.BadParameter(str(cause or error["msg"]), param_hint="'URL'") from exc
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        if as_json:
            click.echo(json.dumps(response, indent=2))
        else:
            click.echo(clean_text(response["content"]))

    return scrape
