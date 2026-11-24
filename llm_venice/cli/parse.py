"""Document text extraction command for Venice CLI."""

import json
from pathlib import Path
from typing import get_args

import click

from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceError
from llm_venice.cli.errors import handle_cli_error
from llm_venice.cli.output import clean_text
from llm_venice.types import AugmentTextFormat


def create_parse_command():
    """
    Create the parse command.

    Returns:
        Click command for extracting a document's text
    """

    @click.command(name="parse")
    @click.argument(
        "file", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path)
    )
    @click.option(
        "--format",
        "response_format",
        type=click.Choice(get_args(AugmentTextFormat)),
        default="text",
        show_default=True,
        help="text prints the extracted text; json adds its token count",
    )
    def parse(file, response_format):
        """Extract the text of a document with Venice, without a model.

        Uses Venice's /augment/text-parser endpoint, which reads PDF, DOCX, PPTX,
        XLSX, EPUB, plain text, Markdown, CSV, JSON and most source-code files up to
        25 MB, telling the format from the file extension. Venice parses the file
        in memory and keeps no copy. Each request is billed.

        \b
        Example:
          llm venice parse report.pdf | llm -m venice/qwen-3-8-flash "Summarize this"
        """
        try:
            with VeniceClient() as client:
                result = client.augment_parse_text(file, response_format=response_format)
        except (VeniceError, ValueError, OSError) as e:
            handle_cli_error(e)
        if isinstance(result, str):
            click.echo(clean_text(result))
        else:
            click.echo(json.dumps(result, indent=2))

    return parse
