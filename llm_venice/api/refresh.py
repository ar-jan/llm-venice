"""Model refresh functionality for Venice API."""

import json

import click
import httpx
import llm

from llm_venice.constants import ENDPOINT_MODELS
from llm_venice.api.client import get_auth_headers


def refresh_models():
    """
    Refresh the list of models from the Venice API.

    Fetches all available models and saves them to a cache file.

    Raises:
        click.ClickException: If no models are found.

    Returns:
        List of model dictionaries.
    """
    headers = get_auth_headers()

    models_response = httpx.get(
        ENDPOINT_MODELS,
        headers=headers,
        params={"type": "all"},
    )
    models_response.raise_for_status()
    models = models_response.json()["data"]

    if not models:
        raise click.ClickException("No models found")

    path = llm.user_dir() / "venice_models.json"
    path.write_text(json.dumps(models, indent=4))
    click.echo(f"{len(models)} models saved to {path}", err=True)

    return models
