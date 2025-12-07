"""Model refresh functionality for Venice API."""

import json
from typing import Optional

import click
import httpx
import llm

from llm_venice.constants import ENDPOINT_MODELS
from llm_venice.api.client import get_auth_headers


def refresh_models(key: Optional[str] = None, *, use_click_exceptions: bool = False):
    """
    Refresh the list of models from the Venice API.

    Fetches all available models and saves them to a cache file.

    Raises:
        NeedsKeyException (or click.ClickException when use_click_exceptions=True) if no key is found.
        ValueError if the API returns an empty model list.

    Returns:
        List of model dictionaries.
    """
    headers = get_auth_headers(key, click_exceptions=use_click_exceptions)

    models_response = httpx.get(
        ENDPOINT_MODELS,
        headers=headers,
        params={"type": "all"},
    )
    models_response.raise_for_status()
    models = models_response.json()["data"]

    if not models:
        if use_click_exceptions:
            raise click.ClickException("No models found")
        raise ValueError("No models found")

    path = llm.user_dir() / "venice_models.json"
    path.write_text(json.dumps(models, indent=4))
    click.echo(f"{len(models)} models saved to {path}", err=True)

    return models
