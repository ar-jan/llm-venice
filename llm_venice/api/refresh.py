"""Model refresh functionality for Venice API."""

import json
from typing import Optional
import pathlib

import httpx
import llm

from llm_venice.constants import ENDPOINT_MODELS
from llm_venice.api.client import get_auth_headers
from llm_venice.api.errors import raise_api_error


def fetch_models(key: Optional[str] = None):
    """
    Fetch the list of models from the Venice API.

    Raises:
        NeedsKeyException if no key is found.
        ValueError if the API returns an empty model list.

    Returns:
        List of model dictionaries.
    """
    headers = get_auth_headers(key)

    models_response = httpx.get(
        ENDPOINT_MODELS,
        headers=headers,
        params={"type": "all"},
    )
    try:
        models_response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise_api_error("Fetching model list", exc)
    models = models_response.json()["data"]

    if not models:
        raise ValueError("No models found")

    return models


def persist_models(models, path: Optional[pathlib.Path] = None):
    """
    Persist the provided models list to disk.

    Args:
        models: List of model dictionaries to write.
        path: Optional custom path. Defaults to llm.user_dir()/venice_models.json.

    Returns:
        The path that was written.
    """
    target_path = llm.user_dir() / "venice_models.json" if path is None else path
    target_path.write_text(json.dumps(models, indent=4))
    return target_path
