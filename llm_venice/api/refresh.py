"""Model refresh functionality for Venice API."""

import json
from typing import Optional
import pathlib

import httpx2
import llm

from llm_venice.constants import ENDPOINT_MODELS
from llm_venice.api.client import get_auth_headers
from llm_venice.api.errors import raise_api_error
from llm_venice.utils import write_json_atomic


def default_models_path() -> pathlib.Path:
    """Return the default location of the cached Venice model list."""
    return llm.user_dir() / "venice_models.json"


def fetch_models(key: Optional[str] = None):
    """
    Fetch the list of models from the Venice API.

    Raises:
        NeedsKeyException if no key is found.
        ValueError if the API returns an empty or malformed model list.

    Returns:
        List of model dictionaries.
    """
    headers = get_auth_headers(key)

    models_response = httpx2.get(
        ENDPOINT_MODELS,
        headers=headers,
        params={"type": "all"},
    )
    try:
        models_response.raise_for_status()
    except httpx2.HTTPStatusError as exc:
        raise_api_error("Fetching model list", exc)

    payload = models_response.json()
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError("Unexpected response from Venice API: missing model list")
    if not models:
        raise ValueError("No models found")

    return models


def load_models(path: Optional[pathlib.Path] = None) -> list:
    """
    Load the cached model list from disk.

    Args:
        path: Optional custom path. Defaults to llm.user_dir()/venice_models.json.

    Raises:
        OSError if the file cannot be read.
        ValueError if the file is not valid JSON or does not contain a list.

    Returns:
        List of model dictionaries.
    """
    target_path = default_models_path() if path is None else path
    try:
        models = json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(models, list):
        raise ValueError("expected a JSON list of models")
    return models


def persist_models(models, path: Optional[pathlib.Path] = None):
    """
    Persist the provided models list to disk.

    The file is replaced atomically, so an interrupted write never leaves
    a truncated cache behind.

    Args:
        models: List of model dictionaries to write.
        path: Optional custom path. Defaults to llm.user_dir()/venice_models.json.

    Returns:
        The path that was written.
    """
    target_path = default_models_path() if path is None else path
    write_json_atomic(target_path, models)
    return target_path
