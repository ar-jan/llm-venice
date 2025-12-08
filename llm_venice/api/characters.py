"""Character retrieval for Venice API."""

import json
import pathlib
from typing import Optional

import httpx
import llm

from llm_venice.api.client import get_auth_headers
from llm_venice.api.errors import raise_api_error
from llm_venice.constants import ENDPOINT_CHARACTERS


def list_characters(
    key: Optional[str] = None,
    *,
    web_enabled: Optional[str] = None,
    adult: Optional[str] = None,
) -> dict:
    """
    Fetch public characters from the Venice API.

    Args:
        key: Optional explicit API key.
        web_enabled: Optional "true"/"false" filter.
        adult: Optional "true"/"false" filter.

    Returns:
        Parsed JSON response from the API.
    """
    headers = get_auth_headers(key)
    params = {k: v for k, v in {"isWebEnabled": web_enabled, "isAdult": adult}.items() if v}

    response = httpx.get(
        ENDPOINT_CHARACTERS,
        headers=headers,
        params=params,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise_api_error("Listing characters", exc)
    return response.json()


def persist_characters(characters: dict, path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """
    Write character data to disk.

    Args:
        characters: Parsed character data.
        path: Optional custom path, defaults to llm.user_dir()/venice_characters.json.

    Returns:
        The written path.
    """
    target_path = llm.user_dir() / "venice_characters.json" if path is None else path
    target_path.write_text(json.dumps(characters, indent=4))
    return target_path
