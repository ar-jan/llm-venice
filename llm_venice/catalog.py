"""Explicit catalog persistence. Loading never performs network I/O or prints."""

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any, cast

import llm

from . import _catalog_state
from ._endpoints import model_resource
from .client import VeniceClient
from .errors import VeniceResponseError
from .types import ModelInfo
from .utils import write_json_atomic


def catalog_path() -> Path:
    """Return llm's Venice catalog path; does not fetch a catalog."""
    return llm.user_dir() / "venice_models.json"


def read_catalog(path: Path) -> list[ModelInfo]:
    """Internal strict reader, allowing registration to distinguish cache failures."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("expected a JSON list of models")
    models: list[ModelInfo] = []
    for item in cast(list[Any], raw):
        try:
            models.append(model_resource(item, "Reading catalog"))
        except VeniceResponseError:
            continue
    if not models:
        raise ValueError("no usable models in catalog")
    return models


def load_catalog(path: Path | None = None) -> list[ModelInfo] | None:
    """Read disk only; return None for a missing, unreadable or unusable cache."""
    try:
        return read_catalog(catalog_path() if path is None else path)
    except (OSError, ValueError):
        return None


def save_catalog(models: Sequence[ModelInfo], path: Path | None = None) -> Path:
    """Atomically save a catalog list, preserving unknown fields and wire timestamps."""
    target = catalog_path() if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, list(models))
    return target


def refresh_catalog(
    client: VeniceClient | None = None, path: Path | None = None
) -> list[ModelInfo]:
    """Fetch and save a nonempty catalog; close only a client created here.

    Malformed entries are skipped; empty or wholly invalid responses never
    replace a good cache. Service and local file errors propagate. A successful
    explicit refresh clears failed registration attempts.
    """
    if client is None:
        with VeniceClient() as owned:
            models = owned.list_models()
    else:
        models = client.list_models()
    # list_models() already skipped malformed entries; reject an empty result.
    if not models:
        raise VeniceResponseError("Refreshing catalog", "no usable models returned")
    target = save_catalog(models, path)
    normalized = str(target.resolve())
    _catalog_state.failed_bootstraps.difference_update(
        [entry for entry in _catalog_state.failed_bootstraps if entry[0] == normalized]
    )
    for entry in list(_catalog_state.fetched_catalogs):
        if entry[0] == normalized:
            del _catalog_state.fetched_catalogs[entry]
    return models
