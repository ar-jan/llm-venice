"""Venice model implementations."""

from typing import Any
import pathlib
import sys

import llm

from llm_venice.models.chat import (
    AsyncVeniceChat,
    VeniceChat,
    VeniceWebSearch,
    VeniceXSearch,
)
from llm_venice.models.audio import AsyncVeniceSpeech, VeniceSpeech
from llm_venice.models.image import AsyncVeniceImage, VeniceImage
from llm_venice.constants import VENICE_API_BASE
from llm_venice.catalog import catalog_path, read_catalog, save_catalog
from llm_venice.client import VeniceClient
from llm_venice import _catalog_state
from llm_venice.errors import VeniceError, VeniceResponseError
from llm_venice.notices import VeniceNotice, render_notice

# Model registration can run several times per process (once per model lookup),
# so remember which warnings were already shown.
_emitted_warnings: set[str] = set()


def _warn_once(message: str) -> None:
    if message in _emitted_warnings:
        return
    _emitted_warnings.add(message)
    # Stay click-free here: library modules must not depend on the CLI layer.
    print(render_notice(VeniceNotice("warning", message)), file=sys.stderr)


def _load_or_fetch_models(venice_models_path: pathlib.Path):
    """Cache first; bootstrap once per path/key with a bounded timeout."""
    cache_error = None
    try:
        return read_catalog(venice_models_path)
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        cache_error = exc
    try:
        key = llm.get_key(None, "venice", "LLM_VENICE_KEY")
    except (OSError, ValueError, llm.NeedsKeyException):
        key = None
    token = (str(venice_models_path.resolve()), key or "")
    if token in _catalog_state.fetched_catalogs:
        return _catalog_state.fetched_catalogs[token]
    models = None
    if key and token not in _catalog_state.failed_bootstraps:
        try:
            with VeniceClient(key=key, timeout=5.0) as client:
                models = client.list_models()
            if not models:
                raise VeniceResponseError("Fetching model list", "no usable models returned")
        except VeniceError as exc:
            _catalog_state.failed_bootstraps.add(token)
            if cache_error is None:
                _warn_once(f"could not fetch Venice model catalog: {exc}; run 'llm venice refresh'")
            models = None
    if models is None:
        if cache_error is not None:
            _warn_once(
                f"ignoring unreadable Venice model cache {venice_models_path} "
                f"({cache_error}); run 'llm venice refresh' to rebuild it"
            )
        return None
    try:
        save_catalog(models, venice_models_path)
    except OSError as exc:
        _catalog_state.fetched_catalogs[token] = models
        _warn_once(f"could not write Venice model cache {venice_models_path}: {exc}")
    return models


def register_venice_models(register):
    """Register concrete online IDs, using the same metadata for sync and async."""
    models = _load_or_fetch_models(catalog_path())
    for model in models or []:
        spec = model.get("model_spec") or {}
        if spec.get("offline") is True:
            continue
        capabilities = spec.get("capabilities") or {}
        if not isinstance(capabilities, dict):
            capabilities = {}
        constraints = spec.get("constraints")
        if not isinstance(constraints, dict):
            constraints = None
        common: dict[str, Any] = dict(model_id=model["id"], model_name=model["id"], model_spec=spec)
        web = spec.get("supportsWebSearch", capabilities.get("supportsWebSearch")) is True
        x_search = spec.get("supportsXSearch", capabilities.get("supportsXSearch")) is True
        if model["type"] == "text":
            common.update(
                api_base=VENICE_API_BASE,
                can_stream=True,
                vision=capabilities.get("supportsVision") is True,
                supports_schema=capabilities.get("supportsResponseSchema") is True,
                supports_tools=capabilities.get("supportsFunctionCalling") is True,
                supports_web_search=web,
                supports_x_search=x_search,
            )
            register(VeniceChat(**common), async_model=AsyncVeniceChat(**common))
        elif model["type"] == "image":
            common.update(image_constraints=constraints, supports_web_search=web)
            register(VeniceImage(**common), async_model=AsyncVeniceImage(**common))
        elif model["type"] == "tts":
            register(VeniceSpeech(**common), async_model=AsyncVeniceSpeech(**common))


__all__ = [
    "VeniceChat",
    "AsyncVeniceChat",
    "VeniceWebSearch",
    "VeniceXSearch",
    "VeniceImage",
    "AsyncVeniceImage",
    "VeniceSpeech",
    "AsyncVeniceSpeech",
    "register_venice_models",
]
