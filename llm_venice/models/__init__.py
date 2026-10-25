"""Venice model implementations."""

import pathlib
import sys

import httpx2
import llm

from llm_venice.models.chat import AsyncVeniceChat, VeniceChat
from llm_venice.models.audio import AsyncVeniceSpeech, VeniceSpeech
from llm_venice.models.image import AsyncVeniceImage, VeniceImage
from llm_venice.constants import VENICE_API_BASE
from llm_venice.api.refresh import default_models_path, fetch_models, load_models, persist_models
from llm_venice.api.errors import VeniceAPIError
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
    """
    Return the Venice model list, or None if it is unavailable.

    Uses the cached file when it is readable. A missing or unreadable cache is
    (re)built from the API when a key is available. Never raises: a broken
    cache or network failure must not break unrelated llm commands.
    """
    cache_error = None
    if venice_models_path.exists():
        try:
            return load_models(venice_models_path)
        except (OSError, ValueError) as exc:
            cache_error = exc

    key = llm.get_key("", "venice", "LLM_VENICE_KEY")
    models = None
    if key:
        try:
            models = fetch_models(key)
        except (httpx2.RequestError, ValueError, VeniceAPIError):
            models = None

    if models is None:
        if cache_error is not None:
            _warn_once(
                f"ignoring unreadable Venice model cache {venice_models_path} "
                f"({cache_error}); run 'llm venice refresh' to rebuild it"
            )
        return None

    try:
        persist_models(models, venice_models_path)
    except OSError as exc:
        _warn_once(f"could not write Venice model cache {venice_models_path}: {exc}")
    return models


def register_venice_models(register):
    """
    Register all Venice models with the LLM plugin system.

    Args:
        register: The LLM registration function
    """
    models = _load_or_fetch_models(default_models_path())
    if not models:
        return

    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        model_id = model["id"]
        model_spec = model.get("model_spec") or {}
        if not isinstance(model_spec, dict):
            continue
        capabilities = model_spec.get("capabilities") or {}
        if not isinstance(capabilities, dict):
            continue
        constraints = model_spec.get("constraints")
        if constraints is not None and not isinstance(constraints, dict):
            continue
        supports_web_search = model_spec.get(
            "supportsWebSearch",
            capabilities.get("supportsWebSearch", False),
        )

        if model.get("type") == "text":
            model_instance = VeniceChat(
                model_id=f"venice/{model_id}",
                model_name=model_id,
                api_base=VENICE_API_BASE,
                can_stream=True,
                vision=capabilities.get("supportsVision", False),
                supports_schema=capabilities.get("supportsResponseSchema", False),
                supports_tools=capabilities.get("supportsFunctionCalling", False),
                supports_web_search=supports_web_search,
            )
            async_model_instance = AsyncVeniceChat(
                model_id=f"venice/{model_id}",
                model_name=model_id,
                api_base=VENICE_API_BASE,
                can_stream=True,
                vision=capabilities.get("supportsVision", False),
                supports_schema=capabilities.get("supportsResponseSchema", False),
                supports_tools=capabilities.get("supportsFunctionCalling", False),
                supports_web_search=supports_web_search,
            )
            register(model_instance, async_model=async_model_instance)
        elif model.get("type") == "image":
            model_instance = VeniceImage(
                model_id=model_id,
                model_name=model_id,
                image_constraints=constraints,
                supports_web_search=supports_web_search,
            )
            async_model_instance = AsyncVeniceImage(
                model_id=model_id,
                model_name=model_id,
                image_constraints=constraints,
                supports_web_search=supports_web_search,
            )
            register(
                model_instance,
                async_model=async_model_instance,
            )
        elif model.get("type") == "tts":
            register(
                VeniceSpeech(model_id=model_id, model_name=model_id),
                async_model=AsyncVeniceSpeech(model_id=model_id, model_name=model_id),
            )


__all__ = [
    "VeniceChat",
    "AsyncVeniceChat",
    "VeniceImage",
    "AsyncVeniceImage",
    "VeniceSpeech",
    "AsyncVeniceSpeech",
    "register_venice_models",
]
