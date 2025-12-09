import json

import llm
import pytest
from llm_venice import AsyncVeniceImage, VeniceImage
from llm_venice.api.refresh import fetch_models
from llm_venice.models import register_venice_models


def test_registers_from_cache_without_key(monkeypatch, tmp_path):
    """Ensure cached models register even when no key is configured."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "venice_models.json").write_text(
        json.dumps(
            [
                {
                    "id": "qwen3-4b",
                    "type": "text",
                    "model_spec": {"capabilities": {}},
                }
            ]
        )
    )

    # Simulate no stored/env key
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)

    registered = []
    registered_async = []

    def register(model, async_model=None, aliases=None):
        registered.append(model)
        if async_model:
            registered_async.append(async_model)

    register_venice_models(register)

    assert [m.model_id for m in registered] == ["venice/qwen3-4b"]
    assert [m.model_id for m in registered_async] == ["venice/qwen3-4b"]


def test_register_skips_without_cache_or_key(monkeypatch, tmp_path):
    """Skip registration when cache is missing and no key is available."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)

    registered = []

    def register(model, async_model=None, aliases=None):
        registered.append(model)

    register_venice_models(register)

    assert registered == []


def test_fetch_models_missing_key_raises_needs_key(monkeypatch):
    """fetch_models should raise NeedsKeyException when no key is available."""
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)

    with pytest.raises(llm.NeedsKeyException):
        fetch_models()


def test_registers_image_async_model(monkeypatch, tmp_path):
    """Ensure image models register both sync and async variants."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "venice_models.json").write_text(
        json.dumps(
            [
                {
                    "id": "qwen-image",
                    "type": "image",
                    "model_spec": {"capabilities": {}},
                }
            ]
        )
    )

    registered = []
    registered_async = []

    def register(model, async_model=None, aliases=None):
        registered.append(model)
        registered_async.append(async_model)

    register_venice_models(register)

    assert isinstance(registered[0], VeniceImage)
    assert isinstance(registered_async[0], AsyncVeniceImage)
