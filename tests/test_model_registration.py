import json

import httpx2
import llm
import pytest
from llm_venice import AsyncVeniceImage, AsyncVeniceSpeech, VeniceImage, VeniceSpeech
from llm_venice.api.refresh import fetch_models
from llm_venice.models import register_venice_models


def test_registers_from_cache_without_key(monkeypatch, tmp_path):
    """Ensure cached models register even when no key is configured."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "venice_models.json").write_text(
        json.dumps(
            [
                {
                    "id": "minimax-m25",
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

    assert [m.model_id for m in registered] == ["venice/minimax-m25"]
    assert [m.model_id for m in registered_async] == ["venice/minimax-m25"]


def test_registers_text_model_web_search_capability(monkeypatch, tmp_path):
    """Ensure supportsWebSearch from the catalog reaches text model instances."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "venice_models.json").write_text(
        json.dumps(
            [
                {
                    "id": "search-model",
                    "type": "text",
                    "model_spec": {"capabilities": {"supportsWebSearch": True}},
                },
                {
                    "id": "plain-model",
                    "type": "text",
                    "model_spec": {"capabilities": {}},
                },
            ]
        )
    )
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)

    registered = []
    registered_async = []

    def register(model, async_model=None, aliases=None):
        registered.append(model)
        if async_model:
            registered_async.append(async_model)

    register_venice_models(register)

    by_id = {m.model_id: m for m in registered}
    async_by_id = {m.model_id: m for m in registered_async}
    assert by_id["venice/search-model"].supports_web_search is True
    assert async_by_id["venice/search-model"].supports_web_search is True
    assert by_id["venice/plain-model"].supports_web_search is False
    assert async_by_id["venice/plain-model"].supports_web_search is False


def test_register_skips_without_cache_or_key(monkeypatch, tmp_path):
    """Skip registration when cache is missing and no key is available."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)

    registered = []

    def register(model, async_model=None, aliases=None):
        registered.append(model)

    register_venice_models(register)

    assert registered == []


def test_register_skips_on_request_error_without_cache(monkeypatch, tmp_path):
    """Network failures should never crash plugin loading on first run."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: "test-key")

    def fetch_failure(_key):
        raise httpx2.RequestError(
            "Network down",
            request=httpx2.Request("GET", "https://api.venice.ai/api/v1/models"),
        )

    monkeypatch.setattr("llm_venice.models.fetch_models", fetch_failure)

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
    constraints = {
        "aspectRatios": ["1:1", "16:9"],
        "defaultAspectRatio": "1:1",
        "resolutions": ["1K", "2K"],
        "defaultResolution": "1K",
    }
    (tmp_path / "venice_models.json").write_text(
        json.dumps(
            [
                {
                    "id": "qwen-image",
                    "type": "image",
                    "model_spec": {
                        "capabilities": {},
                        "constraints": constraints,
                        "supportsWebSearch": True,
                    },
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
    assert registered[0].image_constraints == constraints
    assert registered_async[0].image_constraints == constraints
    assert registered[0].supports_web_search is True
    assert registered_async[0].supports_web_search is True


def test_registers_tts_async_model(monkeypatch, tmp_path):
    """Ensure TTS models register both sync and async variants."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "venice_models.json").write_text(
        json.dumps(
            [
                {
                    "id": "tts-kokoro",
                    "type": "tts",
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

    assert isinstance(registered[0], VeniceSpeech)
    assert isinstance(registered_async[0], AsyncVeniceSpeech)


@pytest.fixture
def clear_registration_warnings(monkeypatch):
    """Reset the warn-once cache so each test observes its own warnings."""
    monkeypatch.setattr("llm_venice.models._emitted_warnings", set())


@pytest.mark.parametrize(
    "content",
    [
        '[{"id": broken',  # truncated / invalid JSON
        '{"data": []}',  # valid JSON, wrong shape
        "",  # empty file
    ],
)
def test_corrupt_cache_without_key_warns_and_skips(
    monkeypatch, tmp_path, capsys, clear_registration_warnings, content
):
    """A corrupt cache must not crash plugin loading."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "venice_models.json").write_text(content)
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)

    registered = []
    register_venice_models(lambda model, async_model=None, aliases=None: registered.append(model))
    # A second registration pass in the same process must not repeat the warning
    register_venice_models(lambda model, async_model=None, aliases=None: registered.append(model))

    assert registered == []
    err = capsys.readouterr().err
    assert err.count("Warning: ignoring unreadable Venice model cache") == 1
    assert "llm venice refresh" in err


def test_corrupt_cache_is_rebuilt_when_key_available(
    monkeypatch, tmp_path, capsys, clear_registration_warnings
):
    """With a key, a corrupt cache is refetched and replaced without warning."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    cache = tmp_path / "venice_models.json"
    cache.write_text('[{"id": broken')
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: "test-key")
    fresh = [{"id": "fresh-model", "type": "text", "model_spec": {"capabilities": {}}}]
    monkeypatch.setattr("llm_venice.models.fetch_models", lambda _key: fresh)

    registered = []
    register_venice_models(lambda model, async_model=None, aliases=None: registered.append(model))

    assert [m.model_id for m in registered] == ["venice/fresh-model"]
    assert json.loads(cache.read_text()) == fresh
    assert capsys.readouterr().err == ""


def test_corrupt_cache_and_failed_fetch_warns(
    monkeypatch, tmp_path, capsys, clear_registration_warnings
):
    """If rebuilding fails too, warn and keep the broken file for inspection."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    cache = tmp_path / "venice_models.json"
    cache.write_text("not json")
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: "test-key")

    def fetch_failure(_key):
        raise httpx2.RequestError(
            "Network down",
            request=httpx2.Request("GET", "https://api.venice.ai/api/v1/models"),
        )

    monkeypatch.setattr("llm_venice.models.fetch_models", fetch_failure)

    registered = []
    register_venice_models(lambda model, async_model=None, aliases=None: registered.append(model))

    assert registered == []
    assert cache.read_text() == "not json"
    assert "ignoring unreadable Venice model cache" in capsys.readouterr().err


def test_malformed_entries_are_skipped(monkeypatch, tmp_path):
    """Individual malformed entries are skipped; valid ones still register."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "venice_models.json").write_text(
        json.dumps(
            [
                "not-a-dict",
                {"type": "text"},  # missing id
                {"id": "null-spec", "type": "text", "model_spec": None},
                {"id": "ok", "type": "text", "model_spec": {"capabilities": None}},
            ]
        )
    )
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)

    registered = []
    register_venice_models(lambda model, async_model=None, aliases=None: registered.append(model))

    assert [m.model_id for m in registered] == ["venice/null-spec", "venice/ok"]


@pytest.mark.parametrize(
    "bad_entry",
    [
        {"id": "string-spec", "type": "text", "model_spec": "broken"},
        {"id": "list-spec", "type": "text", "model_spec": ["broken"]},
        {"id": "string-capabilities", "type": "text", "model_spec": {"capabilities": "broken"}},
        {"id": "list-capabilities", "type": "text", "model_spec": {"capabilities": ["broken"]}},
    ],
)
def test_malformed_nested_fields_are_skipped(monkeypatch, tmp_path, bad_entry):
    """Non-dict model_spec and capabilities values are skipped, not read."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "venice_models.json").write_text(
        json.dumps(
            [
                bad_entry,
                {"id": "ok", "type": "text", "model_spec": {"capabilities": {}}},
            ]
        )
    )
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)

    registered = []
    register_venice_models(lambda model, async_model=None, aliases=None: registered.append(model))

    assert [m.model_id for m in registered] == ["venice/ok"]


def test_persist_failure_still_registers(
    monkeypatch, tmp_path, capsys, clear_registration_warnings
):
    """Failing to write the cache should not prevent registration."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: "test-key")
    fresh = [{"id": "m", "type": "text", "model_spec": {}}]
    monkeypatch.setattr("llm_venice.models.fetch_models", lambda _key: fresh)

    def persist_failure(*_args, **_kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr("llm_venice.models.persist_models", persist_failure)

    registered = []
    register_venice_models(lambda model, async_model=None, aliases=None: registered.append(model))

    assert [m.model_id for m in registered] == ["venice/m"]
    assert "could not write Venice model cache" in capsys.readouterr().err
