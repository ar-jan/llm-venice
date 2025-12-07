import json

import llm
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

    register_venice_models(registered.append)

    assert [m.model_id for m in registered] == ["venice/qwen3-4b"]


def test_register_warns_without_cache_or_key(monkeypatch, tmp_path, capsys):
    """Skip registration when cache is missing and no key is available."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)

    register_venice_models(lambda *_: None)

    err = capsys.readouterr().err
    assert "skipped refreshing venice models" in err.lower()
