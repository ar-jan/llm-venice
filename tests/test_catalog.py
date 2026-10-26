"""Disk-only persistence and bounded, cache-first registration policy."""

import json
import os
from unittest.mock import patch

import httpx2
import llm
import pytest

from llm_venice.catalog import load_catalog, refresh_catalog, save_catalog
from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceResponseError
from llm_venice.models import register_venice_models
from llm_venice.models._common import warn_deprecated
from llm_venice import _catalog_state


def registered():
    models = []
    register_venice_models(lambda model, async_model=None: models.extend((model, async_model)))
    return models


def test_old_cache_avoids_keys_and_http(tmp_path, monkeypatch, httpx2_mock, capsys):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    path = tmp_path / "venice_models.json"
    save_catalog([{"id": "m", "type": "text"}], path)
    os.utime(path, (0, 0))

    def no_key(*args, **kwargs):
        pytest.fail("usable cache must not resolve keys")

    monkeypatch.setattr(llm, "get_key", no_key)
    assert len(registered()) == 2
    assert load_catalog(path) == [{"id": "m", "type": "text"}]
    assert not httpx2_mock.get_requests()
    assert not capsys.readouterr().err


def test_failed_bootstrap_is_memoized_per_configuration(tmp_path, monkeypatch, httpx2_mock):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setattr(llm, "get_key", lambda *a: "first")
    httpx2_mock.add_exception(httpx2.ReadTimeout("unavailable"))
    assert registered() == registered() == []
    assert len(httpx2_mock.get_requests()) == 1
    assert httpx2_mock.get_requests()[0].extensions["timeout"]["read"] == 5
    monkeypatch.setattr(llm, "get_key", lambda *a: "second")
    httpx2_mock.add_exception(httpx2.ReadTimeout("unavailable"))
    assert registered() == []
    assert len(httpx2_mock.get_requests()) == 2
    httpx2_mock.add_response(json={"data": [{"id": "m", "type": "text"}]})
    refresh_catalog()
    assert not any(
        item[0] == str((tmp_path / "venice_models.json").resolve())
        for item in _catalog_state.failed_bootstraps
    )
    assert len(registered()) == 2


@pytest.mark.parametrize("payload", [{"data": []}, {"data": [{"id": 42}]}, {"oops": []}])
def test_bad_refresh_preserves_good_cache(tmp_path, httpx2_mock, payload):
    path = tmp_path / "catalog"
    save_catalog([{"id": "old", "type": "text"}], path)
    previous = path.read_bytes()
    httpx2_mock.add_response(json=payload)
    with VeniceClient(key="k") as client:
        with pytest.raises(VeniceResponseError):
            refresh_catalog(client, path)
    assert path.read_bytes() == previous


def test_refresh_owns_only_temporary_client(tmp_path, httpx2_mock, monkeypatch):
    monkeypatch.setattr(llm, "get_key", lambda *a: "k")
    httpx2_mock.add_response(json={"data": [{"id": "m", "type": "text"}]})
    with patch.object(VeniceClient, "close", autospec=True) as close:
        refresh_catalog(path=tmp_path / "catalog")
        assert close.call_count == 1
    with VeniceClient(key="k") as client:
        httpx2_mock.add_response(json={"data": [{"id": "m", "type": "text"}]})
        refresh_catalog(client, tmp_path / "catalog")
        httpx2_mock.add_response(json={"data": []})
        assert client.list_models() == []


def test_raw_fields_retained_and_bad_entries_skipped(tmp_path, capsys):
    path = tmp_path / "catalog"
    raw = [
        None,
        {"id": "m", "type": "future", "future": {"value": 1}, "model_spec": {"traits": ["raw"]}},
    ]
    path.write_text(json.dumps(raw))
    loaded = load_catalog(path)
    assert loaded == raw[1:]
    assert loaded is not None
    save_catalog(loaded, path)
    assert json.loads(path.read_text()) == raw[1:]
    assert capsys.readouterr().err == ""
    path.write_text("bad json")
    assert load_catalog(path) is None


def test_offline_and_traits_and_deprecation(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    save_catalog(
        [
            {"id": "offline", "type": "text", "model_spec": {"offline": True}},
            {
                "id": "venice/old",
                "type": "tts",
                "model_spec": {"traits": ["fast"], "deprecation": {"replacementModelId": "new"}},
            },
        ]
    )
    models = registered()
    assert [m.model_id for m in models] == ["venice/old", "venice/old"]
    assert capsys.readouterr().err == ""
    warn_deprecated(models[0])
    warn_deprecated(models[1])
    assert capsys.readouterr().err == "Warning: venice/old is deprecated; use venice/new instead\n"
