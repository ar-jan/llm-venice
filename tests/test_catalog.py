"""Disk-only persistence, cache-first registration policy, and model trait mappings."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import httpx2
import llm
import pytest
from llm.cli import cli
from pydantic import ValidationError
from typing_extensions import assert_type

from llm_venice.catalog import load_catalog, refresh_catalog, save_catalog
from llm_venice.client import AsyncVeniceClient, VeniceClient
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


def test_refresh_skips_malformed_entries(tmp_path, httpx2_mock):
    path = tmp_path / "catalog"
    valid = {"id": "m", "type": "text", "model_spec": {"future": {"x": 1}}}
    httpx2_mock.add_response(json={"data": [{"id": "bad", "model_spec": None}, valid]})
    with VeniceClient(key="k") as client:
        assert refresh_catalog(client, path) == [valid]
    assert json.loads(path.read_text()) == [valid]


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


# Model traits and compatibility mappings: /models/traits and /models/compatibility_mapping.

API = "https://api.venice.ai/api/v1"
MAPPINGS = [
    ("list_model_traits", "models/traits", "model_traits", "traits"),
    (
        "get_model_compatibility_mapping",
        "models/compatibility_mapping",
        "model_compatibility",
        "compatibility-mapping",
    ),
]
DATA = {"default": "qwen-3-8-flash", "most_uncensored": "venice-uncensored-1-2"}


def invoke(is_async, method, **kwargs):
    if is_async:

        async def run():
            async with AsyncVeniceClient(key="test") as client:
                return await getattr(client, method)(**kwargs)

        return asyncio.run(run())
    with VeniceClient(key="test") as client:
        return getattr(client, method)(**kwargs)


@pytest.mark.parametrize(
    "path,schema_name",
    [
        ("/models/traits", "ModelTraitSchema"),
        ("/models/compatibility_mapping", "ModelCompatibilitySchema"),
    ],
)
def test_mappings_match_dated_fixture(path, schema_name):
    fixture = json.loads((Path(__file__).parent / "fixtures/openapi-2026-11-22.json").read_text())
    resource = fixture["resources"][f"GET {path}"]
    (parameter,) = resource["parameters"]
    assert (parameter["name"], parameter["in"], parameter["schema"]["default"]) == (
        "type",
        "query",
        "text",
    )
    assert {"text", "image", "embedding", "inpaint"} <= set(parameter["schema"]["enum"])
    schema = resource["response"]["content"]["application/json"]["schema"]
    assert schema["properties"]["data"] == {"$ref": f"#/components/schemas/{schema_name}"}
    mapping = fixture["schemas"][schema_name]
    assert (mapping["type"], mapping["additionalProperties"]) == ("object", {"type": "string"})


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("method,path,operation,command", MAPPINGS)
def test_mapping_clients(is_async, method, path, operation, command, httpx2_mock):
    body = {"object": "list", "type": "text", "data": DATA}
    httpx2_mock.add_response(method="GET", url=f"{API}/{path}", json=body)
    assert invoke(is_async, method) == DATA
    httpx2_mock.add_response(method="GET", url=f"{API}/{path}?type=image", json=body)
    assert invoke(is_async, method, type="image") == DATA
    with pytest.raises(ValidationError):
        invoke(is_async, method, type="")
    assert len(httpx2_mock.get_requests()) == 2


@pytest.mark.parametrize("method,path,operation,command", MAPPINGS)
@pytest.mark.parametrize("data", [None, ["a"], {"trait": 1}, {"trait": None}])
def test_malformed_mappings(method, path, operation, command, data, httpx2_mock):
    httpx2_mock.add_response(url=f"{API}/{path}", json={"object": "list", "data": data})
    with pytest.raises(VeniceResponseError, match="expected data mapping names to model IDs"):
        invoke(False, method)


@pytest.mark.parametrize("method,path,operation,command", MAPPINGS)
def test_mapping_commands(
    method, path, operation, command, httpx2_mock, cli_runner, mock_venice_api_key
):
    body = {"object": "list", "type": "embedding", "data": {"x": "y"}}
    httpx2_mock.add_response(url=f"{API}/{path}?type=embedding", json=body)
    result = cli_runner.invoke(cli, ["venice", command, "--type", "embedding"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == body
    httpx2_mock.add_response(url=f"{API}/{path}", status_code=400, json={"error": "Bad type"})
    result = cli_runner.invoke(cli, ["venice", command])
    assert result.exit_code == 1
    assert "400" in result.output and "Bad type" in result.output


def test_mapping_result_types(httpx2_mock):
    httpx2_mock.add_response(json={"data": DATA}, is_reusable=True)
    with VeniceClient(key="test") as client:
        assert_type(client.list_model_traits(), dict[str, str])
        assert_type(client.get_model_compatibility_mapping(), dict[str, str])
