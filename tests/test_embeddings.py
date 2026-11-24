"""Embeddings through /embeddings: request contract, clients and the llm adapter."""

import asyncio
import json
from pathlib import Path

import httpx2
import llm
import pytest
from llm.cli import cli
from pydantic import ValidationError
from typing_extensions import assert_type

from llm_venice import VeniceEmbeddingModel
from llm_venice.models import _common, register_venice_embedding_models
from llm_venice._endpoints import EMBEDDING_INPUT_LIMIT, EmbeddingInput, parse_embeddings
from llm_venice.client import AsyncVeniceClient, VeniceClient
from llm_venice.errors import VeniceAPIError, VeniceResponseError
from llm_venice.types import EmbeddingData, EmbeddingResponse, EmbeddingUsage

URL = "https://api.venice.ai/api/v1/embeddings"
FIXTURE = Path(__file__).parent / "fixtures/openapi-2026-11-22.json"


def embedding(index, vector=(0.5, -1.0)):
    return {"object": "embedding", "index": index, "embedding": list(vector)}


def payload(*items):
    return {
        "object": "list",
        "model": "text-embedding-bge-m3",
        "data": list(items),
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }


def invoke(is_async, *args, **kwargs):
    if is_async:

        async def run():
            async with AsyncVeniceClient(key="test") as client:
                return await client.create_embeddings(*args, **kwargs)

        return asyncio.run(run())
    with VeniceClient(key="test") as client:
        return client.create_embeddings(*args, **kwargs)


def parse(body, count):
    response = httpx2.Response(200, json=body, request=httpx2.Request("POST", URL))
    return parse_embeddings(response, count)


def test_request_body_omits_unset_options():
    assert EmbeddingInput(model="m", input="hi").body() == {"model": "m", "input": "hi"}
    body = EmbeddingInput.model_validate(
        {"model": "m", "input": ("a", ""), "dimensions": 256, "encoding_format": "base64"}
    ).body()
    # Venice embeds an empty string inside a list; only a lone empty string is rejected.
    assert body == {
        "model": "m",
        "input": ["a", ""],
        "dimensions": 256,
        "encoding_format": "base64",
    }
    assert EmbeddingInput(model="m", input="hi").count() == 1
    assert EmbeddingInput(model="m", input=["a", "b"]).count() == 2


@pytest.mark.parametrize(
    "inputs",
    [
        {"model": "m", "input": ""},
        {"model": "m", "input": []},
        {"model": "m", "input": ["x"] * (EMBEDDING_INPUT_LIMIT + 1)},
        {"model": "m", "input": b"bytes"},
        {"model": "m", "input": ["a", b"b"]},
        {"model": "m", "input": ["a", 1]},
        {"model": "", "input": "hi"},
        {"model": "m", "input": "hi", "dimensions": 0},
        {"model": "m", "input": "hi", "dimensions": "256"},
        {"model": "m", "input": "hi", "encoding_format": "int8"},
        {"model": "m", "input": "hi", "user": "u"},
    ],
)
def test_request_limits(inputs):
    with pytest.raises(ValidationError):
        EmbeddingInput.model_validate(inputs)


@pytest.mark.parametrize("tokens", [[1, 2, 3], [[1, 2], [3]], (4, 5)])
def test_token_arrays_are_rejected_by_name(tokens):
    with pytest.raises(ValidationError, match="token arrays are not supported"):
        EmbeddingInput(model="m", input=tokens)


def test_request_matches_dated_fixture():
    fixture = json.loads(FIXTURE.read_text())
    schema = fixture["schemas"]["CreateEmbeddingRequestSchema"]
    properties = schema["properties"]
    body = EmbeddingInput(model="m", input=["a"], dimensions=1, encoding_format="float").body()
    # `user` is accepted by Venice but discarded, so it is not sent.
    assert set(body) == set(properties) - {"user"}
    assert set(schema["required"]) <= set(body)
    text, array = properties["input"]["anyOf"]
    assert (text["type"], text["minLength"]) == ("string", 1)
    assert (array["minItems"], array["maxItems"]) == (1, EMBEDDING_INPUT_LIMIT)
    assert set(properties["encoding_format"]["enum"]) == {"float", "base64"}
    assert properties["dimensions"]["minimum"] == 1
    resource = fixture["resources"]["POST /embeddings"]
    response = resource["response"]["content"]["application/json"]["schema"]
    item = response["properties"]["data"]["items"]
    assert set(EmbeddingResponse.__annotations__) == set(response["properties"])
    assert EmbeddingResponse.__required_keys__ <= set(response["required"])
    assert set(EmbeddingData.__annotations__) == set(item["properties"])
    assert EmbeddingData.__required_keys__ <= set(item["required"])
    assert set(EmbeddingUsage.__annotations__) == set(response["properties"]["usage"]["properties"])


def test_response_is_returned_in_input_order_with_unknown_fields():
    body = {**payload(embedding(1, (2,)), embedding(0, (1.5,))), "future": True}
    result = parse(body, 2)
    assert [item["index"] for item in result["data"]] == [0, 1]
    assert result["data"][0]["embedding"] == [1.5]
    assert dict(result)["future"] is True
    # Base64 embeddings are passed through as text.
    encoded = {"object": "embedding", "index": 0, "embedding": "AACAPw=="}
    assert parse(payload(encoded), 1)["data"][0]["embedding"] == "AACAPw=="


@pytest.mark.parametrize(
    "body,count,detail",
    [
        (payload(embedding(0)), 2, r"expected 2 embedding\(s\), one per input, but received 1"),
        (payload(embedding(0), embedding(1)), 1, "but received 2"),
        (payload(embedding(0), embedding(0)), 2, "indexes do not match"),
        (payload(embedding(1)), 1, "indexes do not match"),
        (payload({"index": 0}), 1, "invalid EmbeddingResponse: data"),
        (payload({"index": 0, "embedding": ["x"]}), 1, "invalid EmbeddingResponse: data"),
        (payload({"index": "0", "embedding": [1.0]}), 1, "invalid EmbeddingResponse: data"),
        ({"object": "list"}, 1, "invalid EmbeddingResponse: data"),
        ([embedding(0)], 1, "expected a JSON object"),
    ],
)
def test_malformed_responses(body, count, detail):
    with pytest.raises(VeniceResponseError, match=detail):
        parse(body, count)


@pytest.mark.parametrize("is_async", [False, True])
def test_client_parity(is_async, httpx2_mock):
    httpx2_mock.add_response(
        method="POST",
        url=URL,
        match_json={"model": "m", "input": ["a", "b"], "dimensions": 2},
        json=payload(embedding(0), embedding(1)),
    )
    result = invoke(is_async, "m", ["a", "b"], dimensions=2)
    assert_type(result, EmbeddingResponse)
    assert [item["index"] for item in result["data"]] == [0, 1]
    request = httpx2_mock.get_request()
    assert request.headers["authorization"] == "Bearer test"
    assert request.headers["content-type"] == "application/json"


@pytest.mark.parametrize("is_async", [False, True])
def test_client_sends_a_string_as_one_input(is_async, httpx2_mock):
    httpx2_mock.add_response(
        method="POST",
        url=URL,
        match_json={"model": "m", "input": "hello", "encoding_format": "base64"},
        json=payload({"object": "embedding", "index": 0, "embedding": "AACAPw=="}),
    )
    result = invoke(is_async, "m", "hello", encoding_format="base64")
    assert result["data"][0]["embedding"] == "AACAPw=="


@pytest.mark.parametrize("is_async", [False, True])
def test_client_validation_precedes_http(is_async, httpx2_mock):
    with pytest.raises(ValidationError, match="token arrays"):
        invoke(is_async, "m", [1, 2, 3])
    with pytest.raises(ValidationError):
        invoke(is_async, "m", "")
    assert not httpx2_mock.get_requests()


@pytest.mark.parametrize("is_async", [False, True])
def test_client_errors(is_async, httpx2_mock):
    httpx2_mock.add_response(
        url=URL, status_code=404, json={"error": "Specified model not found: nope."}
    )
    with pytest.raises(VeniceAPIError, match="Creating embeddings failed: .*404.*not found"):
        invoke(is_async, "nope", "hi")
    httpx2_mock.add_response(url=URL, json=payload())
    with pytest.raises(VeniceResponseError, match="received 0"):
        invoke(is_async, "m", "hi")


# The llm adapter: registration, `llm embed` and dimension variants.

EMBEDDING_SPEC = {
    "embeddingDimensions": 4,
    "maxInputTokens": 8192,
    "supportsCustomDimensions": True,
}


@pytest.fixture
def embedding_catalog(hermetic_venice_catalog, monkeypatch):
    monkeypatch.delenv("LLM_VENICE_KEY", raising=False)
    hermetic_venice_catalog(
        [
            {"id": "emb", "type": "embedding", "model_spec": EMBEDDING_SPEC},
            {"id": "fixed", "type": "embedding", "model_spec": {"embeddingDimensions": 4}},
            {"id": "gone", "type": "embedding", "model_spec": {"offline": True}},
            {"id": "chat", "type": "text", "model_spec": {}},
        ]
    )


def registered(model_id):
    model = llm.get_embedding_model(model_id)
    assert isinstance(model, VeniceEmbeddingModel)
    return model


def respond(httpx2_mock, *vectors, **match):
    httpx2_mock.add_response(
        method="POST",
        url=URL,
        json=payload(*(embedding(index, vector) for index, vector in enumerate(vectors))),
        **({"match_json": match} if match else {}),
    )


def test_registers_online_embedding_models_only(embedding_catalog):
    ids = {model.model_id for model in llm.get_embedding_models()}
    assert {"venice/emb", "venice/fixed"} <= ids
    assert not {"venice/gone", "venice/chat"} & ids
    assert "venice/emb" not in {model.model_id for model in llm.get_models()}
    model = registered("venice/emb")
    assert (model.model_name, model.dimensions, model.batch_size) == ("emb", None, 100)
    assert str(model) == "Venice Embedding: venice/emb"
    assert model.model_spec == EMBEDDING_SPEC


def test_registration_without_catalog_or_key_registers_nothing(hermetic_llm_user_path, monkeypatch):
    monkeypatch.delenv("LLM_VENICE_KEY", raising=False)
    registered = []
    register_venice_embedding_models(lambda model, aliases=None: registered.append(model))
    assert registered == []


def test_embed_sends_the_catalog_model_and_returns_floats(embedding_catalog, httpx2_mock):
    respond(httpx2_mock, (1, 0.5, -2, 0), model="emb", input=["hello"])
    vector = llm.get_embedding_model("venice/emb").embed("hello", key="k")
    assert vector == [1.0, 0.5, -2.0, 0.0]
    assert all(type(value) is float for value in vector)
    assert httpx2_mock.get_request().headers["authorization"] == "Bearer k"


def test_embed_multi_batches_by_100(embedding_catalog, httpx2_mock):
    texts = [f"text {index}" for index in range(150)]
    respond(httpx2_mock, *[(index,) for index in range(100)], model="emb", input=texts[:100])
    respond(httpx2_mock, *[(index,) for index in range(50)], model="emb", input=texts[100:])
    vectors = list(llm.get_embedding_model("venice/emb").embed_multi(texts, key="k"))
    assert len(vectors) == 150
    assert vectors[99] == [99.0] and vectors[100] == [0.0]
    assert len(httpx2_mock.get_requests()) == 2


def test_empty_batch_sends_no_request(embedding_catalog, httpx2_mock):
    assert list(llm.get_embedding_model("venice/emb").embed_multi([], key="k")) == []
    assert not httpx2_mock.get_requests()


def test_dimension_variant_has_its_own_id_and_checks_the_size(embedding_catalog, httpx2_mock):
    spec = registered("venice/emb").model_spec
    model = VeniceEmbeddingModel("emb", model_spec=spec, dimensions=2)
    assert (model.model_id, model.model_name) == ("venice/emb-2", "emb")
    respond(httpx2_mock, (1, 2), model="emb", input=["hi"], dimensions=2)
    assert model.embed("hi", key="k") == [1.0, 2.0]
    # Venice answers an unsupported size with the native size; a collection needs one size.
    respond(httpx2_mock, (1, 2, 3, 4))
    with pytest.raises(llm.ModelError, match="requested 2 dimensions but received 4"):
        model.embed("hi", key="k")


@pytest.mark.parametrize(
    "model_id,dimensions,message",
    [
        ("fixed", 2, "venice/fixed does not support custom dimensions"),
        ("emb", 5, "at most 4, the native size of venice/emb"),
        ("emb", 0, "positive integer"),
        ("emb", True, "positive integer"),
    ],
)
def test_dimensions_are_checked_against_the_catalog(
    embedding_catalog, model_id, dimensions, message
):
    spec = registered(f"venice/{model_id}").model_spec
    with pytest.raises(ValueError, match=message):
        VeniceEmbeddingModel(model_id, model_spec=spec, dimensions=dimensions)


def test_dimensions_without_catalog_metadata_are_left_to_venice(httpx2_mock):
    model = VeniceEmbeddingModel("unknown", dimensions=8)
    assert model.model_id == "venice/unknown-8"
    respond(httpx2_mock, (0,) * 8, model="unknown", input=["hi"], dimensions=8)
    assert len(model.embed("hi", key="k")) == 8


def test_errors_become_model_errors(embedding_catalog, httpx2_mock):
    model = llm.get_embedding_model("venice/emb")
    httpx2_mock.add_response(url=URL, status_code=400, json={"error": "Input too long"})
    with pytest.raises(llm.ModelError, match="Creating embeddings failed: .*400.*Input too long"):
        model.embed("hi", key="k")
    httpx2_mock.add_response(
        url=URL, json=payload({"object": "embedding", "index": 0, "embedding": "AACAPw=="})
    )
    with pytest.raises(llm.ModelError, match="expected a list of floats"):
        model.embed("hi", key="k")
    with pytest.raises(ValueError, match="does not support binary data"):
        model.embed(b"bytes", key="k")


def test_missing_key_raises_needs_key(embedding_catalog, monkeypatch, httpx2_mock):
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)
    with pytest.raises(llm.NeedsKeyException):
        llm.get_embedding_model("venice/emb").embed("hi")
    assert not httpx2_mock.get_requests()


def test_deprecated_model_warns_once(hermetic_venice_catalog, httpx2_mock, monkeypatch, capsys):
    monkeypatch.setattr(_common, "_deprecated_warnings", set())
    deprecation = {"date": "2026-12-01", "replacementModelId": "new-emb"}
    hermetic_venice_catalog(
        [{"id": "old-emb", "type": "embedding", "model_spec": {"deprecation": deprecation}}]
    )
    model = llm.get_embedding_model("venice/old-emb")
    respond(httpx2_mock, (1,))
    respond(httpx2_mock, (2,))
    model.embed("a", key="k")
    model.embed("b", key="k")
    assert capsys.readouterr().err == (
        "Warning: venice/old-emb is deprecated; use venice/new-emb instead\n"
    )


def test_cli_embed_store_and_similar(
    embedding_catalog, httpx2_mock, cli_runner, mock_venice_api_key
):
    respond(httpx2_mock, (1, 0, 0, 0), model="emb", input=["about cats"])
    respond(httpx2_mock, (0, 1, 0, 0), model="emb", input=["about dogs"])
    for item_id, text in (("cats", "about cats"), ("dogs", "about dogs")):
        result = cli_runner.invoke(cli, ["embed", "pets", item_id, "-m", "venice/emb", "-c", text])
        assert result.exit_code == 0, result.output
    respond(httpx2_mock, (0.9, 0.1, 0, 0), model="emb", input=["kittens"])
    result = cli_runner.invoke(cli, ["similar", "pets", "-c", "kittens", "-n", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["id"] == "cats"
