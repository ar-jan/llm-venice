"""Embeddings through /embeddings: request contract and clients."""

import asyncio
import json
from pathlib import Path

import httpx2
import pytest
from pydantic import ValidationError
from typing_extensions import assert_type

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
