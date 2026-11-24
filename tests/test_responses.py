"""Venice /responses models: request shaping, routing headers, sync/async parity."""

import asyncio
import json
from pathlib import Path

import llm
import pytest

from llm_venice import AsyncVeniceResponses, VeniceResponses, VeniceWebSearch, VeniceXSearch
from llm_venice.models.responses import _CHAT_ONLY_OPTIONS, _VENICE_PARAMETERS

URL = "https://api.venice.ai/api/v1/responses"
FIXTURE = Path(__file__).parent / "fixtures/openapi-2026-11-22.json"
REASONING_SPEC = {"capabilities": {"supportsReasoning": True}}


def message(text, item_id="msg_1"):
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def payload(*output, status="completed", **extra):
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "test-model",
        "status": status,
        "output": list(output),
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        **extra,
    }


def sse(*events):
    lines = [f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def text_stream(text="Hello", headers=None):
    item = message(text)
    return sse(
        {"type": "response.created", "response": payload(status="in_progress")},
        {"type": "response.output_item.added", "output_index": 0, "item": {**item, "content": []}},
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": text},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": payload(item)},
    )


def add_response(httpx2_mock, *, stream, body=None, headers=None):
    headers = {"x-venice-responses-lane": "compat", **(headers or {})}
    if stream:
        httpx2_mock.add_response(
            method="POST",
            url=URL,
            content=body if body is not None else text_stream(),
            headers={"content-type": "text/event-stream", **headers},
        )
    else:
        httpx2_mock.add_response(
            method="POST", url=URL, json=body or payload(message("Hello")), headers=headers
        )


def run(model, *, stream=True, prompt="hi", **options):
    """Run a prompt to completion; return its text and response."""
    if isinstance(model, AsyncVeniceResponses):

        async def go():
            response = model.prompt(prompt, stream=stream, **options)
            return await response.text(), response

        return asyncio.run(go())
    response = model.prompt(prompt, stream=stream, **options)
    return response.text(), response


def json_of(response) -> dict:
    data = response.response_json
    assert isinstance(data, dict)
    return data


def make(is_async, model_id="test-model", **kwargs):
    return (AsyncVeniceResponses if is_async else VeniceResponses)(model_id, **kwargs)


@pytest.fixture
def sent(request_json):
    return request_json


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_requests_are_stateless_and_carry_the_bare_model(
    is_async, stream, httpx2_mock, mock_venice_api_key, sent
):
    add_response(httpx2_mock, stream=stream)
    model = make(is_async, "venice/test-model")
    text, response = run(model, stream=stream, max_tokens=50)
    assert text == "Hello"
    assert str(model) == "Venice Responses: venice/test-model"
    body = sent()
    assert body["model"] == "test-model"
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert body["store"] is False and "previous_response_id" not in body
    assert body["max_output_tokens"] == 50
    assert body.get("stream", False) is stream
    request = httpx2_mock.get_request()
    assert request.headers["authorization"] == "Bearer fake-venice-api-key"
    assert "x-venice-responses-mode" not in request.headers
    assert json_of(response)["venice_headers"] == {"x-venice-responses-lane": "compat"}


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_mode_header_and_routing_headers(is_async, stream, httpx2_mock, mock_venice_api_key):
    add_response(
        httpx2_mock,
        stream=stream,
        headers={"x-venice-responses-compat-reason": "web_search", "x-other": "x"},
    )
    _, response = run(make(is_async), stream=stream, responses_api="compat")
    assert httpx2_mock.get_request().headers["x-venice-responses-mode"] == "compat"
    assert json_of(response)["venice_headers"] == {
        "x-venice-responses-lane": "compat",
        "x-venice-responses-compat-reason": "web_search",
    }


@pytest.mark.parametrize("is_async", [False, True])
def test_venice_fields_are_sent(is_async, httpx2_mock, mock_venice_api_key, sent):
    add_response(httpx2_mock, stream=False)
    run(
        make(is_async, supports_web_search=True),
        stream=False,
        anon_user_id="end-user",
        fallbacks='["venice/claude-opus-4-8"]',
        prompt_cache_retention="24h",
        include_venice_system_prompt=False,
        character_slug="alan-watts",
        enable_web_search="auto",
        enable_web_citations=True,
        extra_body='{"parallel_tool_calls": false, "venice_parameters": {"new_flag": 1}}',
    )
    body = sent()
    assert body["anon_user_id"] == "end-user"
    assert body["fallbacks"] == [{"model": "claude-opus-4-8"}]
    assert body["prompt_cache_retention"] == "24h"
    assert body["parallel_tool_calls"] is False
    assert body["venice_parameters"] == {
        "new_flag": 1,
        "include_venice_system_prompt": False,
        "character_slug": "alan-watts",
        "enable_web_search": "auto",
        "enable_web_citations": True,
    }


@pytest.mark.parametrize("is_async", [False, True])
def test_reasoning_settings(is_async, httpx2_mock, mock_venice_api_key, sent):
    add_response(httpx2_mock, stream=False)
    add_response(httpx2_mock, stream=False)
    add_response(httpx2_mock, stream=False)
    # A reasoning model asks for summaries and encrypted reasoning, as llm does.
    run(make(is_async, model_spec=REASONING_SPEC), stream=False)
    assert sent()["reasoning"] == {"summary": "auto"}
    assert sent()["include"] == ["reasoning.encrypted_content"]
    run(
        make(is_async, model_spec=REASONING_SPEC),
        stream=False,
        reasoning='{"effort": "low", "summary": "detailed"}',
        reasoning_effort="high",
    )
    # reasoning_effort takes precedence over reasoning.effort, as in chat.
    assert sent()["reasoning"] == {"summary": "detailed", "effort": "high"}
    run(make(is_async), stream=False, reasoning='{"enabled": false}')
    assert sent()["reasoning"] == {"enabled": False}
    assert "include" not in sent()


@pytest.mark.parametrize("is_async", [False, True])
def test_web_search_tool_becomes_venice_parameters(
    is_async, httpx2_mock, mock_venice_api_key, sent
):
    add_response(httpx2_mock, stream=False)
    model = make(is_async, supports_web_search=True)
    assert model.supported_server_side_tools == (VeniceWebSearch,)
    run(model, stream=False, tools=[VeniceWebSearch(mode="on")])
    body = sent()
    assert "tools" not in body
    assert body["venice_parameters"] == {
        "enable_web_search": "on",
        "enable_web_citations": True,
        "enable_web_scraping": False,
    }


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "kwargs,options,message",
    [
        ({}, {"min_p": 0.1, "seed": 1}, "does not take seed, min_p"),
        ({}, {"verbosity": "low"}, "does not take verbosity"),
        ({}, {"enable_web_search": "on"}, "does not support web search"),
        (
            {},
            {"extra_body": '{"input": "x"}'},
            "extra_body conflicts with fields set by the request: input",
        ),
        (
            {"model_spec": {"capabilities": {"supportsE2EE": True}}},
            {},
            "E2EE model, which the Venice Responses API does not serve",
        ),
        ({}, {"responses_api": "fast"}, "responses_api"),
    ],
)
def test_requests_that_cannot_be_served_fail_first(
    is_async, kwargs, options, message, httpx2_mock, mock_venice_api_key
):
    with pytest.raises((llm.ModelError, ValueError), match=message):
        run(make(is_async, **kwargs), stream=False, **options)
    assert not httpx2_mock.get_requests()


def test_x_search_is_refused(httpx2_mock, mock_venice_api_key):
    model = make(False)
    with pytest.raises(ValueError, match="does not support server-side tool 'venice_x_search'"):
        run(model, stream=False, tools=[VeniceXSearch()])
    assert not httpx2_mock.get_requests()


def test_options_match_dated_fixture():
    schema = json.loads(FIXTURE.read_text())["schemas"]["ResponsesRequest"]
    properties = schema["properties"]
    for name in ("anon_user_id", "fallbacks", "prompt_cache_retention", "venice_parameters"):
        assert name in properties
    assert set(_VENICE_PARAMETERS) <= set(properties["venice_parameters"]["properties"])
    assert "enable_x_search" not in properties["venice_parameters"]["properties"]
    assert {"default", "extended", "24h"} <= set(properties["prompt_cache_retention"]["enum"])
    assert properties["fallbacks"]["items"]["required"] == ["model"]
    assert set(properties["reasoning"]["properties"]) == {"effort", "summary", "enabled"}
    # Chat-only options have no field in the Responses request.
    wire = {"stop_token_ids", "min_p", "top_k", "repetition_penalty", "logit_bias", "seed"}
    assert wire <= set(_CHAT_ONLY_OPTIONS)
    assert not wire & set(properties)


@pytest.mark.parametrize("is_async", [False, True])
def test_options_are_the_chat_options_plus_the_mode(is_async):
    options = make(is_async).Options
    assert {"responses_api", "min_p", "character_slug", "anon_user_id"} <= set(options.model_fields)
    assert options.model_validate({"responses_api": "native"}).model_dump()["responses_api"] == (
        "native"
    )
