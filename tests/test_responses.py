"""Venice /responses: requests, routing headers, compat-lane streams, chat delegation."""

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import llm
import pytest
from llm.parts import ReasoningPart

from llm.cli import cli

from llm_venice import (
    AsyncVeniceChat,
    AsyncVeniceResponses,
    VeniceChat,
    VeniceResponses,
    VeniceWebSearch,
    VeniceXSearch,
)
from llm_venice.models.responses import (
    _CHAT_ONLY_OPTIONS,
    _VENICE_PARAMETERS,
    _events,
    _translate,
)

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
    if isinstance(model, (AsyncVeniceResponses, AsyncVeniceChat)):

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


# Venice's compat lane and final statuses.

REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_1",
    "status": "completed",
    "summary": ["Thinking about it."],
}


def messages_of(response):
    if isinstance(response, llm.AsyncResponse):
        return asyncio.run(response.messages())
    return response.messages()


def compat_stream(*, final="response.completed", status="completed", output=None):
    item = message("Hi")
    output = output if output is not None else [REASONING_ITEM, item]
    extra = (
        {"incomplete_details": {"reason": "max_output_tokens"}} if status == "incomplete" else {}
    )
    return sse(
        {"type": "response.created", "response": payload(status="in_progress")},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs_1", "status": "in_progress"},
        },
        {"type": "response.reasoning.delta", "item_id": "rs_1", "delta": "Thinking "},
        {"type": "response.reasoning.delta", "item_id": "rs_1", "delta": "about it."},
        {"type": "response.output_item.done", "output_index": 0, "item": REASONING_ITEM},
        {"type": "response.output_item.added", "output_index": 1, "item": {**item, "content": []}},
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hi"},
        {"type": "response.output_item.done", "output_index": 1, "item": item},
        {"type": final, "response": payload(*output, status=status, **extra)},
    )


@pytest.mark.parametrize("is_async", [False, True])
def test_compat_stream_reasoning_and_text(is_async, httpx2_mock, mock_venice_api_key):
    add_response(httpx2_mock, stream=True, body=compat_stream())
    text, response = run(make(is_async, model_spec=REASONING_SPEC))
    assert text == "Hi"
    reasoning = [
        part
        for message_ in messages_of(response)
        for part in message_.parts
        if isinstance(part, ReasoningPart)
    ]
    # The streamed text, then the item's metadata; the text is not repeated.
    assert "".join(part.text for part in reasoning) == "Thinking about it."
    meta = [part.provider_metadata for part in reasoning if part.provider_metadata]
    assert meta == [
        {
            "openai": {
                "id": "rs_1",
                "summary": [{"text": "Thinking about it.", "type": "summary_text"}],
            }
        }
    ]
    data = json_of(response)
    assert data["output"][0]["summary"] == [{"text": "Thinking about it.", "type": "summary_text"}]
    assert (response.input_tokens, response.output_tokens) == (5, 3)


@pytest.mark.parametrize("is_async", [False, True])
def test_compat_summaries_in_a_full_response(is_async, httpx2_mock, mock_venice_api_key):
    add_response(httpx2_mock, stream=False, body=payload(REASONING_ITEM, message("Hi")))
    text, response = run(make(is_async, model_spec=REASONING_SPEC), stream=False)
    assert text == "Hi"
    reasoning = [
        part
        for message_ in messages_of(response)
        for part in message_.parts
        if isinstance(part, ReasoningPart)
    ]
    assert [part.text for part in reasoning] == ["Thinking about it."]
    assert not reasoning[0].redacted


def test_compat_reasoning_is_replayed_on_the_next_turn(httpx2_mock, mock_venice_api_key, sent):
    add_response(httpx2_mock, stream=True, body=compat_stream())
    add_response(httpx2_mock, stream=False)
    model = make(False, model_spec=REASONING_SPEC)
    conversation = model.conversation()
    conversation.prompt("first").text()
    conversation.prompt("second").text()
    replayed = sent()["input"]
    assert replayed[0] == {"role": "user", "content": "first"}
    assert {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"text": "Thinking about it.", "type": "summary_text"}],
    } in replayed
    assert replayed[-2:] == [
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "second"},
    ]
    assert "previous_response_id" not in sent()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_incomplete_responses_keep_their_status_output_and_usage(
    is_async, stream, httpx2_mock, mock_venice_api_key
):
    if stream:
        body = compat_stream(final="response.incomplete", status="incomplete")
    else:
        body = payload(
            message("Hi"), status="incomplete", incomplete_details={"reason": "content_filter"}
        )
    add_response(httpx2_mock, stream=stream, body=body)
    text, response = run(make(is_async), stream=stream)
    assert text == "Hi"
    data = json_of(response)
    assert data["status"] == "incomplete"
    assert data["incomplete_details"]["reason"] in ("max_output_tokens", "content_filter")
    assert response.output_tokens == 3
    assert data["venice_headers"] == {"x-venice-responses-lane": "compat"}


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "event,message",
    [
        (
            {
                "type": "response.failed",
                "response": payload(
                    status="failed", error={"code": "server_error", "message": "Upstream failed"}
                ),
            },
            "Venice response failed: Upstream failed (server_error)",
        ),
        (
            {"type": "error", "code": "rate_limited", "message": "Slow down", "param": None},
            "Venice response failed: Slow down (rate_limited)",
        ),
    ],
)
def test_failed_streams_raise(is_async, event, message, httpx2_mock, mock_venice_api_key):
    body = sse({"type": "response.created", "response": payload(status="in_progress")}, event)
    add_response(httpx2_mock, stream=True, body=body)
    with pytest.raises(llm.ModelError, match=re.escape(message)):
        run(make(is_async))


@pytest.mark.parametrize("is_async", [False, True])
def test_failed_non_streaming_response_raises(is_async, httpx2_mock, mock_venice_api_key):
    body = payload(status="failed", error={"code": "server_error", "message": "Upstream failed"})
    add_response(httpx2_mock, stream=False, body=body)
    with pytest.raises(llm.ModelError, match=re.escape("Upstream failed (server_error)")):
        run(make(is_async), stream=False)


def test_events_close_the_stream_when_abandoned():
    closed = []

    class Stream:
        def __iter__(self):
            yield SimpleNamespace(type="response.created")
            yield SimpleNamespace(type="response.output_text.delta", delta="x")

        def close(self):
            closed.append(True)

    events = _events(Stream())
    next(events)
    events.close()
    assert closed == [True]


def test_translation_leaves_other_events_alone():
    event = SimpleNamespace(type="response.web_search.done", item_id="ws_1")
    assert _translate(event) is event
    delta = _translate(SimpleNamespace(type="response.reasoning.delta", delta=None))
    assert (delta.type, delta.delta, delta.item_id) == ("response.reasoning_text.delta", "", None)


# Chat models hand a prompt to /responses with the responses_api option.

CHAT_URL = "https://api.venice.ai/api/v1/chat/completions"


def chat_completion(text):
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def chat_model(is_async, **kwargs):
    return (AsyncVeniceChat if is_async else VeniceChat)("venice/test-model", **kwargs)


def test_responses_models_take_the_chat_options():
    assert VeniceResponses.Options is VeniceChat.Options
    assert AsyncVeniceResponses.Options is VeniceChat.Options
    assert "responses_api" in VeniceChat.Options.model_fields


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_chat_models_delegate_with_responses_api(
    is_async, stream, httpx2_mock, mock_venice_api_key, sent
):
    add_response(httpx2_mock, stream=stream, headers={"x-venice-responses-lane": "native"})
    model = chat_model(is_async, model_spec=REASONING_SPEC)
    text, response = run(model, stream=stream, responses_api="native", reasoning_effort="low")
    assert text == "Hello"
    request = httpx2_mock.get_request()
    assert str(request.url) == URL
    assert request.headers["x-venice-responses-mode"] == "native"
    assert sent()["model"] == "test-model"
    assert sent()["reasoning"] == {"summary": "auto", "effort": "low"}
    assert json_of(response)["venice_headers"] == {"x-venice-responses-lane": "native"}
    assert response.model is model


@pytest.mark.parametrize("is_async", [False, True])
def test_chat_models_keep_chat_completions_by_default(is_async, httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(
        method="POST",
        url=CHAT_URL,
        json=chat_completion("Chat"),
    )
    text, _ = run(chat_model(is_async), stream=False)
    assert text == "Chat"
    assert str(httpx2_mock.get_request().url) == CHAT_URL


@pytest.mark.parametrize("is_async", [False, True])
def test_delegated_prompts_are_checked_for_the_responses_api(
    is_async, httpx2_mock, mock_venice_api_key
):
    with pytest.raises(llm.ModelError, match="does not take min_p"):
        run(chat_model(is_async), stream=False, responses_api="auto", min_p=0.2)
    model = chat_model(is_async, supports_x_search=True)
    with pytest.raises(llm.ModelError, match="venice_x_search is not available"):
        run(model, stream=False, responses_api="auto", tools=[VeniceXSearch()])
    assert not httpx2_mock.get_requests()


def test_delegated_web_search_tool(httpx2_mock, mock_venice_api_key, sent):
    add_response(httpx2_mock, stream=False)
    model = chat_model(False, supports_web_search=True)
    run(model, stream=False, responses_api="compat", tools=[VeniceWebSearch()])
    assert sent()["venice_parameters"]["enable_web_search"] == "auto"


@pytest.mark.parametrize("is_async", [False, True])
def test_responses_reject_audio_before_request(is_async, httpx2_mock, mock_venice_api_key):
    audio = llm.Attachment(content=b"audio", type="audio/wav")
    with pytest.raises(llm.ModelError, match="audio/wav.*use Chat Completions"):
        run(chat_model(is_async, audio=True), attachments=[audio], responses_api="compat")
    assert not httpx2_mock.get_requests()


def test_switching_to_responses_checks_earlier_attachments(httpx2_mock, mock_venice_api_key):
    httpx2_mock.add_response(method="POST", url=CHAT_URL, json=chat_completion("One"))
    conversation = chat_model(False, audio=True).conversation()
    audio = llm.Attachment(content=b"audio", type="audio/wav")
    conversation.prompt("first", attachments=[audio], stream=False).text()
    with pytest.raises(llm.ModelError, match="audio/wav.*use Chat Completions"):
        conversation.prompt("second", responses_api="auto").text()
    assert len(httpx2_mock.get_requests()) == 1


def test_a_conversation_can_switch_to_responses(httpx2_mock, mock_venice_api_key, sent):
    httpx2_mock.add_response(
        method="POST",
        url=CHAT_URL,
        json=chat_completion("One"),
    )
    add_response(httpx2_mock, stream=False)
    conversation = chat_model(False).conversation()
    conversation.prompt("first", stream=False).text()
    conversation.prompt("second", stream=False, responses_api="auto").text()
    assert sent()["input"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "One"},
        {"role": "user", "content": "second"},
    ]


def test_cli_option(cli_runner, httpx2_mock, mock_venice_api_key, hermetic_venice_catalog, sent):
    hermetic_venice_catalog()
    add_response(httpx2_mock, stream=True)
    result = cli_runner.invoke(
        cli, ["-m", "venice/test-model", "-o", "responses_api", "compat", "hi"], input=""
    )
    assert result.exit_code == 0, result.output
    assert result.output == "Hello\n"
    assert httpx2_mock.get_request().headers["x-venice-responses-mode"] == "compat"
    result = cli_runner.invoke(
        cli, ["-m", "venice/test-model", "-o", "responses_api", "fast", "hi"], input=""
    )
    assert result.exit_code == 1
    assert "responses_api" in result.output
