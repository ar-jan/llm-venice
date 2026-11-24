"""Chat provider metadata sent back on later turns, and cache_control breakpoints."""

import asyncio
import json

import llm
import pytest
from llm.parts import ReasoningPart

from llm_venice import AsyncVeniceChat, VeniceChat

CHAT_URL = "https://api.venice.ai/api/v1/chat/completions"
SIGNATURE = "AY89a1-signature"
EXTRA = {"google": {"thought_signature": SIGNATURE}}
DETAILS = [{"type": "reasoning.encrypted", "data": "abc", "index": 0}]


def completion(content: str | None = "One", **message):
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "gemini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, **message},
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def chunk(**delta):
    return {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gemini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


def add_stream(httpx2_mock, *chunks):
    body = "".join(f"data: {json.dumps(item)}\n\n" for item in chunks) + "data: [DONE]\n\n"
    httpx2_mock.add_response(
        method="POST",
        url=CHAT_URL,
        content=body.encode(),
        headers={"content-type": "text/event-stream"},
    )


def add_completion(httpx2_mock, body):
    httpx2_mock.add_response(method="POST", url=CHAT_URL, json=body)


def make(is_async, model_id="venice/gemini", **kwargs):
    return (AsyncVeniceChat if is_async else VeniceChat)(model_id, **kwargs)


def converse(model, *prompts, stream=False, **options):
    """Run prompts as one conversation; return the responses."""
    if isinstance(model, AsyncVeniceChat):

        async def go():
            conversation = model.conversation()
            responses = []
            for text in prompts:
                response = conversation.prompt(text, stream=stream, **options)
                await response.text()
                responses.append(response)
            return responses

        return asyncio.run(go())
    conversation = model.conversation()
    responses = []
    for text in prompts:
        response = conversation.prompt(text, stream=stream, **options)
        response.text()
        responses.append(response)
    return responses


def assistant(request_json, index=-1):
    return next(m for m in request_json(index)["messages"] if m["role"] == "assistant")


@pytest.mark.parametrize("is_async", [False, True])
def test_message_fields_are_sent_back(is_async, httpx2_mock, mock_venice_api_key, request_json):
    add_completion(
        httpx2_mock,
        completion(thought_signature=SIGNATURE, extra_content=EXTRA, reasoning_details=DETAILS),
    )
    add_completion(httpx2_mock, completion("Two"))
    converse(make(is_async), "first", "second")
    assert assistant(request_json) == {
        "role": "assistant",
        "content": "One",
        "thought_signature": SIGNATURE,
        "extra_content": EXTRA,
        "reasoning_details": DETAILS,
    }


@pytest.mark.parametrize("is_async", [False, True])
def test_streamed_fields_are_sent_back(is_async, httpx2_mock, mock_venice_api_key, request_json):
    add_stream(
        httpx2_mock,
        chunk(role="assistant", content="O"),
        chunk(content="ne", thought_signature=SIGNATURE, extra_content=EXTRA),
    )
    add_completion(httpx2_mock, completion("Two"))
    responses = converse(make(is_async), "first", "second", stream=True)
    sent = assistant(request_json)
    assert (sent["content"], sent["thought_signature"], sent["extra_content"]) == (
        "One",
        SIGNATURE,
        EXTRA,
    )
    assert "reasoning_details" not in sent
    messages = asyncio.run(responses[0].messages()) if is_async else responses[0].messages()
    (part,) = [p for p in messages[0].parts if isinstance(p, ReasoningPart)]
    assert (part.text, part.redacted) == ("", False)
    assert part.provider_metadata == {
        "venice": {
            "model": "gemini",
            "message": {"thought_signature": SIGNATURE, "extra_content": EXTRA},
        }
    }


def tool_call(call_id, city, signature=None):
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": "weather", "arguments": json.dumps({"city": city})},
    }
    return {**call, "thought_signature": signature} if signature else call


def weather(city: str) -> str:
    "Weather for a city"
    return f"Sunny in {city}"


@pytest.mark.parametrize("stream", [False, True])
def test_tool_call_signatures_are_sent_back(stream, httpx2_mock, mock_venice_api_key, request_json):
    if stream:
        add_stream(
            httpx2_mock,
            chunk(
                role="assistant",
                tool_calls=[
                    {
                        **tool_call("call_1", "Paris", SIGNATURE),
                        "index": 0,
                        "function": {"name": "weather", "arguments": ""},
                    }
                ],
            ),
            chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"city": "Paris"}'}}]),
            chunk(
                tool_calls=[
                    {
                        **tool_call("call_2", "Rome"),
                        "index": 1,
                        "function": {"name": "weather", "arguments": '{"city": "Rome"}'},
                    }
                ]
            ),
        )
    else:
        add_completion(
            httpx2_mock,
            completion(
                None,
                tool_calls=[tool_call("call_1", "Paris", SIGNATURE), tool_call("call_2", "Rome")],
            ),
        )
    if stream:
        add_stream(httpx2_mock, chunk(role="assistant", content="Sunny in both"))
    else:
        add_completion(httpx2_mock, completion("Sunny in both"))
    model = make(False, supports_tools=True)
    assert model.chain("Weather?", tools=[weather], stream=stream).text() == "Sunny in both"
    calls = assistant(request_json)["tool_calls"]
    assert [call.get("thought_signature") for call in calls] == [SIGNATURE, None]
    assert [call["id"] for call in calls] == ["call_1", "call_2"]


def test_fields_go_back_only_to_the_model_that_sent_them(
    httpx2_mock, mock_venice_api_key, request_json
):
    add_completion(httpx2_mock, completion(thought_signature=SIGNATURE))
    add_completion(httpx2_mock, completion("Two"))
    conversation = make(False).conversation()
    conversation.prompt("first", stream=False).text()
    # Continue the same history with another model.
    conversation.model = make(False, "venice/other")
    conversation.prompt("second", stream=False).text()
    assert assistant(request_json) == {"role": "assistant", "content": "One"}


def test_responses_without_fields_add_no_parts(httpx2_mock, mock_venice_api_key):
    add_completion(httpx2_mock, completion())
    (response,) = converse(make(False), "first")
    assert not [p for p in response.messages()[0].parts if isinstance(p, ReasoningPart)]


# cache_control breakpoints.


@pytest.mark.parametrize("is_async", [False, True])
def test_cache_control_marks_the_system_prompt_and_newest_message(
    is_async, httpx2_mock, mock_venice_api_key, request_json
):
    add_completion(httpx2_mock, completion())
    add_completion(httpx2_mock, completion("Two"))
    converse(make(is_async), "first", "second", system="Be brief.", cache_control="1h")
    control = {"type": "ephemeral", "ttl": "1h"}
    body = request_json()
    assert body["messages"] == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Be brief.", "cache_control": control}],
        },
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "One"},
        {"role": "user", "content": [{"type": "text", "text": "second", "cache_control": control}]},
    ]
    assert "cache_control" not in body


def test_cache_control_on_attachments_and_tool_results(
    httpx2_mock, mock_venice_api_key, request_json
):
    add_completion(httpx2_mock, completion())
    png = llm.Attachment(content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    make(False, vision=True).prompt(
        "look", attachments=[png], stream=False, cache_control="5m"
    ).text()
    last_block = request_json()["messages"][-1]["content"][-1]
    assert last_block["type"] == "image_url"
    assert last_block["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    add_completion(httpx2_mock, completion(None, tool_calls=[tool_call("call_1", "Paris")]))
    add_completion(httpx2_mock, completion("Sunny"))
    model = make(False, supports_tools=True)
    model.chain("Weather?", tools=[weather], stream=False, options={"cache_control": "5m"}).text()
    tool_message = request_json()["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_cache_control_is_checked(httpx2_mock, mock_venice_api_key):
    with pytest.raises(ValueError):
        make(False).prompt("x", cache_control="2h", stream=False).text()
    with pytest.raises(llm.ModelError, match="does not take cache_control"):
        make(False).prompt("x", cache_control="5m", responses_api="auto", stream=False).text()
    assert not httpx2_mock.get_requests()
