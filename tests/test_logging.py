"""Tests for LLM_VENICE_SHOW_RESPONSES / LLM_OPENAI_SHOW_RESPONSES logging."""

import asyncio
import base64
import json

import httpx2
import openai
import pytest
from pytest_httpx2 import IteratorStream

from llm_venice import AsyncVeniceChat, VeniceChat, _logging
from llm_venice.client import AsyncVeniceClient, VeniceClient
from llm_venice.errors import VeniceAPIError

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
API_BASE = "https://api.venice.ai/api/v1"
CHAT_URL = f"{API_BASE}/chat/completions"
COMPLETION = {
    "id": "x",
    "object": "chat.completion",
    "created": 0,
    "model": "llama-3.3-70b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "logged-reply"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def set_env(monkeypatch, venice, openai):
    for name, value in ((_logging.VENICE_ENV, venice), (_logging.OPENAI_ENV, openai)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    "venice,openai,expected",
    [
        (None, None, False),
        ("1", None, True),
        ("yes", None, True),
        (None, "1", True),
        ("", "1", True),  # blank counts as unset and falls back
        ("  ", "1", True),
        ("off", "1", False),  # explicit Venice value wins
        ("1", "0", True),
        (None, "0", False),
        ("FALSE", None, False),
        ("no", None, False),
        ("", "", False),
    ],
)
def test_enabled_precedence_and_off_values(monkeypatch, venice, openai, expected):
    set_env(monkeypatch, venice, openai)
    assert _logging.enabled() is expected


@pytest.mark.parametrize("is_async", [False, True])
def test_media_request_and_response_bodies_are_logged(is_async, httpx2_mock, monkeypatch, capsys):
    monkeypatch.setenv(_logging.VENICE_ENV, "1")
    httpx2_mock.add_response(json={"id": "img", "images": [base64.b64encode(PNG).decode()]})

    async def run():
        async with AsyncVeniceClient(key="secret-auth") as client:
            await client.generate_image("m", "a red fox")

    if is_async:
        asyncio.run(run())
    else:
        with VeniceClient(key="secret-auth") as client:
            client.generate_image("m", "a red fox")
    err = capsys.readouterr().err
    assert "Venice request: POST https://api.venice.ai/api/v1/image/generate" in err
    assert "authorization: [redacted]" in err
    assert '"prompt": "a red fox"' in err
    assert "Venice response: 200 OK" in err
    assert '"id": "img"' in err
    assert "secret-auth" not in err


def test_multipart_request_is_summarised_without_consuming(httpx2_mock, monkeypatch, capsys):
    monkeypatch.setenv(_logging.VENICE_ENV, "1")
    httpx2_mock.add_response(content=PNG, headers={"content-type": "image/png"})
    with VeniceClient(key="k") as client:
        result = client.upscale_image(PNG)
    assert result.data == PNG
    sent = httpx2_mock.get_requests()[0]
    assert PNG in sent.content
    err = capsys.readouterr().err
    assert "[multipart/form-data, " in err
    assert f"[image/png, {len(PNG)} bytes]" in err


def test_stream_error_body_is_logged(httpx2_mock, monkeypatch, capsys):
    monkeypatch.setenv(_logging.VENICE_ENV, "1")
    httpx2_mock.add_response(status_code=400, json={"error": "bad voice"})
    with VeniceClient(key="k") as client:
        with pytest.raises(VeniceAPIError):
            with client.stream_speech("m", "text"):
                pass
    assert '"error": "bad voice"' in capsys.readouterr().err


def run_chat(is_async, stream):
    cls = AsyncVeniceChat if is_async else VeniceChat
    chat = cls(model_id="llama-3.3-70b", api_base=API_BASE)

    async def run_async():
        return await chat.prompt("logged-prompt", stream=stream).text()  # type: ignore[union-attr]

    return (
        asyncio.run(run_async()) if is_async else chat.prompt("logged-prompt", stream=stream).text()
    )


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "venice,openai", [("1", None), (None, "1"), ("", "1")], ids=["venice", "openai", "blank"]
)
def test_chat_logs_with_either_variable(
    is_async, venice, openai, httpx2_mock, monkeypatch, capsys, mock_venice_api_key
):
    set_env(monkeypatch, venice, openai)
    httpx2_mock.add_response(method="POST", url=CHAT_URL, json=COMPLETION)
    assert run_chat(is_async, stream=False) == "logged-reply"
    err = capsys.readouterr().err
    assert f"Venice request: POST {CHAT_URL}" in err
    assert '"content": "logged-prompt"' in err
    assert "Venice response: 200 OK" in err
    assert "logged-reply" in err
    assert mock_venice_api_key not in err


@pytest.mark.parametrize("is_async", [False, True])
def test_venice_off_silences_chat_despite_openai_variable(
    is_async, httpx2_mock, monkeypatch, capsys, mock_venice_api_key
):
    set_env(monkeypatch, "off", "1")
    httpx2_mock.add_response(method="POST", url=CHAT_URL, json=COMPLETION)
    assert run_chat(is_async, stream=False) == "logged-reply"
    assert capsys.readouterr().err == ""


def sse_chunk(content, **extra):
    return {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "llama-3.3-70b",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        **extra,
    }


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("chunked", [True, False], ids=["chunked", "single"])
def test_streamed_chat_events_are_logged_and_redacted(
    is_async, chunked, httpx2_mock, monkeypatch, capsys, mock_venice_api_key
):
    monkeypatch.setenv(_logging.VENICE_ENV, "1")
    first = json.dumps(sse_chunk("stréamed-", apiKey="sse-secret"), ensure_ascii=False)
    body = f"data: {first}\n\ndata: {json.dumps(sse_chunk('reply'))}\n\ndata: [DONE]\n\n"
    raw = body.encode()
    # Split mid-line and inside the multi-byte "é" to exercise incremental decoding.
    cut = raw.index("é".encode()) + 1
    response = (
        {"stream": IteratorStream([raw[:cut], raw[cut : cut + 40], raw[cut + 40 :]])}
        if chunked
        else {"content": raw}
    )
    httpx2_mock.add_response(
        method="POST", url=CHAT_URL, headers={"content-type": "text/event-stream"}, **response
    )
    assert run_chat(is_async, stream=True) == "stréamed-reply"
    err = capsys.readouterr().err
    assert "sse-secret" not in err
    assert '"apiKey": "[redacted]"' in err
    assert '"content": "stréamed-"' in err
    assert '"content": "reply"' in err
    assert "    data: [DONE]" in err


@pytest.mark.parametrize("is_async", [False, True])
def test_streamed_json_chat_body_is_redacted_once_complete(
    is_async, httpx2_mock, monkeypatch, capsys, mock_venice_api_key
):
    monkeypatch.setenv(_logging.VENICE_ENV, "1")
    completion = {**COMPLETION, "apiKey": "json-secret"}
    completion["choices"] = [
        {
            **COMPLETION["choices"][0],
            "logprobs": {"content": [{"token": "logged", "logprob": -0.1}]},
        }
    ]
    raw = json.dumps(completion).encode()
    httpx2_mock.add_response(
        method="POST",
        url=CHAT_URL,
        headers={"content-type": "application/json"},
        stream=IteratorStream([raw[:20], raw[20:]]),
    )
    assert run_chat(is_async, stream=False) == "logged-reply"
    err = capsys.readouterr().err
    assert "json-secret" not in err
    assert '"token": "logged"' in err
    assert '\n      "apiKey": "[redacted]"' in err  # pretty-printed, not raw chunks


def test_already_read_chat_body_is_logged_directly(monkeypatch, capsys):
    monkeypatch.setenv(_logging.VENICE_ENV, "1")
    response = httpx2.Response(200, json={"apiKey": "read-secret", "ok": True})
    _logging.log_chat_response(response)
    assert type(response) is httpx2.Response
    err = capsys.readouterr().err
    assert "read-secret" not in err
    assert '"ok": true' in err


def test_redaction_is_limited_to_credentials():
    body = {
        "apiKey": "k",
        "API_KEY": "k",
        "tools": [{"parameters": {"properties": {"token": {}, "secret": {}}}}],
        "logprobs": [{"token": "t"}],
    }
    assert _logging.redact(body) == {
        "apiKey": "[redacted]",
        "API_KEY": "[redacted]",
        "tools": [{"parameters": {"properties": {"token": {}, "secret": {}}}}],
        "logprobs": [{"token": "t"}],
    }


def test_credential_headers_are_redacted(httpx2_mock, monkeypatch, capsys):
    monkeypatch.setenv(_logging.VENICE_ENV, "1")
    httpx2_mock.add_response(json={"ok": True})
    with _logging.chat_http_client() as client:
        client.get("https://api.venice.ai/x", headers={"x-api-key": "hdr-secret"})
    err = capsys.readouterr().err
    assert "hdr-secret" not in err
    assert "x-api-key: [redacted]" in err


@pytest.mark.parametrize(
    "factory,default",
    [
        (_logging.chat_http_client, openai.DefaultHttpxClient),
        (_logging.async_chat_http_client, openai.DefaultAsyncHttpxClient),
    ],
)
def test_chat_logging_keeps_environment_proxies(monkeypatch, factory, default):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("NO_PROXY", "localhost")
    # httpx keeps environment proxies as URL-pattern mounts; supplying a custom
    # transport would drop them, so the logging client must match the default.
    logging_client, default_client = factory(), default()
    patterns = [pattern.pattern for pattern in logging_client._mounts]
    assert patterns
    assert patterns == [pattern.pattern for pattern in default_client._mounts]


def test_shorten_keeps_prose_and_truncates_blobs():
    prose = "word " * 100
    blob = "A" * 1000
    data_url = "data:image/png;base64," + blob
    shortened = _logging.shorten({"text": prose, "blob": blob, "url": data_url})
    assert shortened["text"] == prose
    assert shortened["blob"] == f"{blob[:64]}...[1000 chars]"
    assert shortened["url"].endswith(f"...[{len(data_url)} chars]")
