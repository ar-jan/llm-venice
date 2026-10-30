"""Tests for LLM_VENICE_SHOW_RESPONSES / LLM_OPENAI_SHOW_RESPONSES logging."""

import asyncio
import base64
import json

import pytest

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


@pytest.mark.parametrize("is_async", [False, True])
def test_streamed_chat_chunks_are_echoed(
    is_async, httpx2_mock, monkeypatch, capsys, mock_venice_api_key
):
    monkeypatch.setenv(_logging.VENICE_ENV, "1")
    chunk = {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "llama-3.3-70b",
        "choices": [{"index": 0, "delta": {"content": "streamed-reply"}, "finish_reason": None}],
    }
    body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
    httpx2_mock.add_response(
        method="POST",
        url=CHAT_URL,
        content=body,
        headers={"content-type": "text/event-stream"},
    )
    assert run_chat(is_async, stream=True) == "streamed-reply"
    err = capsys.readouterr().err
    assert '"content": "streamed-reply"' in err
    assert "data: [DONE]" in err


def test_shorten_keeps_prose_and_truncates_blobs():
    prose = "word " * 100
    blob = "A" * 1000
    data_url = "data:image/png;base64," + blob
    shortened = _logging.shorten({"text": prose, "blob": blob, "url": data_url})
    assert shortened["text"] == prose
    assert shortened["blob"] == f"{blob[:64]}...[1000 chars]"
    assert shortened["url"].endswith(f"...[{len(data_url)} chars]")
