"""Chat attachments beyond llm's own: more audio formats, and video."""

import asyncio
import base64
import json

import llm
import pytest

from llm_venice import AsyncVeniceChat, VeniceChat
from llm_venice.models import register_venice_models
from llm_venice.models.chat import AUDIO_FORMATS, VIDEO_TYPES

CHAT_URL = "https://api.venice.ai/api/v1/chat/completions"
FLAC = b"fLaC\x00\x00\x00\x22" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00" * 40
MPEG_PS = b"\x00\x00\x01\xba\x44\x00\x04\x00\x04\x01" + b"\x00" * 60
OGG = b"OggS\x00\x02" + b"\x00" * 22 + b"\x01vorbis" + b"\x00" * 40


def completion(text="ok"):
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


def make(is_async, **kwargs):
    return (AsyncVeniceChat if is_async else VeniceChat)("venice/test-model", **kwargs)


def run(model, *attachments, **kwargs):
    if isinstance(model, AsyncVeniceChat):

        async def go():
            return await model.prompt(
                "describe", attachments=list(attachments), stream=False, **kwargs
            ).text()

        return asyncio.run(go())
    return model.prompt("describe", attachments=list(attachments), stream=False, **kwargs).text()


def sent_content(request_json):
    return request_json()["messages"][-1]["content"]


def b64(data):
    return base64.b64encode(data).decode()


@pytest.mark.parametrize("is_async", [False, True])
def test_media_types_follow_the_capabilities(is_async):
    assert make(is_async).attachment_types == set()
    audio = make(is_async, audio=True)
    assert audio.attachment_types == set(AUDIO_FORMATS)
    video = make(is_async, video=True, vision=True)
    assert set(VIDEO_TYPES) <= video.attachment_types
    assert not set(AUDIO_FORMATS) & video.attachment_types
    assert "image/png" in video.attachment_types


def test_registration_reads_video_input(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    capabilities = {"supportsVideoInput": True, "supportsAudioInput": True}
    (tmp_path / "venice_models.json").write_text(
        json.dumps([{"id": "av", "type": "text", "model_spec": {"capabilities": capabilities}}])
    )
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)
    models = []
    register_venice_models(
        lambda model, async_model=None, aliases=None: models.extend((model, async_model))
    )
    for model in models:
        assert set(VIDEO_TYPES) | set(AUDIO_FORMATS) <= model.attachment_types


def test_formats_match_swagger():
    # Swagger 20261122.230254: the input_audio.format enum and the video_url formats.
    documented = {"wav", "mp3", "aiff", "aac", "ogg", "flac", "m4a", "pcm16", "pcm24"}
    assert set(AUDIO_FORMATS.values()) == documented - {"pcm16", "pcm24"}
    assert set(VIDEO_TYPES.values()) == {"video/mp4", "video/mpeg", "video/quicktime", "video/webm"}


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "attachment,format_",
    [
        (llm.Attachment(content=FLAC), "flac"),
        (llm.Attachment(content=OGG), "ogg"),
        (llm.Attachment(content=b"audio", type="audio/mp4"), "m4a"),
        (llm.Attachment(content=b"audio", type="audio/aac"), "aac"),
        (llm.Attachment(content=b"audio", type="audio/x-aiff"), "aiff"),
        (llm.Attachment(content=b"audio", type="audio/wav"), "wav"),
        (llm.Attachment(content=b"audio", type="audio/mpeg"), "mp3"),
    ],
)
def test_audio_is_sent_in_its_format(
    is_async, attachment, format_, httpx2_mock, mock_venice_api_key, request_json
):
    httpx2_mock.add_response(method="POST", url=CHAT_URL, json=completion())
    run(make(is_async, audio=True), attachment)
    text, audio = sent_content(request_json)
    assert text == {"type": "text", "text": "describe"}
    assert audio == {
        "type": "input_audio",
        "input_audio": {"data": attachment.base64_content(), "format": format_},
    }


@pytest.mark.parametrize("is_async", [False, True])
def test_remote_ogg_loads_audio_bytes(is_async, httpx2_mock, mock_venice_api_key, request_json):
    url = "https://example.com/tone.ogg"
    httpx2_mock.add_response(method="GET", url=url, content=OGG)
    httpx2_mock.add_response(method="POST", url=CHAT_URL, json=completion())
    run(make(is_async, audio=True), llm.Attachment(url=url, type="application/ogg"))
    assert sent_content(request_json)[-1]["input_audio"] == {"format": "ogg", "data": b64(OGG)}


@pytest.mark.parametrize("is_async", [False, True])
def test_video_is_rejected_on_responses_route(is_async, httpx2_mock, mock_venice_api_key):
    with pytest.raises(llm.ModelError, match="video/mp4.*use Chat Completions"):
        run(make(is_async, video=True), llm.Attachment(content=MP4), responses_api="compat")
    assert not httpx2_mock.get_requests()


@pytest.mark.parametrize("is_async", [False, True])
def test_video_is_sent_as_video_url(is_async, httpx2_mock, mock_venice_api_key, request_json):
    httpx2_mock.add_response(method="POST", url=CHAT_URL, json=completion())
    url = "https://example.com/clip.webm"
    run(
        make(is_async, video=True),
        llm.Attachment(content=MP4),
        llm.Attachment(content=MPEG_PS),
        llm.Attachment(url=url, type="video/webm"),
    )
    _, mp4, mpeg, remote = sent_content(request_json)
    assert mp4 == {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64(MP4)}"}}
    # llm detects an MPEG program stream as video/dvd; Venice takes video/mpeg.
    assert mpeg["video_url"]["url"] == f"data:video/mpeg;base64,{b64(MPEG_PS)}"
    assert remote == {"type": "video_url", "video_url": {"url": url}}


@pytest.mark.parametrize("is_async", [False, True])
def test_mixed_attachments_keep_their_order(
    is_async, httpx2_mock, mock_venice_api_key, request_json
):
    httpx2_mock.add_response(method="POST", url=CHAT_URL, json=completion())
    png = llm.Attachment(content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    run(
        make(is_async, vision=True, audio=True, video=True),
        llm.Attachment(content=MP4),
        png,
        llm.Attachment(content=FLAC),
    )
    _, video, image, audio = sent_content(request_json)
    assert video["type"] == "video_url"
    assert image == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{png.base64_content()}"},
    }
    assert audio["input_audio"]["format"] == "flac"


def test_earlier_turns_are_sent_the_same_way(httpx2_mock, mock_venice_api_key, request_json):
    httpx2_mock.add_response(method="POST", url=CHAT_URL, json=completion("first"))
    httpx2_mock.add_response(method="POST", url=CHAT_URL, json=completion("second"))
    conversation = make(False, audio=True, video=True).conversation()
    conversation.prompt("one", attachments=[llm.Attachment(content=FLAC)], stream=False).text()
    conversation.prompt("two", stream=False).text()
    first_turn = request_json()["messages"][0]["content"]
    assert first_turn[1]["input_audio"]["format"] == "flac"


def test_unsupported_media_is_refused_by_llm(mock_venice_api_key):
    with pytest.raises(ValueError, match="does not support attachments of type 'audio/flac'"):
        run(make(False, video=True), llm.Attachment(content=FLAC))
