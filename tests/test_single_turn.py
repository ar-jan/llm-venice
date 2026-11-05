"""Image and speech models are single-turn; llm rejects conversation continuation."""

import asyncio
import base64
import json

import llm
import pytest
from llm.cli import cli

from llm_venice import (
    AsyncVeniceChat,
    AsyncVeniceImage,
    AsyncVeniceSpeech,
    VeniceChat,
    VeniceImage,
    VeniceSpeech,
)
from llm_venice.constants import ENDPOINT_AUDIO_SPEECH, ENDPOINT_IMAGE_GENERATE

PNG = b"\x89PNG\r\n\x1a\n"
AUDIO = b"ID3audio"
MEDIA_MODELS = ["venice/img-model", "venice/tts-model"]


@pytest.fixture
def media(hermetic_venice_catalog, httpx2_mock, mock_venice_api_key, tmp_path):
    """A catalog with image, tts and text models, and reusable media responses."""
    hermetic_venice_catalog(
        [
            {"id": "img-model", "type": "image", "model_spec": {"capabilities": {}}},
            {"id": "tts-model", "type": "tts"},
            {"id": "text-model", "type": "text", "model_spec": {"capabilities": {}}},
        ]
    )
    httpx2_mock.add_response(
        method="POST",
        url=ENDPOINT_IMAGE_GENERATE,
        json={"images": [base64.b64encode(PNG).decode()], "request": {}, "timing": {}},
        is_optional=True,
        is_reusable=True,
    )
    httpx2_mock.add_response(
        method="POST",
        url=ENDPOINT_AUDIO_SPEECH,
        content=AUDIO,
        headers={"content-type": "audio/mpeg"},
        is_optional=True,
        is_reusable=True,
    )
    return str(tmp_path)


def sent_prompts(httpx2_mock) -> list[str]:
    """Return the prompt or speech input of each request sent to Venice."""
    bodies = [json.loads(request.read()) for request in httpx2_mock.get_requests()]
    return [body.get("prompt", body.get("input")) for body in bodies]


def test_media_models_declare_single_turn():
    for model_class in (VeniceImage, AsyncVeniceImage, VeniceSpeech, AsyncVeniceSpeech):
        assert model_class.supports_conversation is False
    for model_class in (VeniceChat, AsyncVeniceChat):
        assert model_class.supports_conversation is True


@pytest.mark.parametrize("model_id", MEDIA_MODELS)
def test_continue_flag_is_rejected_without_a_request(model_id, media, cli_runner, httpx2_mock):
    first = cli_runner.invoke(cli, ["-m", model_id, "-o", "output_dir", media, "first"])
    assert first.exit_code == 0, first.output

    result = cli_runner.invoke(cli, ["-c", "-o", "output_dir", media, "second"])

    assert result.exit_code == 1
    assert f"{model_id} does not support conversations" in result.output
    assert sent_prompts(httpx2_mock) == ["first"]


@pytest.mark.parametrize("model_id", MEDIA_MODELS)
def test_conversation_id_is_rejected_without_a_request(model_id, media, cli_runner, httpx2_mock):
    first = cli_runner.invoke(cli, ["-m", model_id, "-o", "output_dir", media, "first"])
    assert first.exit_code == 0, first.output
    logs = json.loads(cli_runner.invoke(cli, ["logs", "--json", "-n", "1"]).output)

    result = cli_runner.invoke(
        cli, ["--cid", logs[0]["conversation_id"], "-o", "output_dir", media, "second"]
    )

    assert result.exit_code == 1
    assert f"{model_id} does not support conversations" in result.output
    assert sent_prompts(httpx2_mock) == ["first"]


@pytest.mark.parametrize("model_id", MEDIA_MODELS)
def test_chat_session_is_refused_before_it_starts(model_id, media, cli_runner, httpx2_mock):
    result = cli_runner.invoke(cli, ["chat", "-m", model_id], input="hi\nquit\n")

    assert result.exit_code == 1
    assert f"{model_id} does not support conversations" in result.output
    assert "Chatting with" not in result.output
    assert httpx2_mock.get_requests() == []


@pytest.mark.parametrize("model_id", MEDIA_MODELS)
def test_python_conversation_reply_raises(model_id, media, httpx2_mock):
    conversation = llm.get_model(model_id).conversation()
    conversation.prompt("first", output_dir=media).text()

    with pytest.raises(llm.ConversationNotSupported):
        conversation.prompt("second", output_dir=media).text()

    assert sent_prompts(httpx2_mock) == ["first"]


@pytest.mark.parametrize("model_id", MEDIA_MODELS)
def test_async_conversation_reply_raises(model_id, media, httpx2_mock):
    async def run():
        conversation = llm.get_async_model(model_id).conversation()
        await conversation.prompt("first", output_dir=media).text()
        await conversation.prompt("second", output_dir=media).text()

    with pytest.raises(llm.ConversationNotSupported):
        asyncio.run(run())

    assert sent_prompts(httpx2_mock) == ["first"]


@pytest.mark.parametrize("model_id", MEDIA_MODELS)
def test_single_turn_with_system_prompt_still_works(model_id, media, cli_runner, httpx2_mock):
    result = cli_runner.invoke(
        cli, ["-m", model_id, "-s", "be brief", "-o", "output_dir", media, "only"]
    )

    assert result.exit_code == 0, result.output
    assert sent_prompts(httpx2_mock) == ["only"]


def test_models_json_reports_conversation_support(media, cli_runner):
    result = cli_runner.invoke(cli, ["models", "list", "--json"])

    assert result.exit_code == 0, result.output
    support = {
        entry["model_id"]: entry["supports_conversation"]
        for entry in json.loads(result.output)
        if entry["model_id"].startswith("venice/")
    }
    assert support == {
        "venice/img-model": False,
        "venice/tts-model": False,
        "venice/text-model": True,
    }
