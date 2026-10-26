"""Output collisions must preserve audio and report the path actually written."""

import asyncio
import pathlib
from unittest.mock import MagicMock

import httpx2
import pytest

from llm_venice.models.audio import AsyncVeniceSpeech, VeniceSpeech


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_model_output_collision(
    httpx2_mock, mock_venice_api_key, tmp_path, asynchronous, stream, overwrite
):
    target = tmp_path / "out.mp3"

    def racing_response(request):
        # The name was available during request preparation, but another writer
        # finishes before this response arrives.
        target.write_bytes(b"other download")
        return httpx2.Response(200, content=b"new audio", headers={"Content-Type": "audio/mpeg"})

    httpx2_mock.add_callback(
        racing_response, method="POST", url="https://api.venice.ai/api/v1/audio/speech"
    )
    model = AsyncVeniceSpeech("tts-kokoro") if asynchronous else VeniceSpeech("tts-kokoro")
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options = model.Options(
        output_dir=tmp_path, output_filename=target.name, overwrite_files=overwrite
    )
    response = MagicMock()

    if isinstance(model, AsyncVeniceSpeech):

        async def collect():
            return [message async for message in model.execute(prompt, stream, response)]

        messages = asyncio.run(collect())
    else:
        messages = list(model.execute(prompt, stream, response))

    saved = pathlib.Path(response.response_json["output_path"])
    assert messages[-1] == f"Audio saved to {saved}"
    assert saved.read_bytes() == b"new audio"
    assert response.response_json["bytes_written"] == len(b"new audio")
    if overwrite:
        assert saved == target
    else:
        assert saved != target
        assert target.read_bytes() == b"other download"
    assert not list(tmp_path.glob("*.part"))
