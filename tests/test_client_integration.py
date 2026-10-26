"""Manual API smoke tests; image generation needs an additional explicit opt-in."""

import asyncio
import os

import pytest

from llm_venice import VeniceClient, AsyncVeniceClient, refresh_catalog

pytestmark = pytest.mark.integration


def test_catalog_characters_and_limits(venice_api_key, tmp_path):
    with VeniceClient(key=venice_api_key) as client:
        assert refresh_catalog(client, tmp_path / "catalog.json")
        assert isinstance(client.list_characters(), list)
        assert "accessPermitted" in client.get_rate_limits()


def test_short_speech_sync_and_async(venice_api_key):
    with VeniceClient(key=venice_api_key) as client:
        models = client.list_models(type="tts")
        model = next(m for m in models if not m.get("model_spec", {}).get("offline"))["id"]
        result = client.generate_speech(model, "Hello.")
        assert result.data and result.format

    async def run():
        async with AsyncVeniceClient(key=venice_api_key) as client:
            async with client.stream_speech(model, "Hello.") as stream:
                assert b"".join([chunk async for chunk in stream.aiter_bytes()])

    asyncio.run(run())


@pytest.mark.skipif(
    os.environ.get("LLM_VENICE_TEST_IMAGES") != "1",
    reason="Set LLM_VENICE_TEST_IMAGES=1 to opt into paid image generation",
)
def test_paid_image_generation(venice_api_key):
    with VeniceClient(key=venice_api_key) as client:
        models = client.list_models(type="image")
        model = next(m for m in models if not m.get("model_spec", {}).get("offline"))["id"]
        result = client.generate_image(model, "A red apple on a plain white table")
        assert result.images or result.content_violation or result.content_policy_violation_count
