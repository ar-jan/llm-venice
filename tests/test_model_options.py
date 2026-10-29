"""Catalog-aware field schemas and positional validation share one policy."""

import asyncio
import json
from unittest.mock import MagicMock
from typing import Any

import llm
from llm.cli import cli
import pytest
from pydantic import ValidationError

from llm_venice import (
    VeniceImage,
    AsyncVeniceImage,
    VeniceSpeech,
    AsyncVeniceSpeech,
    VeniceChat,
    AsyncVeniceChat,
)
from llm_venice._options import options_class


@pytest.mark.parametrize("cls", [VeniceImage, AsyncVeniceImage])
def test_image_constraints_in_schema_and_validation(cls):
    constraints = {
        "steps": {"min": 10, "max": 20},
        "widthHeightDivisor": 32,
        "aspectRatios": ["1:1", "16:9"],
        "resolutions": ["1K"],
        "qualities": ["high"],
        "supportsTransparentBackground": False,
    }
    model = cls("m", image_constraints=constraints)
    opts = model.Options
    for values in [
        dict(steps=9),
        dict(steps=21),
        dict(width=63),
        dict(height=33),
        dict(aspect_ratio="9:16"),
        dict(resolution="2K"),
        dict(quality="low"),
        dict(transparent_background=True),
    ]:
        with pytest.raises(ValidationError):
            opts.model_validate(values)
    valid = opts(steps=12, width=64, quality="high", extra_body={"future": None})
    assert valid.image_format == "png" and valid.safe_mode is False and valid.hide_watermark is True
    schema = opts.model_json_schema()["properties"]
    assert schema["steps"]["anyOf"][0]["minimum"] == 10
    assert "high" in schema["quality"]["enum"]
    assert "format" in schema and "output_dir" in schema
    assert cls.Options(steps=2).steps == 2


@pytest.mark.parametrize("cls", [VeniceImage, AsyncVeniceImage])
def test_constrained_boolean_option_still_coerces_cli_strings(cls):
    """Restricting a bool option to False must not break string coercion."""
    opts = cls("m", image_constraints={"supportsTransparentBackground": False}).Options
    assert opts.model_validate({"transparent_background": "false"}).transparent_background is False
    assert opts.model_validate({}).transparent_background is None
    for value in ("true", True):
        with pytest.raises(ValidationError, match="one of"):
            opts.model_validate({"transparent_background": value})


@pytest.mark.parametrize("cls", [VeniceImage, AsyncVeniceImage])
def test_supported_boolean_option_accepts_both_values(cls):
    """True means transparency is allowed, so neither value is constrained."""
    opts = cls("m", image_constraints={"supportsTransparentBackground": True}).Options
    for value, expected in (("true", True), (True, True), ("false", False), (False, False)):
        assert (
            opts.model_validate({"transparent_background": value}).transparent_background
            is expected
        )
    assert opts.model_validate({}).transparent_background is None


@pytest.mark.parametrize("is_async", [False, True])
def test_prompt_limit_is_checked_outside_options(is_async, httpx2_mock, mock_venice_api_key):
    model: Any = (AsyncVeniceImage if is_async else VeniceImage)(
        "m", image_constraints={"promptCharacterLimit": 3}
    )
    prompt = MagicMock(prompt="too long", options=model.Options())

    async def run():
        return [part async for part in model.execute(prompt, False, MagicMock())]

    with pytest.raises(llm.ModelError, match="at most 3"):
        if is_async:
            asyncio.run(run())
        else:
            list(model.execute(prompt, False, MagicMock()))
    assert not httpx2_mock.get_requests()


def test_cache_tracks_constraints_and_is_bounded():
    first = VeniceImage("m", image_constraints={"steps": {"min": 2}})
    second = AsyncVeniceImage("m", image_constraints={"steps": {"min": 2}})
    changed = VeniceImage("m", image_constraints={"steps": {"min": 3}})
    assert first.Options is second.Options
    assert first.Options is not changed.Options
    with pytest.raises(ValidationError):
        changed.Options(steps=2)
    assert options_class.cache_info().maxsize == 128


def test_options_descriptor_class_and_instance_access():
    base = VeniceImage.Options
    assert AsyncVeniceImage.Options is base
    model = VeniceImage("m", image_constraints={"steps": {"min": 2}})
    specialized = model.Options
    assert specialized is not base and issubclass(specialized, base)
    assert specialized.__name__ == "VeniceImageOptionsForModel"
    assert model.Options is specialized
    model.catalog_policy = {"kind": "image", "fields": {"steps": {"ge": 5}}}
    with pytest.raises(ValidationError):
        model.Options(steps=4)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "broken",
        [],
        {
            "steps": {"min": "bad", "max": []},
            "widthHeightDivisor": 0,
            "aspectRatios": [1],
            "qualities": {},
        },
    ],
)
def test_malformed_metadata_retains_base_validation(bad):
    model = VeniceImage("m", model_spec={"constraints": bad})
    assert model.Options(width=31, steps=2, aspect_ratio="new").width == 31


@pytest.mark.parametrize("cls", [VeniceSpeech, AsyncVeniceSpeech])
def test_speech_formats_speed_and_open_voices(cls):
    model = cls(
        "tts",
        model_spec={
            "supported_formats": ["wav", "pcm"],
            "default_format": "wav",
            "voices": ["stock"],
            "constraints": {"speed": {"min": 0.5, "max": 2}},
        },
    )
    assert model.Options().response_format == "wav"
    assert model.Options().voice is None
    assert model.Options(voice="vv_clone").voice == "vv_clone"
    assert model.Options(voice="custom-provider-id").voice == "custom-provider-id"
    with pytest.raises(ValidationError):
        model.Options(response_format="mp3")
    with pytest.raises(ValidationError):
        model.Options(speed=3)
    assert cls.Options().response_format == "mp3"


@pytest.mark.parametrize("cls", [VeniceChat, AsyncVeniceChat])
def test_reasoning_capabilities(cls):
    unsupported = cls(model_id="m", model_spec={"capabilities": {"supportsReasoningEffort": False}})
    with pytest.raises(ValidationError):
        unsupported.Options(reasoning_effort="low")
    supported = cls(
        model_id="m",
        model_spec={
            "capabilities": {
                "supportsReasoningEffort": True,
                "reasoningEffortOptions": ["low", "high"],
            }
        },
    )
    assert supported.Options(reasoning_effort="low").reasoning_effort == "low"
    with pytest.raises(ValidationError):
        supported.Options(reasoning_effort="medium")
    assert cls(model_id="m").Options(reasoning_effort="future").reasoning_effort == "future"


def test_tts_uses_model_specific_defaults_and_validation(
    tmp_path, monkeypatch, httpx2_mock, cli_runner, mock_venice_api_key
):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "venice_models.json").write_text(
        json.dumps(
            [
                {
                    "id": "voice",
                    "type": "tts",
                    "model_spec": {
                        "supported_formats": ["wav"],
                        "default_format": "wav",
                        "constraints": {"inputCharacterLimit": 4, "speed": {"max": 2}},
                    },
                }
            ]
        )
    )
    for args in [["long text"], ["hey", "--speed", "3"], ["hey", "--format", "mp3"]]:
        output = cli_runner.invoke(cli, ["venice", "tts", *args, "-o", "-"])
        assert output.exit_code != 0
    assert not httpx2_mock.get_requests()
    httpx2_mock.add_response(content=b"audio", headers={"content-type": "audio/wav"})
    output = cli_runner.invoke(cli, ["venice", "tts", "hey", "-o", "-"])
    assert output.exit_code == 0, output.output
    assert output.stdout_bytes == b"audio"
    payload = json.loads(httpx2_mock.get_request().content)
    assert payload["response_format"] == "wav"
    assert "voice" not in payload


def test_models_options_shows_model_limits(tmp_path, monkeypatch, cli_runner):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    (tmp_path / "venice_models.json").write_text(
        json.dumps(
            [
                {
                    "id": "image",
                    "type": "image",
                    "model_spec": {"constraints": {"qualities": ["high"], "aspectRatios": ["1:1"]}},
                }
            ]
        )
    )
    output = cli_runner.invoke(cli, ["models", "--options", "-m", "venice/image"])
    assert output.exit_code == 0, output.output
    assert "quality" in output.output and "high" in output.output
    assert "return_binary" not in output.output
