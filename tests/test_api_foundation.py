"""Contracts shared by sync/async clients, independent of transport and adapters."""

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import httpx2
import pytest
from pydantic import ValidationError
from typing_extensions import assert_type

from llm_venice import _files
from llm_venice._endpoints import (
    KeyInput,
    UpscaleInput,
    image_body,
    model_resource,
    parse_image,
    parse_speech,
    parse_upscale,
    resource,
    speech_body,
)
from llm_venice._http import check_status
from llm_venice.errors import (
    AuthenticationError,
    RateLimitError,
    VeniceAPIError,
    VeniceError,
    VeniceResponseError,
)
from llm_venice.params import ImageParams, SpeechParams, copy_params
from llm_venice.results import GeneratedImage, ImageResult, SpeechResult
from llm_venice.types import ApiKey, CreatedApiKey, ModelInfo

PNG = b"\x89PNG\r\n\x1a\nbytes"
WEBP = b"RIFF0000WEBPbytes"


def response(**kwargs):
    return httpx2.Response(200, request=httpx2.Request("GET", "https://example.com"), **kwargs)


def test_defaults_aliases_and_extra_body_survive_copy_without_mutation():
    extra = {"future": {"value": [1]}, "explicit_null": None}
    params = ImageParams.model_validate({"format": "webp", "extra_body": extra})
    copied = copy_params(ImageParams, params)
    body = image_body("model", "hello", copied)
    assert body == {
        "model": "model",
        "prompt": "hello",
        "format": "webp",
        "return_binary": False,
        **extra,
    }
    assert params.extra_body == extra
    assert image_body("m", "p", None) == {"model": "m", "prompt": "p", "return_binary": False}
    assert speech_body("m", "p", None, streaming=True) == {
        "model": "m",
        "input": "p",
        "streaming": True,
    }


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "prompt",
        "format",
        "image_format",
        "return_binary",
        "streaming",
        "output_dir",
        "output_filename",
        "overwrite_files",
        "extra_body",
        "width",
    ],
)
def test_reserved_extras_fail_even_when_field_is_omitted(field):
    with pytest.raises(ValidationError, match="reserved fields"):
        ImageParams(extra_body={field: None})


def test_mutated_params_revalidated_at_request_boundary():
    params = ImageParams(width=32, steps=200)
    params.width = -1
    with pytest.raises(ValidationError):
        image_body("m", "p", params)
    with pytest.raises(ValidationError):
        image_body("m", "", None)
    with pytest.raises(ValidationError):
        speech_body("m", "x" * 4097, None, streaming=False)
    with pytest.raises(ValidationError):
        ImageParams(image_format="jpeg", transparent_background=True)
    with pytest.raises(ValidationError):
        SpeechParams(speed=float("nan"))


def test_endpoint_limits_match_dated_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures/openapi-2026-10-26.json").read_text())
    image = fixture["schemas"]["GenerateImageRequest"]["properties"]
    assert image["width"]["maximum"] == 1280
    assert "minimum" not in image["steps"]
    assert ImageParams(width=1, steps=200).width == 1
    for value in (0, 1281):
        with pytest.raises(ValidationError):
            ImageParams(width=value)
    assert set(image["quality"]["enum"]) == {"low", "medium", "high"}
    speech = fixture["schemas"]["CreateSpeechRequestSchema"]["properties"]
    for speed in (speech["speed"]["minimum"], speech["speed"]["maximum"]):
        assert SpeechParams(speed=speed).speed == speed
    with pytest.raises(ValidationError):
        UpscaleInput(scale=3)


def test_key_inputs_and_utc_serialization():
    with pytest.raises(ValidationError, match="timezone-aware"):
        KeyInput(key_type="ADMIN", expires_at=datetime(2026, 1, 1))
    with pytest.raises(ValidationError):
        KeyInput(key_type="ADMIN", limit_usd=-1)
    value = KeyInput(key_type="ADMIN", expires_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert value.body()["expiresAt"] == "2026-01-01T00:00:00.000Z"


def test_resource_shapes_and_types_preserve_unknown_fields():
    model = model_resource({"id": "m", "type": "new-kind", "future": {"x": 1}}, "Models")
    assert dict(model)["future"] == {"x": 1}
    assert_type(model, ModelInfo)
    assert_type(model["id"], str)
    assert model["id"] == "m"
    assert_type(model.get("created"), int | None)
    key = resource({"id": "k", "apiKey": "secret", "future": 3}, CreatedApiKey, "Keys")
    assert_type(key["apiKey"], str)
    assert dict(key)["future"] == 3
    with pytest.raises(VeniceResponseError):
        resource({"apiKey": "secret"}, CreatedApiKey, "Keys")
    with pytest.raises(VeniceResponseError):
        resource({"id": 1}, ApiKey, "Keys")
    with pytest.raises(VeniceResponseError):
        model_resource({"id": "x", "type": None}, "Models")


def test_images_preserve_partial_moderation_and_format():
    obj = {
        "images": [base64.b64encode(WEBP).decode()],
        "content_policy_violation_count": 2,
        "timing": {"total": 5},
        "future": True,
    }
    result = parse_image(
        response(
            json=obj,
            headers={
                "x-venice-is-content-violation": "true",
                "x-venice-is-blurred": "true",
            },
        ),
        "m",
        "png",
    )
    assert result.images[0].data == WEBP
    assert result.images[0].format == "webp"
    assert result.content_policy_violation_count == 2
    assert result.content_violation and result.blurred
    assert result.raw == {k: v for k, v in obj.items() if k != "images"}
    assert (
        parse_image(
            response(
                content=b"blocked",
                headers={
                    "x-venice-is-content-violation": "true",
                },
            ),
            "m",
            None,
        ).images
        == ()
    )
    assert (
        parse_image(
            response(
                json={"images": []},
                headers={
                    "x-venice-content-violation-count": "2",
                },
            ),
            "m",
            None,
        ).content_policy_violation_count
        == 2
    )


@pytest.mark.parametrize(
    "obj",
    [
        [],
        {},
        {"images": []},
        {"images": ["%%%"]},
        {"images": [42]},
        {"images": [""]},
        {"images": [], "content_policy_violation_count": -1},
    ],
)
def test_malformed_image_responses(obj):
    with pytest.raises(VeniceResponseError):
        parse_image(response(json=obj), "m", "png")


def test_actual_media_formats_and_unknown_formats():
    assert (
        parse_speech(
            response(content=b"wav", headers={"content-type": "audio/wav"}), "m", None
        ).format
        == "wav"
    )
    assert parse_upscale(response(content=PNG)).format == "png"
    with pytest.raises(VeniceResponseError, match="unknown output format"):
        parse_speech(response(content=b"unknown"), "m", None)


def test_service_errors_and_retry_after():
    assert not issubclass(VeniceAPIError, RuntimeError)
    for code, error in [
        (401, AuthenticationError),
        (403, AuthenticationError),
        (429, RateLimitError),
        (500, VeniceAPIError),
    ]:
        r = httpx2.Response(
            code,
            json={"code": "CODE", "message": "detail"},
            headers={"Retry-After": "12"},
            request=httpx2.Request("GET", "https://example.com"),
        )
        with pytest.raises(error) as caught:
            check_status(r, "Testing")
        assert isinstance(caught.value, VeniceError)
        assert caught.value.__cause__ is not None
        assert "(CODE) - detail" in str(caught.value)
        if isinstance(caught.value, RateLimitError):
            assert caught.value.retry_after == 12


def test_exact_paths_batch_names_and_no_implicit_output(tmp_path):
    image = GeneratedImage(PNG, "png", "image/png")
    result = ImageResult("m", (image, image))
    assert result.save(tmp_path / "nested", filename="windmill.webp") == [
        tmp_path / "nested/windmill_1.webp",
        tmp_path / "nested/windmill_2.webp",
    ]
    assert (tmp_path / "nested/windmill_1.webp").read_bytes() == PNG
    with pytest.raises(ValueError, match="basename"):
        result.save(tmp_path, filename="../bad")
    assert ImageResult("m", ()).save(tmp_path / "absent") == []
    assert not (tmp_path / "absent").exists()
    target = tmp_path / "exact"
    SpeechResult("m", b"speech", "audio/wav", "wav").save(target)
    with pytest.raises(FileExistsError) as caught:
        image.save(target)
    assert target.read_bytes() == b"speech"
    assert getattr(caught.value, "partial_path").read_bytes() == PNG
    with pytest.raises(IsADirectoryError):
        image.save(tmp_path)


def test_atomic_overwrite_and_symlink_handling(tmp_path):
    image = GeneratedImage(PNG, "png", None)
    target = tmp_path / "target"
    target.write_bytes(b"existing")
    target.chmod(0o640)
    image.save(target, overwrite=True)
    assert target.stat().st_mode & 0o777 == 0o640
    link = tmp_path / "symlink"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        image.save(link)
    image.save(link, overwrite=True)
    assert not link.is_symlink()
    assert link.stat().st_mode & 0o777 == 0o600


def test_batch_failure_preserves_earlier_success(tmp_path):
    image = GeneratedImage(PNG, "png", None)
    (tmp_path / "image_2.png").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        ImageResult("m", (image, image)).save(tmp_path)
    assert (tmp_path / "image_1.png").read_bytes() == PNG
    assert (tmp_path / "image_2.png").read_bytes() == b"old"


def test_partial_retention_and_empty_cleanup(tmp_path):
    for content in (b"", b"partial"):
        error = OSError("write failed")

        def chunks():
            yield content
            raise error

        with pytest.raises(OSError) as caught:
            _files.write_chunks(chunks(), tmp_path / "file")
        assert caught.value is error
        if content:
            assert getattr(error, "partial_path").read_bytes() == content
        else:
            assert list(tmp_path.iterdir()) == []


def test_publication_cleanup_diagnostic_is_silent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_files, "try_rename_no_replace", lambda *args: False)
    original = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.suffix == ".part":
            raise OSError("cleanup failed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    output = _files.write_chunks([b"data"], tmp_path / "file")
    assert output.cleanup_error is not None
    assert output.path.read_bytes() == b"data"
    assert output.staging_path is not None
    assert os.stat(output.path).st_ino == os.stat(output.staging_path).st_ino
    assert capsys.readouterr().err == ""
