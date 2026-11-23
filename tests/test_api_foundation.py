"""Contracts shared by sync/async clients, independent of transport and adapters."""

import base64
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle

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
    ContentViolationError,
    MissingKeyError,
    RateLimitError,
    VeniceAPIError,
    VeniceConnectionError,
    VeniceError,
    VeniceResponseError,
)
from llm_venice.params import ImageParams, SpeechParams, copy_params
from llm_venice.results import GeneratedImage, ImageResult, ModelDeprecation, SpeechResult
from llm_venice.types import ApiKey, Character, CreatedApiKey, ModelInfo, RateLimits

PNG = b"\x89PNG\r\n\x1a\nbytes"
WEBP = b"RIFF0000WEBPbytes"
# Targeted extract of Swagger 20261122.230254: request schemas, resources, errors.
FIXTURE = Path(__file__).parent / "fixtures/openapi-2026-11-22.json"


def dated_fixture():
    return json.loads(FIXTURE.read_text())


def response(**kwargs):
    return httpx2.Response(200, request=httpx2.Request("GET", "https://example.com"), **kwargs)


def error_response(status_code, body):
    request = httpx2.Request("POST", "https://example.com")
    if isinstance(body, str):
        return httpx2.Response(status_code, text=body, request=request)
    return httpx2.Response(status_code, json=body, request=request)


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
        "enhance_prompt",
        "disable_prompt_optimization_thinking",
        "anon_user_id",
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
    fixture = dated_fixture()
    image = fixture["schemas"]["GenerateImageRequest"]["properties"]
    for side in ("width", "height"):
        bounds = image[side]
        assert (bounds["minimum"], bounds["exclusiveMinimum"], bounds["maximum"]) == (0, True, 4096)
        for value in (1, bounds["maximum"]):
            assert getattr(ImageParams.model_validate({side: value}), side) == value
        for value in (bounds["minimum"], bounds["maximum"] + 1):
            with pytest.raises(ValidationError):
                ImageParams.model_validate({side: value})
    assert "minimum" not in image["steps"]
    assert ImageParams(width=1, steps=200).width == 1
    assert set(image["quality"]["enum"]) == {"low", "medium", "high"}
    speech = fixture["schemas"]["CreateSpeechRequestSchema"]["properties"]
    for speed in (speech["speed"]["minimum"], speech["speed"]["maximum"]):
        assert SpeechParams(speed=speed).speed == speed
    upscale = fixture["schemas"]["UpscaleImageRequest"]["properties"]
    for scale in (upscale["scale"]["minimum"], 3, 3.5, upscale["scale"]["maximum"]):
        assert UpscaleInput(scale=scale).scale == scale
    for scale in (1.99, 4.01):
        with pytest.raises(ValidationError):
            UpscaleInput(scale=scale)
    assert upscale["creativity"]["default"] == 0.01
    for creativity in (0, 0.01, 0.02):
        assert UpscaleInput(creativity=creativity).creativity == creativity
    for creativity in (-0.01, 0.03):
        with pytest.raises(ValidationError):
            UpscaleInput(creativity=creativity)


def test_image_prompt_controls_and_attribution_match_dated_fixture():
    image = dated_fixture()["schemas"]["GenerateImageRequest"]["properties"]
    for name in ("enhance_prompt", "disable_prompt_optimization_thinking"):
        assert image[name]["type"] == "boolean"
    declared = ImageParams.model_json_schema()["properties"]["anon_user_id"]["anyOf"][0]
    for key in ("minLength", "maxLength", "pattern"):
        assert declared[key] == image["anon_user_id"][key]
    params = ImageParams(
        enhance_prompt=True,
        disable_prompt_optimization_thinking=False,
        anon_user_id=image["anon_user_id"]["example"],
    )
    body = image_body("m", "p", params)
    assert body["enhance_prompt"] is True
    assert body["disable_prompt_optimization_thinking"] is False
    assert body["anon_user_id"] == "end-user-123"
    assert ImageParams(anon_user_id=" ~|" + "x" * 125).anon_user_id is not None
    for value in ("", "x" * 129, "caf\u00e9", "tab\tid", "a||b"):
        with pytest.raises(ValidationError):
            ImageParams(anon_user_id=value)


@pytest.mark.parametrize(
    "resource_name, kind",
    [("GET /api_keys", ApiKey), ("POST /api_keys", CreatedApiKey)],
)
def test_api_key_types_match_dated_fixture(resource_name, kind):
    """Each key type declares exactly the wire fields of the response it describes."""
    fixture = dated_fixture()
    schema = fixture["resources"][resource_name]["response"]["content"]["application/json"]
    data = schema["schema"]["properties"]["data"]
    item = data.get("items", data)
    assert set(kind.__annotations__) == set(item["properties"])


def test_upscale_moderation_refusals_match_dated_fixture():
    """Every upscale 422 is a refusal, though most carry only the error text."""
    fixture = dated_fixture()
    refusal = fixture["resources"]["POST /image/upscale"]["errors"]["422"]
    ref = refusal["content"]["application/json"]["schema"]["$ref"]
    assert ref == "#/components/schemas/ContentViolationError"
    schema = fixture["schemas"]["ContentViolationError"]
    assert schema["required"] == ["error"]
    # A new field here needs a decision on how to expose it.
    assert set(schema["properties"]) == {"error", "suggested_prompt"}
    error = "Image refused by content moderation"
    for body in ({"error": error}, {"error": error, "suggested_prompt": "A lighthouse"}, error):
        with pytest.raises(ContentViolationError) as caught:
            check_status(error_response(422, body), "Upscaling image", policy_422=True)
        assert isinstance(caught.value, VeniceAPIError)
        assert caught.value.status_code == 422
        assert caught.value.detail == error
        expected = body.get("suggested_prompt") if isinstance(body, dict) else None
        assert caught.value.suggested_prompt == expected
        assert caught.value.recommended_model is None and caught.value.retryable is None


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            {
                "error": "Prompt rejected by the provider's content policy",
                "suggested_prompt": "A watercolor painting of a lighthouse",
                "recommended_model": "chroma",
                "retryable": True,
            },
            ("A watercolor painting of a lighthouse", "chroma", True),
        ),
        ({"error": "Prompt rejected", "retryable": False}, (None, None, False)),
        ({"error": "Prompt rejected", "recommended_model": "chroma"}, (None, "chroma", None)),
    ],
)
def test_generation_policy_refusals_match_dated_fixture(body, expected):
    fixture = dated_fixture()
    refusal = fixture["resources"]["POST /image/generate"]["errors"]["422"]
    ref = refusal["content"]["application/json"]["schema"]["$ref"]
    assert ref == "#/components/schemas/ImageContentViolationError"
    schema = fixture["schemas"]["ImageContentViolationError"]
    assert schema["required"] == ["error"]
    assert set(schema["properties"]) == {
        "error",
        "suggested_prompt",
        "recommended_model",
        "retryable",
    }
    assert schema["properties"]["retryable"]["type"] == "boolean"
    with pytest.raises(ContentViolationError) as caught:
        check_status(error_response(422, body), "Generating image")
    error = caught.value
    assert (error.status_code, error.detail) == (422, body["error"])
    assert (error.suggested_prompt, error.recommended_model, error.retryable) == expected


def test_content_violation_message_adds_hint_lines_after_the_api_error():
    summary = "Generating image failed: Venice API returned 422 Unprocessable Content - Refused"
    error = ContentViolationError(
        "Generating image",
        422,
        "Unprocessable Content",
        detail="Refused",
        suggested_prompt="A watercolor painting of a lighthouse",
        recommended_model="chroma",
        retryable=True,
    )
    assert str(error) == (
        f"{summary}\n"
        "Suggested prompt: A watercolor painting of a lighthouse\n"
        "Recommended model: chroma\n"
        "Retrying may succeed."
    )
    assert error.describe(model_prefix="venice/").splitlines()[2] == (
        "Recommended model: venice/chroma"
    )
    refused = ContentViolationError(
        "Generating image", 422, "Unprocessable Content", detail="Refused", retryable=False
    )
    assert str(refused) == f"{summary}\nRetrying the same request will not succeed."
    # Without hints the message is exactly VeniceAPIError's.
    bare = ContentViolationError("Generating image", 422, "Unprocessable Content", detail="Refused")
    assert str(bare) == bare.describe(model_prefix="venice/") == summary
    assert str(bare) == str(
        VeniceAPIError("Generating image", 422, "Unprocessable Content", detail="Refused")
    )


def test_other_422s_and_statuses_remain_plain_api_errors():
    # An oversized style reference is a 422 without retryable or suggestions.
    body = {"error": "Style reference image exceeds the pixel limit"}
    with pytest.raises(VeniceAPIError) as caught:
        check_status(error_response(422, body), "Generating image")
    assert type(caught.value) is VeniceAPIError
    assert caught.value.detail == body["error"]
    # Policy fields count only on a 422.
    with pytest.raises(VeniceAPIError) as caught:
        check_status(error_response(400, {"error": "Bad", "retryable": True}), "Generating image")
    assert type(caught.value) is VeniceAPIError


def test_drifted_policy_fields_do_not_break_error_mapping():
    body = {"error": "Refused", "suggested_prompt": 3, "recommended_model": "", "retryable": "yes"}
    with pytest.raises(ContentViolationError) as caught:
        check_status(error_response(422, body), "Generating image")
    error = caught.value
    assert (error.suggested_prompt, error.recommended_model, error.retryable) == (None, None, None)
    assert error.detail == "Refused"


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


def test_resource_checks_required_keys_but_tolerates_optional_drift():
    # Optional fields that drift from the documented types must not break parsing.
    drifted = {"id": "c", "slug": "s", "modelId": None, "adult": "maybe", "stats": None}
    assert resource(drifted, Character, "Characters") == drifted
    limits = {"accessPermitted": True, "balances": {"USD": None}, "apiTier": None}
    assert resource(limits, RateLimits, "Limits") == limits
    with pytest.raises(VeniceResponseError, match="invalid Character: slug"):
        resource({"id": "c"}, Character, "Characters")
    with pytest.raises(VeniceResponseError, match="invalid RateLimits: accessPermitted"):
        resource({"accessPermitted": "yes"}, RateLimits, "Limits")
    with pytest.raises(VeniceResponseError, match="expected an object"):
        resource(None, RateLimits, "Limits")


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


def test_image_headers_carry_enhanced_prompt_and_deprecation():
    images = {"images": [base64.b64encode(PNG).decode()]}
    plain = parse_image(response(json=images), "m", None)
    assert plain.enhanced_prompt is None and plain.deprecation is None
    result = parse_image(
        response(
            json=images,
            headers={
                "x-venice-enhanced-prompt": "A%20lighthouse%2C%201%2B1%20%E2%86%92%20dusk",
                "x-venice-deprecated": "true",
                "x-venice-model-deprecation-date": "2026-12-01T00:00:00.000Z",
                "x-venice-deprecated-replacement": "new-model",
            },
        ),
        "m",
        None,
    )
    assert result.enhanced_prompt == "A lighthouse, 1+1 \u2192 dusk"
    assert result.deprecation == ModelDeprecation(
        deprecated=True, date="2026-12-01T00:00:00.000Z", replacement="new-model"
    )
    scheduled = parse_image(
        response(json=images, headers={"x-venice-model-deprecation-warning": "Moving soon"}),
        "m",
        None,
    )
    assert scheduled.deprecation == ModelDeprecation(warning="Moving soon")


def test_unexpected_advisory_headers_do_not_fail_generation():
    """The images are already paid for, so odd advisory values are ignored."""
    result = parse_image(
        response(
            json={"images": [base64.b64encode(PNG).decode()]},
            headers={
                "x-venice-deprecated": "maybe",
                "x-venice-deprecated-replacement": " ",
                "x-venice-enhanced-prompt": "",
            },
        ),
        "m",
        None,
    )
    assert result.images[0].data == PNG
    assert result.deprecation is None and result.enhanced_prompt is None


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


@pytest.mark.parametrize(
    "header, expected",
    [
        (b"\xff\xf1\x50\x40", "aac"),  # ADTS, MPEG-4 AAC (ffmpeg -f adts)
        (b"\xff\xf9\x50\x40", "aac"),  # ADTS, MPEG-2 AAC
        (b"\xff\xfb\x50\xc4", "mp3"),  # MPEG-1 Layer III (ffmpeg -f mp3)
        (b"\xff\xf3\x50\xc4", "mp3"),  # MPEG-2 Layer III
        (b"\xff\xe3\x50\xc4", "mp3"),  # MPEG-2.5 Layer III
        (b"ID3\x04\x00", "mp3"),
    ],
)
def test_binary_speech_is_identified_by_frame_header(header, expected):
    """ADTS shares MPEG audio's sync bits; its zero layer field tells them apart."""
    result = parse_speech(
        response(
            content=header + b"\x00" * 16, headers={"content-type": "application/octet-stream"}
        ),
        "m",
        None,
    )
    assert result.format == expected


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


ERROR_FACTORIES = [
    lambda: VeniceError("Venice failed"),
    lambda: MissingKeyError("No key found for Venice"),
    lambda: VeniceConnectionError("Generating image failed: timeout"),
    lambda: VeniceResponseError("Generating image", "invalid JSON"),
    lambda: VeniceAPIError(
        "Listing models",
        500,
        "Internal Server Error",
        error_code="E500",
        detail="Inference failed",
        headers={"x-request-id": "req-1"},
    ),
    lambda: AuthenticationError("Listing API keys", 401, "Unauthorized", detail="wrong key"),
    lambda: RateLimitError(
        "Generating image", 429, "Too Many Requests", headers={"Retry-After": "12"}
    ),
    lambda: ContentViolationError(
        "Generating image",
        422,
        "Unprocessable Content",
        detail="Refused",
        suggested_prompt="A lighthouse",
        recommended_model="chroma",
        retryable=True,
    ),
]


@pytest.mark.parametrize("make_error", ERROR_FACTORIES, ids=lambda make: type(make()).__name__)
def test_errors_survive_pickle_and_copy(make_error):
    """Process pools pickle a worker's exception, which needs no constructor arguments."""
    error = make_error()
    error.add_note("while testing")
    restored = [
        pickle.loads(pickle.dumps(error, protocol))
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
    ]
    restored += [copy.copy(error), copy.deepcopy(error)]
    for value in restored:
        assert type(value) is type(error)
        assert value.args == error.args and str(value) == str(error)
        assert vars(value) == vars(error)
        assert value.__notes__ == ["while testing"]
        if isinstance(value, RateLimitError):
            assert value.retry_after == 12
        if isinstance(value, ContentViolationError):
            assert isinstance(error, ContentViolationError)
            assert value.describe(model_prefix="venice/") == error.describe(model_prefix="venice/")


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
