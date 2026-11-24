"""Image editing through /image/edit: contract, clients and CLI."""

import asyncio
import io
import json
import re
from pathlib import Path

import httpx2
import pytest
from llm.cli import cli
from pydantic import ValidationError
from typing_extensions import assert_type

from llm_venice import GeneratedImage, ImageEditResult
from llm_venice._endpoints import (
    ASPECT_RATIO_PATTERN,
    ImageEditInput,
    SingleImageEditInput,
    form_fields,
    parse_image_edit,
)
from llm_venice.client import AsyncVeniceClient, VeniceClient
from llm_venice.errors import VeniceAPIError, VeniceResponseError
from llm_venice.results import ModelDeprecation

EDIT_URL = "https://api.venice.ai/api/v1/image/edit"
FIXTURE = Path(__file__).parent / "fixtures/openapi-2026-11-22.json"
PNG = b"\x89PNG\r\n\x1a\nsource"
EDITED = b"\xff\xd8\xffedited jpeg"
IMAGE_URL = "https://example.com/photos/sky.png"
HEADERS = {
    "content-type": "image/jpeg",
    "x-venice-model-id": "qwen-image-2-1-turbo-edit",
    "x-venice-model-name": "Qwen Image 2.1 Turbo",
    "x-venice-is-blurred": "false",
    "x-venice-is-content-violation": "false",
}
CATALOG = [
    {
        "id": "edit-model",
        "type": "inpaint",
        "model_spec": {
            "constraints": {
                "aspectRatios": ["auto", "1:1", "16:9"],
                "resolutions": ["1K", "2K"],
                "qualities": ["low", "high"],
                "maxInputImages": 2,
                "promptCharacterLimit": 20,
            }
        },
    },
    {
        "id": "clear-model",
        "type": "inpaint",
        "model_spec": {"constraints": {"supportsTransparentBackground": True}},
    },
]


def invoke(is_async, method, *args, **kwargs):
    if is_async:

        async def run():
            async with AsyncVeniceClient(key="test") as client:
                return await getattr(client, method)(*args, **kwargs)

        return asyncio.run(run())
    with VeniceClient(key="test") as client:
        return getattr(client, method)(*args, **kwargs)


def respond(httpx2_mock, url=EDIT_URL, content=EDITED, headers=None):
    httpx2_mock.add_response(
        method="POST", url=url, content=content, headers={**HEADERS, **(headers or {})}
    )


def edited_response(content=EDITED, headers=None):
    return httpx2.Response(
        200, content=content, headers=headers or {}, request=httpx2.Request("POST", EDIT_URL)
    )


def test_body_omits_unset_options_and_names_the_model():
    assert ImageEditInput(prompt="p").body() == {"prompt": "p"}
    every = ImageEditInput(
        prompt="p",
        model="m",
        aspect_ratio="19.5:9",
        resolution="2K",
        output_format="webp",
        safe_mode=False,
        transparent_background=True,
        enhance_prompt=True,
        disable_prompt_optimization_thinking=False,
        anon_user_id="u",
    )
    assert every.body()["model"] == "m"
    assert every.body("modelId")["modelId"] == "m" and "model" not in every.body("modelId")
    assert form_fields(every.body()) == {
        "model": "m",
        "prompt": "p",
        "aspect_ratio": "19.5:9",
        "resolution": "2K",
        "output_format": "webp",
        "safe_mode": "false",
        "transparent_background": "true",
        "enhance_prompt": "true",
        "disable_prompt_optimization_thinking": "false",
        "anon_user_id": "u",
    }
    url = SingleImageEditInput(prompt="p", image_url=IMAGE_URL).body()
    assert url == {"image": IMAGE_URL, "prompt": "p"}


@pytest.mark.parametrize(
    "inputs",
    [
        {"prompt": ""},
        {"prompt": "x" * 32769},
        {"prompt": "p", "model": ""},
        {"prompt": "p", "aspect_ratio": "16x9"},
        {"prompt": "p", "aspect_ratio": "auto:1"},
        {"prompt": "p", "aspect_ratio": "1" * 31 + ":1"},
        {"prompt": "p", "resolution": ""},
        {"prompt": "p", "resolution": "x" * 11},
        {"prompt": "p", "output_format": "jpg"},
        {"prompt": "p", "safe_mode": "false"},
        {"prompt": "p", "anon_user_id": "a||b"},
        {"prompt": "p", "transparent_background": True, "output_format": "jpeg"},
        {"prompt": "p", "modelId": "m"},
        {"prompt": "p", "quality": "high"},
    ],
)
def test_options_are_validated(inputs):
    with pytest.raises(ValidationError):
        ImageEditInput.model_validate(inputs)


def test_request_matches_dated_fixture():
    fixture = json.loads(FIXTURE.read_text())
    schema = fixture["schemas"]["EditImageRequest"]
    properties = schema["properties"]
    every = SingleImageEditInput(
        prompt="p",
        model="m",
        image_url=IMAGE_URL,
        aspect_ratio="auto",
        resolution="1K",
        output_format="png",
        safe_mode=True,
        transparent_background=True,
        enhance_prompt=True,
        disable_prompt_optimization_thinking=True,
        anon_user_id="u",
    ).body()
    # modelId is the deprecated alias of model.
    assert set(every) == set(properties) - {"modelId"}
    assert set(schema["required"]) == {"image", "prompt"}
    ratio = properties["aspect_ratio"]
    assert (ratio["pattern"], ratio["maxLength"]) == (ASPECT_RATIO_PATTERN, 32)
    for value in ("auto", "16:9", "19.5:9", "2:1.5"):
        assert re.fullmatch(ratio["pattern"], value)
        ImageEditInput(prompt="p", aspect_ratio=value)
    assert properties["prompt"]["maxLength"] == 32768
    assert set(properties["output_format"]["enum"]) == {"jpeg", "png", "webp"}
    resolution = properties["resolution"]
    declared = ImageEditInput.model_json_schema()["properties"]["resolution"]["anyOf"][0]
    assert (declared["minLength"], declared["maxLength"]) == (
        resolution["minLength"],
        resolution["maxLength"],
    )
    resource = fixture["resources"]["POST /image/edit"]
    assert set(resource["requestBody"]["content"]) == {"application/json", "multipart/form-data"}
    headers = set(resource["response"]["headers"])
    assert {
        "x-venice-is-content-violation",
        "x-venice-model-id",
        "x-venice-model-name",
        "x-venice-deprecated",
        "x-venice-deprecated-replacement",
    } <= headers


def test_response_headers_are_reported():
    result = parse_image_edit(
        edited_response(
            headers={
                **HEADERS,
                "x-venice-is-blurred": "true",
                "x-venice-enhanced-prompt": "a%20blue%20sky%2C%20sunrise",
                "x-venice-deprecated": "true",
                "x-venice-deprecated-replacement": "new-edit",
            },
        ),
        "Editing image",
    )
    assert result == ImageEditResult(
        GeneratedImage(EDITED, "jpeg", "image/jpeg"),
        model_id="qwen-image-2-1-turbo-edit",
        model_name="Qwen Image 2.1 Turbo",
        blurred=True,
        enhanced_prompt="a blue sky, sunrise",
        deprecation=ModelDeprecation(deprecated=True, replacement="new-edit"),
    )
    bare = parse_image_edit(edited_response(PNG), "Editing image")
    assert bare == ImageEditResult(GeneratedImage(PNG, "png", None))


@pytest.mark.parametrize(
    "content,headers,detail",
    [
        (EDITED, {"x-venice-is-blurred": "maybe"}, "invalid x-venice-is-blurred header"),
        (b"", {}, "response did not include an image"),
        (b"", {"x-venice-is-content-violation": "true"}, "flagged a content violation"),
        (b"not an image", {}, "unknown output format"),
    ],
)
def test_malformed_responses(content, headers, detail):
    with pytest.raises(VeniceResponseError, match=detail):
        parse_image_edit(edited_response(content, headers), "Editing image")


def test_flagged_images_are_kept():
    result = parse_image_edit(
        edited_response(headers={**HEADERS, "x-venice-is-content-violation": "true"}),
        "Editing image",
    )
    assert result.content_violation and result.image.data == EDITED


@pytest.mark.parametrize("is_async", [False, True])
def test_local_images_are_uploaded_with_form_fields(
    is_async, httpx2_mock, multipart_parts, tmp_path
):
    path = tmp_path / "photo.png"
    path.write_bytes(PNG)
    respond(httpx2_mock)
    result = invoke(
        is_async,
        "edit_image",
        path,
        "make the sky purple",
        model="qwen-image-2-1-turbo-edit",
        safe_mode=False,
        aspect_ratio="16:9",
    )
    assert result.model_id == "qwen-image-2-1-turbo-edit"
    assert result.save(tmp_path / "out.jpg").read_bytes() == EDITED
    parts = multipart_parts()
    assert parts["image"] == [("photo.png", "image/png", PNG)]
    fields = {name: value[0][2] for name, value in parts.items() if name != "image"}
    assert fields == {
        "model": b"qwen-image-2-1-turbo-edit",
        "prompt": b"make the sky purple",
        "safe_mode": b"false",
        "aspect_ratio": b"16:9",
    }
    assert httpx2_mock.get_request().headers["authorization"] == "Bearer test"


@pytest.mark.parametrize("is_async", [False, True])
def test_bytes_and_files_are_uploaded(is_async, httpx2_mock, multipart_parts):
    respond(httpx2_mock)
    invoke(is_async, "edit_image", PNG, "p")
    handle = io.BytesIO(PNG)
    respond(httpx2_mock)
    invoke(is_async, "edit_image", handle, "p")
    assert not handle.closed
    for index in (0, 1):
        parts = multipart_parts(index)
        assert parts["image"] == [("image.png", "image/png", PNG)]
        assert set(parts) == {"image", "prompt"}


@pytest.mark.parametrize("is_async", [False, True])
def test_urls_are_sent_as_json(is_async, httpx2_mock, request_json):
    respond(httpx2_mock)
    invoke(is_async, "edit_image", IMAGE_URL, "p", enhance_prompt=True)
    assert httpx2_mock.get_request().headers["content-type"] == "application/json"
    assert request_json() == {"image": IMAGE_URL, "prompt": "p", "enhance_prompt": True}


@pytest.mark.parametrize("is_async", [False, True])
def test_validation_precedes_reading_and_http(is_async, httpx2_mock, tmp_path):
    missing = tmp_path / "missing.png"
    with pytest.raises(ValidationError):
        invoke(is_async, "edit_image", missing, "p", aspect_ratio="wide")
    with pytest.raises(ValidationError):
        invoke(is_async, "edit_image", missing, "p", unknown=True)
    with pytest.raises(ValidationError):
        invoke(is_async, "edit_image", "https://", "p")
    assert not httpx2_mock.get_requests()


@pytest.mark.parametrize("is_async", [False, True])
def test_api_errors(is_async, httpx2_mock):
    httpx2_mock.add_response(url=EDIT_URL, status_code=503, json={"error": "At capacity"})
    with pytest.raises(VeniceAPIError, match="Editing image failed: .*503.*At capacity"):
        invoke(is_async, "edit_image", PNG, "p")


def test_result_type(httpx2_mock):
    respond(httpx2_mock)
    with VeniceClient(key="test") as client:
        assert_type(client.edit_image(PNG, "p"), ImageEditResult)


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "photo.png"
    path.write_bytes(PNG)
    return path


def edit_cli(cli_runner, *args):
    return cli_runner.invoke(cli, ["venice", "edit", *map(str, args)])


def test_cli_edits_and_saves_next_to_the_image(
    cli_runner, httpx2_mock, mock_venice_api_key, multipart_parts, photo
):
    respond(httpx2_mock, headers={"x-venice-enhanced-prompt": "a%20purple%20sky"})
    result = edit_cli(
        cli_runner, photo, "make", "the sky purple", "-m", "venice/edit-model", "--enhance-prompt"
    )
    assert result.exit_code == 0, result.output
    saved = photo.with_name("photo_edited.jpg")
    assert result.stdout == f"Edited image saved to {saved}\n"
    assert result.stderr == "Enhanced prompt: a purple sky\n"
    assert saved.read_bytes() == EDITED
    fields = {name: value[0][2] for name, value in multipart_parts().items() if name != "image"}
    # Safe mode is off unless asked for, as for image models.
    assert fields == {
        "model": b"edit-model",
        "prompt": b"make the sky purple",
        "safe_mode": b"false",
        "enhance_prompt": b"true",
    }


def test_cli_sends_every_option(
    cli_runner, httpx2_mock, mock_venice_api_key, multipart_parts, photo, tmp_path
):
    respond(httpx2_mock, content=PNG, headers={"content-type": "image/png"})
    result = edit_cli(
        cli_runner,
        photo,
        "p",
        "--aspect-ratio",
        "auto",
        "--resolution",
        "2K",
        "--format",
        "png",
        "--safe-mode",
        "--transparent-background",
        "--disable-prompt-optimization-thinking",
        "--anon-user-id",
        "u",
        "-o",
        tmp_path / "exact.png",
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "exact.png").read_bytes() == PNG
    fields = {name: value[0][2] for name, value in multipart_parts().items() if name != "image"}
    assert fields == {
        "prompt": b"p",
        "aspect_ratio": b"auto",
        "resolution": b"2K",
        "output_format": b"png",
        "safe_mode": b"true",
        "transparent_background": b"true",
        "disable_prompt_optimization_thinking": b"true",
        "anon_user_id": b"u",
    }


def test_cli_edits_a_url_image(
    cli_runner, httpx2_mock, mock_venice_api_key, request_json, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    respond(httpx2_mock)
    result = edit_cli(cli_runner, IMAGE_URL, "p")
    assert result.exit_code == 0, result.output
    assert request_json() == {"image": IMAGE_URL, "prompt": "p", "safe_mode": False}
    assert (tmp_path / "sky_edited.jpg").read_bytes() == EDITED


def test_cli_warns_about_blurring_and_deprecation(
    cli_runner, httpx2_mock, mock_venice_api_key, photo
):
    respond(
        httpx2_mock,
        headers={
            "x-venice-is-blurred": "true",
            "x-venice-model-deprecation-warning": "Moving to v2",
            "x-venice-model-deprecation-date": "2027-01-01",
            "x-venice-deprecated-replacement": "new-edit",
        },
    )
    result = edit_cli(cli_runner, photo, "p")
    assert result.exit_code == 0, result.output
    assert result.stderr == (
        "Warning: edited image was blurred because Safe Venice filtered adult material\n"
        "Warning: qwen-image-2-1-turbo-edit is scheduled for deprecation on 2027-01-01:"
        " Moving to v2; use new-edit instead\n"
    )


def test_cli_does_not_save_a_content_violation(cli_runner, httpx2_mock, mock_venice_api_key, photo):
    respond(httpx2_mock, headers={"x-venice-is-content-violation": "true"})
    result = edit_cli(cli_runner, photo, "p")
    assert result.exit_code == 1
    assert "content violation; it was not saved" in result.output
    assert not photo.with_name("photo_edited.jpg").exists()


@pytest.mark.parametrize(
    "model,args,message",
    [
        (
            "edit-model",
            ["--aspect-ratio", "4:3"],
            "--aspect-ratio for edit-model must be one of: auto, 1:1, 16:9",
        ),
        (
            "edit-model",
            ["--resolution", "4K"],
            "--resolution for edit-model must be one of: 1K, 2K",
        ),
        (
            "edit-model",
            ["--transparent-background"],
            "edit-model does not support transparent backgrounds",
        ),
        # The catalog is checked first; a model without an entry gets the request checks.
        ("other", ["--aspect-ratio", "wide"], "Invalid value for --aspect-ratio: String should"),
        ("other", ["--format", "jpeg", "--transparent-background"], "output_format png or webp"),
    ],
)
def test_cli_checks_options_before_editing(
    cli_runner, httpx2_mock, photo, hermetic_venice_catalog, model, args, message
):
    hermetic_venice_catalog(CATALOG)
    result = edit_cli(cli_runner, photo, "p", "-m", model, *args)
    assert result.exit_code == 2
    assert message in result.output
    assert not httpx2_mock.get_requests()


def test_cli_checks_the_prompt_length_and_skips_unknown_models(
    cli_runner, httpx2_mock, mock_venice_api_key, photo, hermetic_venice_catalog
):
    hermetic_venice_catalog(CATALOG)
    result = edit_cli(cli_runner, photo, "x" * 21, "-m", "edit-model")
    assert result.exit_code == 2
    assert "edit-model accepts prompts of at most 20 characters" in result.output
    respond(httpx2_mock)
    result = edit_cli(cli_runner, photo, "x" * 21, "-m", "unknown", "--aspect-ratio", "4:3")
    assert result.exit_code == 0, result.output
    respond(httpx2_mock)
    result = edit_cli(cli_runner, photo, "p", "-m", "clear-model", "--transparent-background")
    assert result.exit_code == 0, result.output


def test_cli_errors(cli_runner, httpx2_mock, mock_venice_api_key, photo, tmp_path):
    result = edit_cli(cli_runner, tmp_path / "missing.png", "p")
    assert result.exit_code == 2
    assert "does not exist or is not a file" in result.output
    result = edit_cli(cli_runner, photo)
    assert result.exit_code == 2
    assert "Missing argument 'PROMPT...'" in result.output
    httpx2_mock.add_response(url=EDIT_URL, status_code=400, json={"error": "Image too small"})
    result = edit_cli(cli_runner, photo, "p")
    assert result.exit_code == 1
    assert result.output == (
        "Error: Editing image failed: Venice API returned 400 Bad Request - Image too small\n"
    )
