import asyncio
import base64
import datetime
from unittest.mock import Mock, MagicMock, patch
import pytest

import click
import llm
from pydantic import ValidationError
from click.testing import CliRunner

from llm_venice import AsyncVeniceImage, VeniceImage
from llm_venice.constants import ENDPOINT_IMAGE_GENERATE
from llm_venice.models.image import ImageGenerationResult
from llm_venice.notices import VeniceNotice
from llm_venice.cli.notices import emit_cli_notices


def add_image_response(
    httpx2_mock,
    *,
    json_data=None,
    content=b"",
    headers=None,
    status_code=200,
    is_reusable=False,
):
    """Register a mocked POST to the image generation endpoint."""
    if json_data is None and content and status_code == 200:
        json_data = {"images": [base64.b64encode(content).decode()], "request": {}, "timing": {}}
    kwargs = {"json": json_data} if json_data is not None else {"content": content}
    httpx2_mock.add_response(
        method="POST",
        url=ENDPOINT_IMAGE_GENERATE,
        status_code=status_code,
        headers=headers or {},
        is_reusable=is_reusable,
        **kwargs,
    )


@pytest.mark.parametrize("model_cls", [VeniceImage, AsyncVeniceImage])
def test_venice_image_honors_model_name(model_cls):
    """An explicit model_name is the API model ID; model_id only names the llm model."""
    model = model_cls("my-alias", model_name="real-model")
    assert model.model_id == "venice/my-alias"
    assert model.model_name == "real-model"


@pytest.mark.parametrize("model_cls", [VeniceImage, AsyncVeniceImage])
def test_venice_image_model_name_defaults_to_model_id(model_cls):
    model = model_cls("test-model")
    assert model.model_name == "test-model"


def test_venice_image_sends_model_name_in_payload(
    mock_venice_api_key, tmp_path, httpx2_mock, request_json
):
    """The request payload uses model_name, not the registered model_id."""
    model = VeniceImage("my-alias", model_name="real-model")
    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.options = VeniceImage.Options(output_dir=str(tmp_path))

    image_json = {
        "images": ["YmFzZTY0ZGF0YQ=="],
        "request": {"model": "real-model"},
        "timing": {},
    }
    add_image_response(httpx2_mock, json_data=image_json)
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        list(model.execute(prompt, False, MagicMock(), None))
    assert request_json()["model"] == "real-model"


@pytest.mark.parametrize("model_cls", [VeniceImage, AsyncVeniceImage])
def test_venice_image_empty_key_raises_needs_key(model_cls, httpx2_mock):
    """An empty key is rejected before any request is made."""
    model = model_cls("test-model")
    prompt = MagicMock()

    with patch.object(model, "get_key", return_value=""):
        with pytest.raises(llm.NeedsKeyException):
            result = model.execute(prompt, False, MagicMock(), None)
            if hasattr(result, "__anext__"):

                async def drain():
                    return [item async for item in result]

                asyncio.run(drain())
            else:
                list(result)

    assert httpx2_mock.get_requests() == []


def test_venice_image_format_in_payload(mock_venice_api_key, httpx2_mock, request_json):
    """Test that image format is correctly included in the API payload."""
    # Create a VeniceImage model instance
    model = VeniceImage("test-model")

    # Create a prompt object with the format option
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    image_json = {
        "images": [
            "YmFzZTY0ZGF0YQ=="  # "base64data" encoded with padding
        ],
        "request": {"model": "test-model"},
        "timing": {},
    }
    add_image_response(httpx2_mock, json_data=image_json, is_reusable=True)

    # Test with different format options
    for format_value in ["png", "jpeg", "webp"]:
        # Setup options that include the format
        options = Mock()
        options.model_dump.return_value = {
            "format": format_value,
            "width": 1024,
            "height": 1024,
        }
        prompt.options = options

        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            list(model.execute(prompt, False, MagicMock(), None))

            # Ensure the alias "format" is used instead of "image_format"

            # Extract and verify the payload
            payload = request_json()
            assert payload["format"] == format_value


def test_venice_image_aspect_ratio_and_resolution_in_payload(
    mock_venice_api_key, httpx2_mock, request_json
):
    """Test that aspect_ratio and resolution are included in the API payload."""
    model = VeniceImage("test-model")

    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.options = VeniceImage.Options(
        aspect_ratio="16:9",
        resolution="1K",
    )

    add_image_response(httpx2_mock, content=b"\x89PNG\r\n\x1a\n")
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        list(model.execute(prompt, False, MagicMock(), None))

    payload = request_json()
    assert payload["aspect_ratio"] == "16:9"
    assert payload["resolution"] == "1K"


def test_venice_image_enable_web_search_in_payload(mock_venice_api_key, httpx2_mock, request_json):
    """Test that enable_web_search is included as a top-level image payload field."""
    model = VeniceImage("test-model", supports_web_search=True)

    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.options = VeniceImage.Options(enable_web_search=True)

    add_image_response(httpx2_mock, content=b"\x89PNG\r\n\x1a\n")
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        list(model.execute(prompt, False, MagicMock(), None))

    payload = request_json()
    assert payload["enable_web_search"] is True


def test_venice_image_variants_in_payload(mock_venice_api_key, httpx2_mock, request_json):
    """Test that variants is included for JSON image responses."""
    model = VeniceImage("test-model")

    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.options = VeniceImage.Options(variants=3)

    base64_encoded = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("utf-8")

    image_json = {
        "images": [base64_encoded, base64_encoded, base64_encoded],
        "request": {"model": "test-model"},
        "timing": {},
    }
    add_image_response(httpx2_mock, json_data=image_json)
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        list(model.execute(prompt, False, MagicMock(), None))

    payload = request_json()
    assert payload["variants"] == 3


def test_venice_image_omits_unset_dimensions_from_payload(
    mock_venice_api_key, httpx2_mock, request_json
):
    """Test that unset width and height are omitted from the API payload."""
    model = VeniceImage("test-model")

    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.options = VeniceImage.Options(
        aspect_ratio="16:9",
        resolution="1K",
    )

    add_image_response(httpx2_mock, content=b"\x89PNG\r\n\x1a\n")
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        list(model.execute(prompt, False, MagicMock(), None))

    payload = request_json()
    assert "width" not in payload
    assert "height" not in payload


def test_venice_image_content_violation_handling(mock_venice_api_key, httpx2_mock):
    """Test that content violation responses are detected and reported."""
    # Create a VeniceImage model instance
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt with inappropriate content"

    # Setup minimal options
    options = Mock()
    options.model_dump.return_value = {
        "width": 1024,
        "height": 1024,
    }
    prompt.options = options

    # Mock the API call with content violation response
    add_image_response(httpx2_mock, content=b"", headers={"x-venice-is-content-violation": "true"})
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        # Execute and collect results
        results = list(model.execute(prompt, False, response, None))

        # Verify the appropriate error message was yielded
        assert len(results) == 1
        assert results[0] == "Response marked as content violation; no image was returned."

        # Verify the API was called
        assert len(httpx2_mock.get_requests()) == 1


def test_venice_image_web_search_capability_guard():
    """Requesting image web search on unsupported models should raise ModelError."""
    model = VeniceImage("test-model")
    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.options = VeniceImage.Options(enable_web_search=True)

    with patch.object(model, "get_key", return_value="test-key"):
        with pytest.raises(llm.ModelError, match="does not support web search"):
            list(model.execute(prompt, False, MagicMock(), None))


def test_venice_image_blurred_header_exposed_for_json_response(
    mock_venice_api_key, tmp_path, httpx2_mock
):
    """Blurred image responses should be surfaced via response_json metadata."""
    model = VeniceImage("test-model")

    prompt = MagicMock()
    prompt.prompt = "Test prompt with adult content"
    prompt.options = VeniceImage.Options()

    raw_binary_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    base64_encoded = base64.b64encode(raw_binary_content).decode("utf-8")

    image_json = {
        "images": [base64_encoded],
        "request": {"model": "test-model"},
        "timing": {"inference": 2.5},
    }
    add_image_response(httpx2_mock, json_data=image_json, headers={"x-venice-is-blurred": "true"})
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        with patch("llm_venice.models.image.llm.user_dir", return_value=tmp_path):
            results = list(model.execute(prompt, False, response, None))

    assert len(results) == 2
    assert (
        results[0]
        == "Warning: generated image was blurred because Safe Venice filtered adult material\n"
    )
    assert results[1].startswith("Image saved to ")
    assert response.response_json["request"]["model"] == "test-model"
    assert response.response_json["timing"]["inference"] == 2.5
    assert response.response_json["is_blurred"] is True


def test_venice_image_variants_save_all_images_and_expose_response_json(
    mock_venice_api_key, tmp_path, httpx2_mock
):
    """JSON image responses with variants should save every returned image."""
    model = VeniceImage("test-model")

    prompt = MagicMock()
    prompt.prompt = "Test variants"
    prompt.options = VeniceImage.Options(
        variants=2,
        output_dir=tmp_path,
        output_filename="variant.png",
    )

    first_image = b"\x89PNG\r\n\x1a\nfirst"
    second_image = b"\x89PNG\r\n\x1a\nsecond"
    first_encoded = base64.b64encode(first_image).decode("utf-8")
    second_encoded = base64.b64encode(second_image).decode("utf-8")

    image_json = {
        "images": [first_encoded, second_encoded],
        "request": {"model": "test-model"},
        "timing": {"inference": 2.5},
    }
    add_image_response(httpx2_mock, json_data=image_json)
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        results = list(model.execute(prompt, False, response, None))
    first_path = tmp_path / "variant_1.png"
    second_path = tmp_path / "variant_2.png"

    assert first_path.exists()
    assert first_path.read_bytes() == first_image
    assert second_path.exists()
    assert second_path.read_bytes() == second_image
    assert results == [f"Image saved to {first_path}\nImage saved to {second_path}"]
    assert response.response_json["request"]["model"] == "test-model"
    assert response.response_json["timing"]["inference"] == 2.5
    assert response.response_json["is_blurred"] is False


def test_venice_image_default_output_directory_creation(mock_venice_api_key, tmp_path, httpx2_mock):
    """Test that default output directory is created from llm.user_dir() when not specified."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test default directory"

    # Setup options without specifying output_dir
    options = Mock()
    options.model_dump.return_value = {
        "format": "png",
    }
    prompt.options = options

    add_image_response(httpx2_mock, content=b"\x89PNG\r\n\x1a\n")
    response = MagicMock()

    with patch("llm_venice.models.image.llm.user_dir", return_value=tmp_path):
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            list(model.execute(prompt, False, response, None))
    images_dir = tmp_path / "images"
    assert images_dir.exists()
    saved_files = list(images_dir.glob("*"))
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == b"\x89PNG\r\n\x1a\n"


def test_venice_image_default_filename_path(mock_venice_api_key, tmp_path, httpx2_mock):
    """Test that default filename path is generated under llm.user_dir()/images."""
    model = VeniceImage("test-model")

    prompt = MagicMock()
    prompt.prompt = "Test default filename"

    options = Mock()
    options.model_dump.return_value = {
        "format": "png",
    }
    prompt.options = options

    add_image_response(httpx2_mock, content=b"\x89PNG\r\n\x1a\n")
    fixed_now = datetime.datetime(2025, 10, 6, 18, 7, 5)

    with patch("llm_venice.models.image.llm.user_dir", return_value=tmp_path):
        with patch("llm_venice.utils.datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            with patch.object(model, "get_key", return_value=mock_venice_api_key):
                results = list(model.execute(prompt, False, MagicMock(), None))

    assert len(httpx2_mock.get_requests()) == 1
    expected_dir = tmp_path / "images"
    expected_filename = f"{fixed_now.strftime('%Y-%m-%dT%H-%M-%S')}_venice_{model.model_name}.png"
    expected_path = expected_dir / expected_filename

    assert expected_dir.exists()
    assert expected_path.exists()
    assert expected_path.read_bytes() == b"\x89PNG\r\n\x1a\n"
    assert results == [f"Image saved to {expected_path}"]


def test_existing_file_no_overwrite_adds_timestamp(mock_venice_api_key, tmp_path, httpx2_mock):
    """Test that existing file with overwrite_files=False gets timestamp appended."""
    model = VeniceImage("test-model")

    # Create an existing file
    existing_file = tmp_path / "test_image.png"
    existing_file.write_bytes(b"existing content")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options with overwrite_files=False
    options = Mock()
    options.model_dump.return_value = {
        "output_dir": str(tmp_path),
        "output_filename": "test_image.png",
        "overwrite_files": False,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    new_content = b"\x89PNG\r\n\x1a\nnew image content"
    add_image_response(httpx2_mock, content=new_content)
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        # Mock datetime to get predictable timestamp
        mock_datetime = Mock()
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000_123456"

        with patch("llm_venice.utils.datetime.datetime", mock_datetime):
            results = list(model.execute(prompt, False, response, None))

    # Verify original file is unchanged
    assert existing_file.read_bytes() == b"existing content"

    # Verify new file was created with timestamp
    expected_new_file = tmp_path / "test_image_20250101_120000_123456.png"
    assert expected_new_file.exists()
    assert expected_new_file.read_bytes() == new_content

    # Verify the success message includes the new filename
    assert len(results) == 1
    assert str(expected_new_file) in results[0]


def test_existing_file_with_overwrite_replaces_file(mock_venice_api_key, tmp_path, httpx2_mock):
    """Test that existing file with overwrite_files=True gets replaced."""
    model = VeniceImage("test-model")

    # Create an existing file
    existing_file = tmp_path / "test_image.png"
    existing_file.write_bytes(b"old content to be replaced")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options with overwrite_files=True
    options = Mock()
    options.model_dump.return_value = {
        "output_dir": str(tmp_path),
        "output_filename": "test_image.png",
        "overwrite_files": True,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    new_content = b"\x89PNG\r\n\x1a\nnew overwritten content"
    add_image_response(httpx2_mock, content=new_content)
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        results = list(model.execute(prompt, False, response, None))

    # Verify the file was replaced with new content
    assert existing_file.read_bytes() == new_content

    # Verify only one file exists (no timestamp version)
    files_in_dir = list(tmp_path.glob("test_image*.png"))
    assert len(files_in_dir) == 1
    assert files_in_dir[0] == existing_file

    # Verify the success message
    assert len(results) == 1
    assert str(existing_file) in results[0]


def test_non_writable_directory_raises_model_error(mock_venice_api_key, tmp_path, httpx2_mock):
    """Test that non-writable directory raises ModelError."""
    model = VeniceImage("test-model")

    # Create a directory and make it non-writable
    non_writable_dir = tmp_path / "readonly"
    non_writable_dir.mkdir()
    non_writable_dir.chmod(0o444)  # Read-only

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options pointing to non-writable directory
    options = Mock()
    options.model_dump.return_value = {
        "output_dir": str(non_writable_dir),
        "output_filename": "test_image.png",
        "overwrite_files": False,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        # Should raise ModelError before making API call
        with pytest.raises(llm.ModelError, match="is not a writable directory"):
            list(model.execute(prompt, False, response, None))

    assert httpx2_mock.get_requests() == []
    # Clean up: restore permissions so pytest can clean up tmp_path
    non_writable_dir.chmod(0o755)


def test_nonexistent_directory_is_created_when_saving(mock_venice_api_key, tmp_path, httpx2_mock):
    """Nonexistent output_dir should be created automatically when saving."""
    model = VeniceImage("test-model")

    # Point to a directory that doesn't exist
    nonexistent_dir = tmp_path / "does_not_exist"

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options pointing to nonexistent directory
    options = Mock()
    options.model_dump.return_value = {
        "output_dir": str(nonexistent_dir),
        "output_filename": "test_image.png",
        "overwrite_files": False,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    add_image_response(httpx2_mock, content=b"\x89PNG\r\n\x1a\n")
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        results = list(model.execute(prompt, False, response, None))

    assert len(httpx2_mock.get_requests()) == 1
    expected_output = nonexistent_dir / "test_image.png"
    assert expected_output.exists()
    assert expected_output.read_bytes() == b"\x89PNG\r\n\x1a\n"
    assert any(str(expected_output) in line for line in results)


def test_file_path_instead_of_directory_raises_model_error(
    mock_venice_api_key, tmp_path, httpx2_mock
):
    """Test that passing a file path instead of directory raises ModelError."""
    model = VeniceImage("test-model")

    # Create a file (not a directory)
    file_path = tmp_path / "somefile.txt"
    file_path.write_text("not a directory")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options pointing to a file instead of directory
    options = Mock()
    options.model_dump.return_value = {
        "output_dir": str(file_path),
        "output_filename": "test_image.png",
        "overwrite_files": False,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        # Should raise ModelError for file path instead of directory
        with pytest.raises(llm.ModelError, match="is not a writable directory"):
            list(model.execute(prompt, False, response, None))

    assert httpx2_mock.get_requests() == []


def test_timestamp_format_in_appended_filename(mock_venice_api_key, tmp_path, httpx2_mock):
    """Test that the timestamp format is correct when appending to existing file."""
    model = VeniceImage("test-model")

    # Create an existing file
    existing_file = tmp_path / "myimage.png"
    existing_file.write_bytes(b"existing")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options
    options = Mock()
    options.model_dump.return_value = {
        "output_dir": str(tmp_path),
        "output_filename": "myimage.png",
        "overwrite_files": False,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    add_image_response(httpx2_mock, content=b"new content")
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        list(model.execute(prompt, False, response, None))

    # Find the new file (should have timestamp pattern)
    new_files = [f for f in tmp_path.glob("myimage_*.png") if f != existing_file]
    assert len(new_files) == 1

    new_file = new_files[0]
    # Verify timestamp format: myimage_YYYYMMDD_HHMMSS_ffffff.png
    # Extract just the timestamp part
    timestamp_part = new_file.stem.replace("myimage_", "")

    # Should have format: YYYYMMDD_HHMMSS_ffffff
    parts = timestamp_part.split("_")
    assert len(parts) == 3
    assert len(parts[0]) == 8  # YYYYMMDD
    assert len(parts[1]) == 6  # HHMMSS
    assert len(parts[2]) == 6  # ffffff (microseconds)

    # Verify all parts are numeric
    assert all(part.isdigit() for part in parts)


def test_no_existing_file_no_timestamp(mock_venice_api_key, tmp_path, httpx2_mock):
    """Test that when file doesn't exist, no timestamp is added even with overwrite_files=False."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options - file doesn't exist yet
    options = Mock()
    options.model_dump.return_value = {
        "output_dir": str(tmp_path),
        "output_filename": "newfile.png",
        "overwrite_files": False,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    content = b"new file content"
    add_image_response(httpx2_mock, content=content)
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        list(model.execute(prompt, False, response, None))

    # Verify file was created without timestamp
    expected_file = tmp_path / "newfile.png"
    assert expected_file.exists()
    assert expected_file.read_bytes() == content

    # Verify no other files were created
    files_in_dir = list(tmp_path.glob("newfile*.png"))
    assert len(files_in_dir) == 1
    assert files_in_dir[0] == expected_file


def test_multiple_collisions_create_multiple_timestamped_files(
    mock_venice_api_key, tmp_path, httpx2_mock
):
    """Test that multiple executions with same filename create multiple timestamped files."""
    model = VeniceImage("test-model")

    # Create initial file
    original_file = tmp_path / "image.png"
    original_file.write_bytes(b"original")

    # Function to execute model with same filename
    def execute_model(content):
        prompt = MagicMock()
        prompt.prompt = "Test prompt"

        options = Mock()
        options.model_dump.return_value = {
            "output_dir": str(tmp_path),
            "output_filename": "image.png",
            "overwrite_files": False,
            "format": "png",
        }
        prompt.options = options

        add_image_response(httpx2_mock, content=content)
        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            list(model.execute(prompt, False, response, None))

    # Execute twice more
    execute_model(b"second")
    execute_model(b"third")

    # Should have 3 files total: original + 2 timestamped
    all_files = list(tmp_path.glob("image*.png"))
    assert len(all_files) == 3

    # Original file should be unchanged
    assert original_file.read_bytes() == b"original"

    # Two timestamped files should exist
    timestamped_files = [f for f in all_files if f != original_file]
    assert len(timestamped_files) == 2


def test_http_error_raises_model_error(mock_venice_api_key, httpx2_mock):
    """Test that HTTP errors from the API are raised as ModelError."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options with actual Pydantic model so payload is serialized via model_dump
    prompt.options = VeniceImage.Options()

    # The API returns a non-JSON error body
    httpx2_mock.add_response(
        method="POST",
        url=ENDPOINT_IMAGE_GENERATE,
        status_code=429,
        content=b"API Error: Rate limit exceeded",
    )

    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        # Should raise ModelError with API error message
        with pytest.raises(llm.ModelError, match="429.*Rate limit exceeded"):
            list(model.execute(prompt, False, response, None))


def test_invalid_base64_data_raises_model_error(mock_venice_api_key, httpx2_mock):
    """Test that invalid base64 data from JSON response raises ModelError."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options with  to trigger base64 decoding
    options = Mock()
    options.model_dump.return_value = {
        "format": "png",
    }
    prompt.options = options

    # Mock the API call with invalid base64 data - using non-string type
    # which will cause base64.b64decode to raise TypeError
    image_json = {
        "images": [
            12345  # Invalid: integer instead of base64 string - will cause TypeError
        ],
        "request": {"model": "test-model"},
        "timing": {},
    }
    add_image_response(httpx2_mock, json_data=image_json)
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        # Should raise ModelError about base64 decoding failure
        with pytest.raises(llm.ModelError, match="invalid base64 image"):
            list(model.execute(prompt, False, response, None))


def test_file_write_failure_raises_model_error(mock_venice_api_key, tmp_path, httpx2_mock):
    """Test that file write failures are caught and raised as ModelError."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Overwriting with an existing directory as the file name fails for real
    (tmp_path / "test.png").mkdir()

    options = Mock()
    options.model_dump.return_value = {
        "output_dir": str(tmp_path),
        "output_filename": "test.png",
        "overwrite_files": True,
        "format": "png",
    }
    prompt.options = options

    add_image_response(httpx2_mock, content=b"\x89PNG\r\n\x1a\n")
    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        # Should raise ModelError about file write failure
        with pytest.raises(llm.ModelError, match="test.png"):
            list(model.execute(prompt, False, response, None))


def test_http_500_error_with_json_body(mock_venice_api_key, httpx2_mock):
    """Test handling of 500 Internal Server Error with JSON error details."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options
    options = Mock()
    options.model_dump.return_value = {
        "format": "png",
    }
    prompt.options = options

    add_image_response(
        httpx2_mock,
        status_code=500,
        json_data={"error": "Internal server error", "code": "server_error"},
    )

    response = MagicMock()
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        with pytest.raises(llm.ModelError, match="500.*server_error"):
            list(model.execute(prompt, False, response, None))


def test_venice_image_options_defaults_and_validation():
    """Test that VeniceImage.Options has correct defaults and validation."""
    from llm_venice import VeniceImage

    # Test default values
    options = VeniceImage.Options()
    assert options.height is None
    assert options.width is None
    assert options.aspect_ratio is None
    assert options.resolution is None
    assert options.image_format == "png"
    assert options.hide_watermark is True
    assert "return_binary" not in type(options).model_fields
    assert options.variants is None
    assert options.safe_mode is False
    assert options.overwrite_files is False
    assert options.embed_exif_metadata is False
    assert options.negative_prompt is None
    assert options.style_preset is None
    assert options.steps is None
    assert options.cfg_scale is None
    assert options.seed is None
    assert options.lora_strength is None
    assert options.enable_web_search is None
    assert options.output_dir is None
    assert options.output_filename is None

    # Test field alias for image_format
    dumped = options.model_dump(by_alias=True)
    assert "format" in dumped
    assert dumped["format"] == "png"
    assert dumped["width"] is None
    assert dumped["height"] is None

    # Test validation - height bounds
    with pytest.raises(ValidationError):
        VeniceImage.Options(height=0)  # Below minimum
    with pytest.raises(ValidationError):
        VeniceImage.Options(height=1281)  # Above maximum

    # Test validation - width bounds
    with pytest.raises(ValidationError):
        VeniceImage.Options(width=0)  # Below minimum
    with pytest.raises(ValidationError):
        VeniceImage.Options(width=1281)  # Above maximum

    # Test validation - steps bounds
    assert VeniceImage.Options(steps=6).steps == 6
    assert VeniceImage.Options(steps=51).steps == 51

    # Test validation - cfg_scale bounds
    with pytest.raises(ValidationError):
        VeniceImage.Options(cfg_scale=0)  # Must be > 0
    with pytest.raises(ValidationError):
        VeniceImage.Options(cfg_scale=20.1)  # Above maximum

    # Test validation - seed bounds
    with pytest.raises(ValidationError):
        VeniceImage.Options(seed=-1000000000)  # Below minimum
    with pytest.raises(ValidationError):
        VeniceImage.Options(seed=1000000000)  # Above maximum

    # Test validation - lora_strength bounds
    with pytest.raises(ValidationError):
        VeniceImage.Options(lora_strength=-1)  # Below minimum
    with pytest.raises(ValidationError):
        VeniceImage.Options(lora_strength=101)  # Above maximum
    with pytest.raises(ValidationError):
        VeniceImage.Options(variants=0)  # Below minimum
    with pytest.raises(ValidationError):
        VeniceImage.Options(variants=5)  # Above maximum
    with pytest.raises(ValidationError):
        VeniceImage.Options.model_validate({"return_binary": True, "variants": 2})


def test_venice_image_keeps_aspect_ratio_when_support_is_unknown(
    mock_venice_api_key, tmp_path, httpx2_mock, request_json
):
    """Missing aspect-ratio metadata retains endpoint validation."""
    model = VeniceImage(
        "test-model",
        image_constraints={"resolutions": ["1K", "2K"]},
    )

    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.options = VeniceImage.Options(aspect_ratio="9:16")

    add_image_response(httpx2_mock, content=b"\x89PNG\r\n\x1a\n")
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        with patch("llm_venice.models.image.llm.user_dir", return_value=tmp_path):
            results = list(model.execute(prompt, False, MagicMock(), None))

    payload = request_json()
    assert payload["aspect_ratio"] == "9:16"
    assert results[0].startswith("Image saved to ")


def test_venice_image_keeps_resolution_when_support_is_unknown(
    mock_venice_api_key, tmp_path, httpx2_mock, request_json
):
    """Missing resolution metadata retains endpoint validation."""
    model = VeniceImage(
        "test-model",
        image_constraints={"aspectRatios": ["1:1", "16:9"]},
    )

    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.options = VeniceImage.Options(resolution="4K")

    add_image_response(httpx2_mock, content=b"\x89PNG\r\n\x1a\n")
    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        with patch("llm_venice.models.image.llm.user_dir", return_value=tmp_path):
            results = list(model.execute(prompt, False, MagicMock(), None))

    payload = request_json()
    assert payload["resolution"] == "4K"
    assert results[0].startswith("Image saved to ")


def test_image_generation_result_exposes_structured_notices():
    """ImageGenerationResult should retain structured notices."""
    result = ImageGenerationResult(
        notices=[
            VeniceNotice(
                level="info",
                message="dropped unsupported options for model 'test-model': aspect_ratio",
            )
        ],
    )

    assert result.notices == [
        VeniceNotice(
            level="info",
            message="dropped unsupported options for model 'test-model': aspect_ratio",
        )
    ]


def test_emit_cli_notices_surfaces_rendered_messages():
    """CLI notice helper should emit rendered messages to stderr."""

    @click.command()
    def cli():
        emit_cli_notices(
            [
                VeniceNotice(
                    level="info",
                    message="dropped unsupported options for model 'test-model': aspect_ratio",
                )
            ]
        )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert (
        result.stderr == "Info: dropped unsupported options for model 'test-model': aspect_ratio\n"
    )


def test_emit_cli_notices_surfaces_blurred_warning():
    """CLI notice helper should surface the blurred-image warning."""

    @click.command()
    def cli():
        emit_cli_notices(
            [
                VeniceNotice(
                    level="warning",
                    message=(
                        "generated image was blurred because Safe Venice filtered adult material"
                    ),
                )
            ]
        )

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert (
        result.stderr
        == "Warning: generated image was blurred because Safe Venice filtered adult material\n"
    )


def test_venice_image_rejects_invalid_aspect_ratio_for_supported_model(mock_venice_api_key):
    """Test invalid aspect_ratio values are still rejected for supporting models."""
    model = VeniceImage(
        "test-model",
        image_constraints={"aspectRatios": ["1:1", "16:9"]},
    )

    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.options = VeniceImage.Options(aspect_ratio="9:16")

    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        with pytest.raises(
            llm.ModelError,
            match="aspect_ratio",
        ):
            list(model.execute(prompt, False, MagicMock(), None))


def test_venice_image_rejects_invalid_resolution_for_supported_model(mock_venice_api_key):
    """Test invalid resolution values are still rejected for supporting models."""
    model = VeniceImage(
        "test-model",
        image_constraints={"resolutions": ["1K", "2K"]},
    )

    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.options = VeniceImage.Options(resolution="4K")

    with patch.object(model, "get_key", return_value=mock_venice_api_key):
        with pytest.raises(
            llm.ModelError,
            match="resolution",
        ):
            list(model.execute(prompt, False, MagicMock(), None))
