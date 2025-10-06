import base64
from unittest.mock import Mock, MagicMock, patch
import pytest

import httpx
from pydantic import ValidationError

from llm_venice import VeniceImage


def test_venice_image_format_in_payload(mock_venice_api_key):
    """Test that image format is correctly included in the API payload."""
    # Create a VeniceImage model instance
    model = VeniceImage("test-model")

    # Create a prompt object with the format option
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Test with different format options
    for format_value in ["png", "webp"]:
        # Setup options that include the format
        options = Mock()
        options.model_dump.return_value = {
            "format": format_value,
            "width": 1024,
            "height": 1024,
        }
        prompt.options = options

        # Mock the API call
        with patch("httpx.post") as mock_post:
            # Configure the mock response
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "images": [
                    "YmFzZTY0ZGF0YQ=="  # "base64data" encoded with padding
                ],
                "request": {"model": "test-model"},
                "timing": {},
            }
            mock_post.return_value = mock_response

            # Mock file operations
            with patch("pathlib.Path.write_bytes"):
                with patch.object(model, "get_key", return_value=mock_venice_api_key):
                    list(model.execute(prompt, False, MagicMock(), None))

                    # Verify model_dump was called with by_alias=True
                    # This ensures the alias "format" is used instead of "image_format"
                    options.model_dump.assert_called_once_with(by_alias=True)

                    mock_post.assert_called_once()
                    call_args = mock_post.call_args

                    # Extract and verify the payload
                    payload = call_args[1]["json"]
                    assert payload["format"] == format_value


def test_venice_image_content_violation_handling(mock_venice_api_key):
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
    with patch("httpx.post") as mock_post:
        # Configure the mock response with content violation header
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"x-venice-is-content-violation": "true"}
        mock_post.return_value = mock_response

        # Mock the response object and get_key
        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            # Execute and collect results
            results = list(model.execute(prompt, False, response, None))

            # Verify the appropriate error message was yielded
            assert len(results) == 1
            assert (
                results[0]
                == "Response marked as content violation; no image was returned."
            )

            # Verify the API was called
            mock_post.assert_called_once()


def test_venice_image_return_binary_vs_json_parsing(mock_venice_api_key):
    """Test return_binary=True uses raw content vs base64 JSON decoding."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test binary response"

    # Test data: raw binary content that would be returned
    raw_binary_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    base64_encoded = base64.b64encode(raw_binary_content).decode("utf-8")

    # Test Case 1: return_binary=True - should use raw content
    prompt.options = VeniceImage.Options(return_binary=True)

    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.content = raw_binary_content  # Raw binary
        mock_post.return_value = mock_response

        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            with patch("pathlib.Path.write_bytes") as mock_write:
                list(model.execute(prompt, False, response, None))

                # Verify raw content was written directly
                mock_write.assert_called_once()
                written_data = mock_write.call_args[0][0]
                assert written_data == raw_binary_content

                # Verify JSON parsing was NOT called
                mock_response.json.assert_not_called()

    # Test Case 2: return_binary=False - should parse JSON and decode base64
    prompt.options = VeniceImage.Options(return_binary=False)

    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.json.return_value = {
            "images": [base64_encoded],
            "request": {"model": "test-model", "seed": 12345},
            "timing": {"inference": 2.5},
        }
        mock_post.return_value = mock_response

        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            with patch("pathlib.Path.write_bytes") as mock_write:
                list(model.execute(prompt, False, response, None))

                # Verify JSON was parsed
                mock_response.json.assert_called_once()

                # Verify base64 was decoded and written
                mock_write.assert_called_once()
                written_data = mock_write.call_args[0][0]
                assert written_data == raw_binary_content

                # Verify response metadata was stored
                assert response.response_json["request"]["seed"] == 12345
                assert response.response_json["timing"]["inference"] == 2.5


def test_venice_image_default_output_directory_creation(mock_venice_api_key):
    """Test that default output directory is created from llm.user_dir() when not specified."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test default directory"

    # Setup options without specifying output_dir
    options = Mock()
    options.model_dump.return_value = {
        "return_binary": True,
        "format": "png",
    }
    prompt.options = options

    # Mock API response with binary content
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.content = b"\x89PNG\r\n\x1a\n"
        mock_post.return_value = mock_response

        response = MagicMock()

        # Mock llm.user_dir() to return a test path
        mock_user_dir = MagicMock()
        mock_images_dir = MagicMock()
        mock_user_dir.__truediv__ = Mock(return_value=mock_images_dir)

        with patch("llm_venice.llm.user_dir", return_value=mock_user_dir):
            with patch.object(model, "get_key", return_value=mock_venice_api_key):
                with patch("pathlib.Path.write_bytes"):
                    list(model.execute(prompt, False, response, None))

                    # Verify llm.user_dir() was called
                    mock_user_dir.__truediv__.assert_called_once_with("images")

                    # Verify mkdir was called with exist_ok=True
                    mock_images_dir.mkdir.assert_called_once_with(exist_ok=True)


def test_existing_file_no_overwrite_adds_timestamp(mock_venice_api_key, tmp_path):
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
        "return_binary": True,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    new_content = b"\x89PNG\r\n\x1a\nnew image content"
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.content = new_content
        mock_post.return_value = mock_response

        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            # Mock datetime to get predictable timestamp
            mock_datetime = Mock()
            mock_datetime.now.return_value.strftime.return_value = (
                "20250101_120000_123456"
            )

            with patch("llm_venice.datetime.datetime", mock_datetime):
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


def test_existing_file_with_overwrite_replaces_file(mock_venice_api_key, tmp_path):
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
        "return_binary": True,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    new_content = b"\x89PNG\r\n\x1a\nnew overwritten content"
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.content = new_content
        mock_post.return_value = mock_response

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


def test_non_writable_directory_raises_valueerror(mock_venice_api_key, tmp_path):
    """Test that non-writable directory raises ValueError."""
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
        "return_binary": True,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.content = b"\x89PNG\r\n\x1a\n"
        mock_post.return_value = mock_response

        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            # Should raise ValueError before making API call
            with pytest.raises(ValueError, match="is not a writable directory"):
                list(model.execute(prompt, False, response, None))

        mock_post.assert_not_called()

    # Clean up: restore permissions so pytest can clean up tmp_path
    non_writable_dir.chmod(0o755)


def test_nonexistent_directory_raises_valueerror(mock_venice_api_key, tmp_path):
    """Test that nonexistent directory raises ValueError."""
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
        "return_binary": True,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.content = b"\x89PNG\r\n\x1a\n"
        mock_post.return_value = mock_response

        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            # Should raise ValueError before making API call
            with pytest.raises(ValueError, match="is not a writable directory"):
                list(model.execute(prompt, False, response, None))

        mock_post.assert_not_called()


def test_file_path_instead_of_directory_raises_valueerror(
    mock_venice_api_key, tmp_path
):
    """Test that passing a file path instead of directory raises ValueError."""
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
        "return_binary": True,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.content = b"\x89PNG\r\n\x1a\n"
        mock_post.return_value = mock_response

        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            # Should raise ValueError for file path instead of directory
            with pytest.raises(ValueError, match="is not a writable directory"):
                list(model.execute(prompt, False, response, None))

        mock_post.assert_not_called()


def test_timestamp_format_in_appended_filename(mock_venice_api_key, tmp_path):
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
        "return_binary": True,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.content = b"new content"
        mock_post.return_value = mock_response

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


def test_no_existing_file_no_timestamp(mock_venice_api_key, tmp_path):
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
        "return_binary": True,
        "format": "png",
    }
    prompt.options = options

    # Mock API response
    content = b"new file content"
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.content = content
        mock_post.return_value = mock_response

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
    mock_venice_api_key, tmp_path
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
            "return_binary": True,
            "format": "png",
        }
        prompt.options = options

        with patch("httpx.post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {}
            mock_response.content = content
            mock_post.return_value = mock_response

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


def test_http_error_raises_valueerror(mock_venice_api_key):
    """Test that HTTP errors from the API are raised as ValueError."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options with actual Pydantic model so payload is serialized via model_dump
    prompt.options = VeniceImage.Options(return_binary=True)

    # Mock the API call with HTTP error
    with patch("httpx.post") as mock_post:
        # Configure the mock response to raise HTTPStatusError
        mock_response = Mock()
        mock_response.text = "API Error: Rate limit exceeded"
        mock_response.status_code = 429

        # Create an HTTPStatusError
        http_error = httpx.HTTPStatusError(
            "429 Client Error",
            request=Mock(),
            response=mock_response,
        )
        mock_response.raise_for_status.side_effect = http_error
        mock_post.return_value = mock_response

        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            # Should raise ValueError with API error message
            with pytest.raises(
                ValueError, match="API request failed:.*Rate limit exceeded"
            ):
                list(model.execute(prompt, False, response, None))


def test_invalid_base64_data_raises_valueerror(mock_venice_api_key):
    """Test that invalid base64 data from JSON response raises ValueError."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options with return_binary=False to trigger base64 decoding
    options = Mock()
    options.model_dump.return_value = {
        "return_binary": False,
        "format": "png",
    }
    prompt.options = options

    # Mock the API call with invalid base64 data - using non-string type
    # which will cause base64.b64decode to raise TypeError
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.json.return_value = {
            "images": [
                12345  # Invalid: integer instead of base64 string - will cause TypeError
            ],
            "request": {"model": "test-model"},
            "timing": {},
        }
        mock_post.return_value = mock_response

        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            # Should raise ValueError about base64 decoding failure
            with pytest.raises(
                ValueError, match="Failed to decode base64 image data"
            ):
                list(model.execute(prompt, False, response, None))


def test_file_write_failure_raises_valueerror(mock_venice_api_key, tmp_path):
    """Test that file write failures are caught and raised as ValueError."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options
    options = Mock()
    options.model_dump.return_value = {
        "output_dir": str(tmp_path),
        "output_filename": "test.png",
        "return_binary": True,
        "format": "png",
    }
    prompt.options = options

    # Mock the API call with valid response
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {}
        mock_response.content = b"\x89PNG\r\n\x1a\n"
        mock_post.return_value = mock_response

        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            # Mock write_bytes to raise an error
            with patch("pathlib.Path.write_bytes") as mock_write:
                mock_write.side_effect = PermissionError("Permission denied")

                # Should raise ValueError about file write failure
                with pytest.raises(ValueError, match="Failed to write image file"):
                    list(model.execute(prompt, False, response, None))


def test_http_500_error_with_json_body(mock_venice_api_key):
    """Test handling of 500 Internal Server Error with JSON error details."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options
    options = Mock()
    options.model_dump.return_value = {
        "return_binary": True,
        "format": "png",
    }
    prompt.options = options

    # Mock the API call with 500 error
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.text = (
            '{"error": "Internal server error", "code": "server_error"}'
        )
        mock_response.status_code = 500

        http_error = httpx.HTTPStatusError(
            "500 Server Error",
            request=Mock(),
            response=mock_response,
        )
        mock_response.raise_for_status.side_effect = http_error
        mock_post.return_value = mock_response

        response = MagicMock()
        with patch.object(model, "get_key", return_value=mock_venice_api_key):
            with pytest.raises(ValueError, match="API request failed:.*server_error"):
                list(model.execute(prompt, False, response, None))


def test_venice_image_options_defaults_and_validation():
    """Test that VeniceImage.Options has correct defaults and validation."""
    from llm_venice import VeniceImage

    # Test default values
    options = VeniceImage.Options()
    assert options.height == 1024
    assert options.width == 1024
    assert options.image_format == "png"
    assert options.hide_watermark is True
    assert options.return_binary is False
    assert options.safe_mode is False
    assert options.overwrite_files is False
    assert options.embed_exif_metadata is False
    assert options.negative_prompt is None
    assert options.style_preset is None
    assert options.steps is None
    assert options.cfg_scale is None
    assert options.seed is None
    assert options.lora_strength is None
    assert options.output_dir is None
    assert options.output_filename is None

    # Test field alias for image_format
    dumped = options.model_dump(by_alias=True)
    assert "format" in dumped
    assert dumped["format"] == "png"

    # Test validation - height bounds
    with pytest.raises(ValidationError):
        VeniceImage.Options(height=63)  # Below minimum
    with pytest.raises(ValidationError):
        VeniceImage.Options(height=1281)  # Above maximum

    # Test validation - width bounds
    with pytest.raises(ValidationError):
        VeniceImage.Options(width=63)  # Below minimum
    with pytest.raises(ValidationError):
        VeniceImage.Options(width=1281)  # Above maximum

    # Test validation - steps bounds
    with pytest.raises(ValidationError):
        VeniceImage.Options(steps=6)  # Below minimum
    with pytest.raises(ValidationError):
        VeniceImage.Options(steps=51)  # Above maximum

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


def test_venice_image_logging_client_usage(mock_venice_api_key, monkeypatch):
    """Test that LLM_VENICE_SHOW_RESPONSES enables logging_client for debugging."""
    model = VeniceImage("test-model")

    # Create a prompt object
    prompt = MagicMock()
    prompt.prompt = "Test prompt"

    # Setup options
    prompt.options = VeniceImage.Options(return_binary=True)

    # Set environment variable to enable logging
    monkeypatch.setenv("LLM_VENICE_SHOW_RESPONSES", "1")

    # Mock both httpx.post and logging_client
    mock_logging_client = Mock()
    mock_client_instance = Mock()
    mock_logging_client.return_value = mock_client_instance

    # Configure the mock response
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.headers = {}
    mock_response.content = b"\x89PNG\r\n\x1a\n"
    mock_client_instance.post.return_value = mock_response

    response = MagicMock()

    with patch("llm_venice.logging_client", mock_logging_client):
        with patch("llm_venice.httpx.post") as mock_httpx_post:
            with patch.object(model, "get_key", return_value=mock_venice_api_key):
                with patch("pathlib.Path.write_bytes"):
                    list(model.execute(prompt, False, response, None))

                    # Verify logging_client was called
                    mock_logging_client.assert_called_once()

                    # Verify the logging client's post method was called instead of httpx.post
                    mock_client_instance.post.assert_called_once()
                    call_args = mock_client_instance.post.call_args

                    # httpx.post should not be used when logging client is enabled
                    mock_httpx_post.assert_not_called()

                # Verify the call had the correct parameters
                assert call_args[0][0] == "https://api.venice.ai/api/v1/image/generate"
                assert "Authorization" in call_args[1]["headers"]
                assert call_args[1]["timeout"] == 120
