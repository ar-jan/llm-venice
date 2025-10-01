import base64
from unittest.mock import Mock, MagicMock, patch

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
            assert results[0] == "Response marked as content violation; no image was returned."

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
    base64_encoded = base64.b64encode(raw_binary_content).decode('utf-8')

    # Test Case 1: return_binary=True - should use raw content
    options_binary = Mock()
    options_binary.model_dump.return_value = {
        "return_binary": True,
        "width": 1024,
        "height": 1024,
    }
    prompt.options = options_binary

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
    options_json = Mock()
    options_json.model_dump.return_value = {
        "return_binary": False,
        "width": 1024,
        "height": 1024,
    }
    prompt.options = options_json

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
