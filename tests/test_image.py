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

                    mock_post.assert_called_once()
                    call_args = mock_post.call_args

                    # Extract and verify the payload
                    payload = call_args[1]["json"]
                    assert payload["format"] == format_value
