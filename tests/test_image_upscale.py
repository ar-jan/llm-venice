from unittest.mock import patch, mock_open

import click
from click.testing import CliRunner
from llm.cli import cli
from llm_venice import image_upscale
import pytest


@pytest.fixture
def mock_image_file():
    return b"fake image data"


@pytest.fixture
def mocked_responses(httpx_mock, mock_image_file):
    """Set up mocked HTTP responses for the upscale API endpoint"""
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        content=b"upscaled image data",
        headers={"Content-Type": "image/jpeg"},
    )
    return httpx_mock


def test_upscale_function(mocked_responses, mock_image_file, tmp_path):
    """Test the image_upscale function directly"""
    # Create a mock image file
    test_img_path = tmp_path / "test.jpg"
    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    # Test the function
    with patch("llm.get_key", return_value="fake-key"):
        image_upscale(str(test_img_path), 2)

    # Verify the API was called correctly
    requests = mocked_responses.get_requests()
    assert len(requests) == 1
    request = requests[0]

    # Check the request headers
    assert request.headers["Authorization"] == "Bearer fake-key"

    # Check multipart form data
    assert "multipart/form-data" in request.headers["Content-Type"]

    # Check if the output file was created with the correct name
    expected_output_path = tmp_path / "test_upscaled.png"
    assert expected_output_path.exists()

    # Verify the content was written
    with open(expected_output_path, "rb") as f:
        assert f.read() == b"upscaled image data"


def test_upscale_command(mocked_responses, mock_image_file, tmp_path):
    """Test the CLI command for upscaling"""
    # Create a mock image file
    test_img_path = tmp_path / "test.jpg"
    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    # Run the CLI command
    runner = CliRunner()
    with patch("llm.get_key", return_value="fake-key"):
        result = runner.invoke(
            cli, ["venice", "upscale", str(test_img_path), "--scale", "4"]
        )

    # Verify the command completed successfully
    assert result.exit_code == 0

    # Verify the output message
    assert f"Upscaled image saved to {tmp_path}/test_upscaled.png" in result.output

    # Check the request was made with the correct scale factor
    requests = mocked_responses.get_requests()
    assert len(requests) == 1

    # Check that scale=4 was sent in the request
    request_body = requests[0].read()
    assert b'name="scale"' in request_body
    assert b"4" in request_body


def test_upscale_error_handling(httpx_mock):
    """Test error handling in the upscale function"""
    # Mock an error response from the API
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        status_code=400,
        json={"error": "Invalid request"},
    )

    # Create a temporary test file
    with patch("builtins.open", mock_open(read_data=b"fake image data")):
        with patch("llm.get_key", return_value="fake-key"):
            with pytest.raises(ValueError) as excinfo:
                image_upscale("test.jpg", 2)

            # Verify the error message includes the API response
            assert "API request failed" in str(excinfo.value)


def test_upscale_missing_api_key():
    """Test behavior when API key is missing"""
    # Mock get_key to return None to simulate missing API key
    with patch("llm.get_key", return_value=None):
        with pytest.raises(click.ClickException) as excinfo:
            image_upscale("test.jpg", 2)

        assert "No key found for Venice" in str(excinfo.value)
