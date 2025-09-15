from unittest.mock import patch

import click
from click.testing import CliRunner
from llm.cli import cli
from llm_venice import image_upscale
import pytest


def test_upscale_function(mocked_responses, mock_image_file, tmp_path, mock_venice_api_key):
    """Test the image_upscale function directly"""
    # Create a mock image file
    test_img_path = tmp_path / "test.jpg"
    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    # Test the function
    image_upscale(str(test_img_path), scale=2)

    # Verify the API was called correctly
    requests = mocked_responses.get_requests()
    assert len(requests) == 1
    request = requests[0]

    # Check the request headers
    assert request.headers["Authorization"] == f"Bearer {mock_venice_api_key}"

    # Check multipart form data
    assert "multipart/form-data" in request.headers["Content-Type"]

    # Check if the output file was created with the correct name
    expected_output_path = tmp_path / "test_upscaled.png"
    assert expected_output_path.exists()

    # Verify the content was written
    with open(expected_output_path, "rb") as f:
        assert f.read() == b"upscaled image data"


def test_upscale_command(mocked_responses, mock_image_file, tmp_path, mock_venice_api_key):
    """Test the CLI command for upscaling"""
    # Create a mock image file
    test_img_path = tmp_path / "test.jpg"
    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    # Run the CLI command
    runner = CliRunner()
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


def test_upscale_error_handling(httpx_mock, mock_venice_api_key, tmp_path):
    """Test error handling in the upscale function"""
    # Mock an error response from the API
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        status_code=400,
        json={"error": "Invalid request"},
    )

    # Create a temporary test file
    test_img_path = tmp_path / "test.jpg"
    with open(test_img_path, "wb") as f:
        f.write(b"fake image data")

    with pytest.raises(ValueError) as excinfo:
        image_upscale(str(test_img_path), scale=2)

    # Verify the error message includes the API response
    assert "API request failed" in str(excinfo.value)


def test_upscale_missing_api_key():
    """Test behavior when API key is missing"""
    # Mock get_key to return None to simulate missing API key
    with patch("llm.get_key", return_value=None):
        with pytest.raises(click.ClickException) as excinfo:
            image_upscale("test.jpg", scale=2)

        assert "No key found for Venice" in str(excinfo.value)


def test_upscale_with_enhancement_options(mocked_responses, mock_image_file, tmp_path, mock_venice_api_key):
    """Test upscaling with enhancement options enabled"""
    test_img_path = tmp_path / "test.jpg"
    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    image_upscale(
            str(test_img_path),
            scale=3,
            enhance=True,
            enhance_creativity=0.7,
            enhance_prompt="spooky",
            replication=0.5,
        )

    # Verify the API was called with enhancement parameters
    requests = mocked_responses.get_requests()
    assert len(requests) == 1
    request_body = requests[0].read()

    assert b'name="enhance"\r\n\r\ntrue' in request_body
    assert b'name="enhanceCreativity"\r\n\r\n0.7' in request_body
    assert b'name="enhancePrompt"\r\n\r\nspooky' in request_body
    assert b'name="replication"\r\n\r\n0.5' in request_body


def test_upscale_custom_output_path_file(mocked_responses, mock_image_file, tmp_path, mock_venice_api_key):
    """Test upscaling with custom output file path"""
    test_img_path = tmp_path / "input.jpg"
    custom_output_path = tmp_path / "custom_output.png"

    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    image_upscale(str(test_img_path), scale=2, output_path=str(custom_output_path))

    # Verify the custom output path was used
    assert custom_output_path.exists()
    with open(custom_output_path, "rb") as f:
        assert f.read() == b"upscaled image data"


def test_upscale_custom_output_path_directory(
    mocked_responses, mock_image_file, tmp_path, mock_venice_api_key
):
    """Test upscaling with custom output directory"""
    test_img_path = tmp_path / "input.jpg"
    output_dir = tmp_path / "output_dir"
    output_dir.mkdir()

    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    image_upscale(str(test_img_path), scale=2, output_path=str(output_dir))

    # Should save to output_dir with default filename
    expected_output = output_dir / "input_upscaled.png"
    assert expected_output.exists()


def test_upscale_avoid_overwrite_with_timestamp(
    mocked_responses, mock_image_file, tmp_path, mock_venice_api_key
):
    """Test that existing files are not overwritten without --overwrite flag"""
    test_img_path = tmp_path / "test.jpg"
    existing_output = tmp_path / "test_upscaled.png"

    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    # Create existing output file
    with open(existing_output, "w") as f:
        f.write("existing content")

    image_upscale(str(test_img_path), scale=2, overwrite=False)

    # Original file should be unchanged
    with open(existing_output, "r") as f:
        assert f.read() == "existing content"

    # A new file with timestamp should be created
    timestamped_files = list(tmp_path.glob("test_upscaled_*.png"))
    assert len(timestamped_files) == 1

    # New file should have the upscaled content
    with open(timestamped_files[0], "rb") as f:
        assert f.read() == b"upscaled image data"


def test_upscale_overwrite_existing_file(mocked_responses, mock_image_file, tmp_path, mock_venice_api_key):
    """Test overwriting existing files when --overwrite is True"""
    test_img_path = tmp_path / "test.jpg"
    existing_output = tmp_path / "test_upscaled.png"

    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    # Create existing output file
    with open(existing_output, "w") as f:
        f.write("existing content")

    image_upscale(str(test_img_path), scale=2, overwrite=True)

    # File should be overwritten
    with open(existing_output, "rb") as f:
        assert f.read() == b"upscaled image data"

    # No timestamped files should exist
    timestamped_files = list(tmp_path.glob("test_upscaled_*.png"))
    assert len(timestamped_files) == 0


def test_upscale_cli_with_all_options(mocked_responses, mock_image_file, tmp_path, mock_venice_api_key):
    """Test CLI command with all available options"""
    test_img_path = tmp_path / "test.jpg"
    output_path = tmp_path / "custom_output.png"

    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    runner = CliRunner()
    result = runner.invoke(
            cli,
            [
                "venice",
                "upscale",
                str(test_img_path),
                "--scale",
                "3.5",
                "--enhance",
                "--enhance-creativity",
                "0.8",
                "--enhance-prompt",
                "spooky",
                "--replication",
                "0.3",
                "--output-path",
                str(output_path),
                "--overwrite",
            ],
        )

    assert result.exit_code == 0
    assert f"Upscaled image saved to {output_path}" in result.output

    # Verify all parameters were sent
    requests = mocked_responses.get_requests()
    assert len(requests) == 1
    request_body = requests[0].read()
    assert b'name="enhance"\r\n\r\ntrue' in request_body
    assert b'name="enhanceCreativity"\r\n\r\n0.8' in request_body
    assert b'name="enhancePrompt"\r\n\r\nspooky' in request_body
    assert b'name="replication"\r\n\r\n0.3' in request_body


def test_upscale_file_not_found(mock_venice_api_key):
    """Test error handling when input file doesn't exist"""
    with pytest.raises(FileNotFoundError):
        image_upscale("nonexistent.jpg", scale=2)


def test_upscale_cli_file_not_found(mock_venice_api_key):
    """Test CLI error handling when input file doesn't exist"""
    runner = CliRunner()
    result = runner.invoke(cli, ["venice", "upscale", "nonexistent.jpg"])

    assert result.exit_code != 0
    assert "does not exist" in result.output or "No such file" in result.output


def test_upscale_network_timeout(httpx_mock, mock_image_file, tmp_path, mock_venice_api_key):
    """Test handling of network timeouts"""
    import httpx

    # Mock a timeout response
    httpx_mock.add_exception(
        httpx.TimeoutException("Request timed out"),
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
    )

    test_img_path = tmp_path / "test.jpg"
    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    with pytest.raises(httpx.TimeoutException):
        image_upscale(str(test_img_path), scale=2)


def test_upscale_binary_response_handling(httpx_mock, mock_image_file, tmp_path, mock_venice_api_key):
    """Test that binary image data is handled correctly"""
    # Mock response with specific binary PNG data
    png_header = b"\x89PNG\r\n\x1a\n"
    mock_png_data = png_header + b"fake png data"

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        content=mock_png_data,
        headers={"Content-Type": "image/png"},
    )

    test_img_path = tmp_path / "test.jpg"
    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    image_upscale(str(test_img_path), scale=2)

    output_file = tmp_path / "test_upscaled.png"
    with open(output_file, "rb") as f:
        saved_data = f.read()
        assert saved_data == mock_png_data
        assert saved_data.startswith(png_header)
