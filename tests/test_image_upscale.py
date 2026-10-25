from unittest.mock import patch

from llm.cli import cli
from llm_venice import VeniceAPIError, perform_image_upscale, write_upscaled_image
import pytest


def test_upscale_function(mocked_responses, temp_image_file, tmp_path, mock_venice_api_key):
    """Test the image_upscale function directly"""
    result = perform_image_upscale(str(temp_image_file), scale=2)
    saved_path = write_upscaled_image(result)

    # Verify the API was called correctly
    requests = mocked_responses.get_requests()
    assert len(requests) == 1
    request = requests[0]

    # Check the request headers
    assert request.headers["Authorization"] == f"Bearer {mock_venice_api_key}"

    # Check multipart form data
    assert "multipart/form-data" in request.headers["Content-Type"]

    # Check if the output file was created with the correct name
    # The mocked response is image/jpeg, so test.jpg results in test_upscaled.jpg
    expected_output_path = tmp_path / "test_upscaled.jpg"
    assert saved_path == expected_output_path
    assert expected_output_path.exists()

    # Verify the content was written
    with open(expected_output_path, "rb") as f:
        assert f.read() == b"upscaled image data"


def test_upscale_command(
    mocked_responses, temp_image_file, tmp_path, mock_venice_api_key, cli_runner
):
    """Test the CLI command for upscaling"""
    # Run the CLI command
    result = cli_runner.invoke(cli, ["venice", "upscale", str(temp_image_file), "--scale", "4"])

    # Verify the command completed successfully
    assert result.exit_code == 0

    # Verify the output message
    assert f"Upscaled image saved to {tmp_path}/test_upscaled.jpg" in result.output

    # Check the request was made with the correct scale factor
    requests = mocked_responses.get_requests()
    assert len(requests) == 1

    # Check that scale=4 was sent in the request
    request_body = requests[0].read()
    assert b'name="scale"' in request_body
    assert b"4" in request_body


def test_upscale_error_handling(httpx2_mock, mock_venice_api_key, temp_image_file):
    """Test error handling in the upscale function"""
    # Mock an error response from the API
    httpx2_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        status_code=400,
        json={"error": "Invalid request"},
    )

    with pytest.raises(VeniceAPIError) as excinfo:
        perform_image_upscale(str(temp_image_file), scale=2)

    # Verify the error message includes the status and detail
    message = str(excinfo.value)
    assert "400" in message
    assert "invalid request" in message.lower()


def test_upscale_missing_api_key():
    """Test behavior when API key is missing"""
    # Mock get_key to return None to simulate missing API key
    with patch("llm.get_key", return_value=None):
        from llm import NeedsKeyException

        with pytest.raises(NeedsKeyException) as excinfo:
            perform_image_upscale("test.jpg", scale=2)

        assert "No key found for Venice" in str(excinfo.value)


def test_upscale_with_enhancement_options(mocked_responses, temp_image_file, mock_venice_api_key):
    """Test upscaling with enhancement options enabled"""
    perform_image_upscale(
        str(temp_image_file),
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


def test_upscale_custom_output_path_file(
    mocked_responses, mock_image_file, tmp_path, mock_venice_api_key
):
    """Test upscaling with custom output file path"""
    test_img_path = tmp_path / "input.jpg"
    custom_output_path = tmp_path / "custom_output.png"

    with open(test_img_path, "wb") as f:
        f.write(mock_image_file)

    result = perform_image_upscale(str(test_img_path), scale=2, output_path=str(custom_output_path))
    write_upscaled_image(result)

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

    result = perform_image_upscale(str(test_img_path), scale=2, output_path=str(output_dir))
    write_upscaled_image(result)

    # Should save to output_dir with default filename
    expected_output = output_dir / "input_upscaled.jpg"
    assert expected_output.exists()


def test_upscale_avoid_overwrite_with_timestamp(
    mocked_responses, temp_image_file, tmp_path, mock_venice_api_key
):
    """Test that existing files are not overwritten without --overwrite flag"""
    # Expected output path (image/jpeg response): test.jpg -> test_upscaled.jpg
    existing_output = tmp_path / "test_upscaled.jpg"

    # Create existing output file
    with open(existing_output, "w") as f:
        f.write("existing content")

    result = perform_image_upscale(str(temp_image_file), scale=2, overwrite=False)
    write_upscaled_image(result)

    # Original file should be unchanged
    with open(existing_output, "r") as f:
        assert f.read() == "existing content"

    # A new file with timestamp should be created
    timestamped_files = list(tmp_path.glob("test_upscaled_*.jpg"))
    assert len(timestamped_files) == 1

    # New file should have the upscaled content
    with open(timestamped_files[0], "rb") as f:
        assert f.read() == b"upscaled image data"


def test_upscale_overwrite_existing_file(
    mocked_responses, temp_image_file, tmp_path, mock_venice_api_key
):
    """Test overwriting existing files when --overwrite is True"""
    # Expected output path (image/jpeg response): test.jpg -> test_upscaled.jpg
    existing_output = tmp_path / "test_upscaled.jpg"

    # Create existing output file
    with open(existing_output, "w") as f:
        f.write("existing content")

    result = perform_image_upscale(str(temp_image_file), scale=2, overwrite=True)
    write_upscaled_image(result)

    # File should be overwritten
    with open(existing_output, "rb") as f:
        assert f.read() == b"upscaled image data"

    # No timestamped files should exist
    timestamped_files = list(tmp_path.glob("test_upscaled_*.jpg"))
    assert len(timestamped_files) == 0


def test_upscale_cli_with_all_options(
    mocked_responses, temp_image_file, tmp_path, mock_venice_api_key, cli_runner
):
    """Test CLI command with all available options"""
    output_path = tmp_path / "custom_output.png"

    result = cli_runner.invoke(
        cli,
        [
            "venice",
            "upscale",
            str(temp_image_file),
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
        perform_image_upscale("nonexistent.jpg", scale=2)


def test_upscale_cli_file_not_found(mock_venice_api_key, cli_runner):
    """Test CLI error handling when input file doesn't exist"""
    result = cli_runner.invoke(cli, ["venice", "upscale", "nonexistent.jpg"])

    assert result.exit_code != 0
    assert "does not exist" in result.output or "No such file" in result.output


def test_upscale_network_timeout(httpx2_mock, temp_image_file, mock_venice_api_key):
    """Test handling of network timeouts"""
    import httpx2

    # Mock a timeout response
    httpx2_mock.add_exception(
        httpx2.TimeoutException("Request timed out"),
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
    )

    with pytest.raises(httpx2.TimeoutException):
        perform_image_upscale(str(temp_image_file), scale=2)


def test_upscale_network_error_cli(cli_runner, httpx2_mock, temp_image_file, mock_venice_api_key):
    """CLI should surface network errors without traceback."""
    import httpx2

    httpx2_mock.add_exception(
        httpx2.TimeoutException("Request timed out"),
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
    )

    result = cli_runner.invoke(cli, ["venice", "upscale", str(temp_image_file), "--scale", "2"])

    assert result.exit_code == 1
    output = result.output
    assert "timed out" in output.lower()
    assert "Traceback" not in output


def test_upscale_binary_response_handling(
    httpx2_mock, temp_image_file, tmp_path, mock_venice_api_key
):
    """Test that binary image data is handled correctly"""
    # Mock response with specific binary PNG data
    png_header = b"\x89PNG\r\n\x1a\n"
    mock_png_data = png_header + b"fake png data"

    httpx2_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        content=mock_png_data,
        headers={"Content-Type": "image/png"},
    )

    result = perform_image_upscale(str(temp_image_file), scale=2)
    write_upscaled_image(result)

    # Verify the output file was created with expected naming convention
    output_file = tmp_path / "test_upscaled.png"
    with open(output_file, "rb") as f:
        saved_data = f.read()
        assert saved_data == mock_png_data
        assert saved_data.startswith(png_header)


@pytest.mark.parametrize(
    ("content_type", "expected_name"),
    [
        ("image/png", "test_upscaled.png"),
        ("image/jpeg", "test_upscaled.jpg"),
        ("image/webp", "test_upscaled.webp"),
        ("IMAGE/WEBP; charset=binary", "test_upscaled.webp"),
        ("application/octet-stream", "test_upscaled.png"),
        (None, "test_upscaled.png"),
    ],
)
def test_upscale_extension_from_content_type(
    httpx2_mock, temp_image_file, tmp_path, mock_venice_api_key, content_type, expected_name
):
    """The default output name uses the extension matching the response Content-Type."""
    headers = {"Content-Type": content_type} if content_type else {}
    httpx2_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        content=b"image bytes",
        headers=headers,
    )

    result = perform_image_upscale(str(temp_image_file), scale=2)

    assert result.output_path == tmp_path / expected_name
    assert result.content_type == content_type


def test_upscale_explicit_output_name_keeps_extension(
    mocked_responses, temp_image_file, tmp_path, mock_venice_api_key
):
    """An explicit output file name is not renamed to match the Content-Type."""
    output_path = tmp_path / "chosen.png"

    result = perform_image_upscale(str(temp_image_file), scale=2, output_path=output_path)

    assert result.content_type == "image/jpeg"
    assert result.output_path == output_path


def test_upscale_scale_default_is_float():
    """--scale defaults to the float 2.0, not the string "2"."""
    from llm_venice.cli.upscale import create_upscale_command

    command = create_upscale_command()
    scale_option = next(p for p in command.params if p.name == "scale")
    assert scale_option.default == 2.0
    assert isinstance(scale_option.default, float)


def test_upscale_command_default_scale_sent(
    mocked_responses, temp_image_file, mock_venice_api_key, cli_runner
):
    """Without --scale, the request sends scale=2.0."""
    result = cli_runner.invoke(cli, ["venice", "upscale", str(temp_image_file)])

    assert result.exit_code == 0, result.output
    request_body = mocked_responses.get_requests()[0].read()
    assert b'name="scale"\r\n\r\n2.0' in request_body
