import pytest
from llm.cli import cli


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
            "4",
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


def test_upscale_cli_file_not_found(mock_venice_api_key, cli_runner):
    """Test CLI error handling when input file doesn't exist"""
    result = cli_runner.invoke(cli, ["venice", "upscale", "nonexistent.jpg"])

    assert result.exit_code != 0
    assert "does not exist" in result.output or "No such file" in result.output


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


def test_upscale_command_default_scale_sent(
    mocked_responses, temp_image_file, mock_venice_api_key, cli_runner
):
    """Without --scale, the request sends scale=2.0."""
    result = cli_runner.invoke(cli, ["venice", "upscale", str(temp_image_file)])

    assert result.exit_code == 0, result.output
    request_body = mocked_responses.get_requests()[0].read()
    assert b'name="scale"\r\n\r\n2.0' in request_body


UPSCALE_URL = "https://api.venice.ai/api/v1/image/upscale"
PNG = b"\x89PNG\r\n\x1a\nupscaled"


def test_upscale_cli_rejects_unsupported_scale(httpx2_mock, temp_image_file, cli_runner):
    """Only 2 and 4 are accepted; other values fail before any request."""
    result = cli_runner.invoke(cli, ["venice", "upscale", str(temp_image_file), "--scale", "3"])

    assert result.exit_code == 2
    assert "Invalid value for '--scale'" in result.output
    assert not httpx2_mock.get_requests()


def test_upscale_cli_output_directory_uses_default_name(
    mocked_responses, temp_image_file, tmp_path, mock_venice_api_key, cli_runner
):
    output_dir = tmp_path / "output_dir"
    output_dir.mkdir()

    result = cli_runner.invoke(
        cli, ["venice", "upscale", str(temp_image_file), "-o", str(output_dir)]
    )

    assert result.exit_code == 0, result.output
    expected = output_dir / "test_upscaled.jpg"
    assert f"Upscaled image saved to {expected}" in result.output
    assert expected.read_bytes() == b"upscaled image data"


def test_upscale_cli_explicit_file_keeps_name_and_creates_parents(
    mocked_responses, temp_image_file, tmp_path, mock_venice_api_key, cli_runner
):
    """An explicit name is used as given, even when its extension differs."""
    target = tmp_path / "new" / "nested" / "custom.png"

    result = cli_runner.invoke(cli, ["venice", "upscale", str(temp_image_file), "-o", str(target)])

    assert result.exit_code == 0, result.output
    assert f"Upscaled image saved to {target}" in result.output
    assert target.read_bytes() == b"upscaled image data"


@pytest.mark.parametrize("explicit", [False, True])
def test_upscale_cli_collision_uses_timestamped_name(
    mocked_responses, temp_image_file, tmp_path, mock_venice_api_key, cli_runner, explicit
):
    existing = tmp_path / ("custom.jpg" if explicit else "test_upscaled.jpg")
    existing.write_bytes(b"existing content")
    args = ["-o", str(existing)] if explicit else []

    result = cli_runner.invoke(cli, ["venice", "upscale", str(temp_image_file), *args])

    assert result.exit_code == 0, result.output
    assert existing.read_bytes() == b"existing content"
    timestamped = list(tmp_path.glob(f"{existing.stem}_*.jpg"))
    assert len(timestamped) == 1
    assert timestamped[0].read_bytes() == b"upscaled image data"
    assert f"Upscaled image saved to {timestamped[0]}" in result.output


def test_upscale_cli_overwrite_replaces_existing_file(
    mocked_responses, temp_image_file, tmp_path, mock_venice_api_key, cli_runner
):
    existing = tmp_path / "test_upscaled.jpg"
    existing.write_bytes(b"existing content")

    result = cli_runner.invoke(cli, ["venice", "upscale", str(temp_image_file), "--overwrite"])

    assert result.exit_code == 0, result.output
    assert existing.read_bytes() == b"upscaled image data"
    assert not list(tmp_path.glob("test_upscaled_*.jpg"))


@pytest.mark.parametrize(
    ("content_type", "content", "expected_name"),
    [
        ("image/png", b"data", "test_upscaled.png"),
        ("image/jpeg", b"data", "test_upscaled.jpg"),
        ("image/webp", b"data", "test_upscaled.webp"),
        ("IMAGE/WEBP; charset=binary", b"data", "test_upscaled.webp"),
        # Without a recognised content type, the image signature decides.
        ("application/octet-stream", PNG, "test_upscaled.png"),
        (None, PNG, "test_upscaled.png"),
    ],
)
def test_upscale_cli_extension_from_returned_format(
    httpx2_mock,
    temp_image_file,
    tmp_path,
    mock_venice_api_key,
    cli_runner,
    content_type,
    content,
    expected_name,
):
    headers = {"Content-Type": content_type} if content_type else {}
    httpx2_mock.add_response(method="POST", url=UPSCALE_URL, content=content, headers=headers)

    result = cli_runner.invoke(cli, ["venice", "upscale", str(temp_image_file)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / expected_name).read_bytes() == content


def test_upscale_cli_unknown_format_writes_nothing(
    httpx2_mock, temp_image_file, tmp_path, mock_venice_api_key, cli_runner
):
    httpx2_mock.add_response(
        method="POST",
        url=UPSCALE_URL,
        content=b"not an image",
        headers={"Content-Type": "application/octet-stream"},
    )

    result = cli_runner.invoke(cli, ["venice", "upscale", str(temp_image_file)])

    assert result.exit_code == 1
    assert "unknown output format" in result.output
    assert "Traceback" not in result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == ["test.jpg"]
