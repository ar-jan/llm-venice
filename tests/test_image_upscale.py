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
