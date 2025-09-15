"""Shared fixtures for llm-venice tests"""

import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def mock_venice_api_key():
    """Mock llm.get_key to return a fake Venice API key.

    This fixture automatically patches llm.get_key for the duration of the test,
    ensuring consistent API key mocking across all tests.

    Returns:
        str: The fake API key value being used
    """
    fake_key = "fake-venice-api-key"
    with patch("llm.get_key", return_value=fake_key):
        yield fake_key


@pytest.fixture
def mock_image_file():
    """Provide fake image data for testing.

    Returns:
        bytes: Fake image data
    """
    return b"fake image data"


@pytest.fixture
def cli_runner():
    """Provide a configured CliRunner instance for testing CLI commands.

    Returns:
        CliRunner: A Click test runner configured for isolated testing
    """
    return CliRunner()


@pytest.fixture
def temp_image_file(tmp_path, mock_image_file):
    """Create a temporary image file on disk for testing.

    This fixture creates a real file on disk with image data that can be
    used for testing file operations, uploads, etc.

    The file is created as 'test.jpg' in the temporary directory.

    Args:
        tmp_path: pytest's tmp_path fixture for temporary directories
        mock_image_file: The mock image data to write to the file

    Returns:
        Path: Path object pointing to the created temporary image file (test.jpg)
    """
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(mock_image_file)
    return image_path


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
