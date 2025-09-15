"""Shared fixtures for llm-venice tests"""

import pytest
from click.testing import CliRunner
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
def mocked_responses(httpx_mock, mock_image_file):
    """Set up mocked HTTP responses for the upscale API endpoint"""
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        content=b"upscaled image data",
        headers={"Content-Type": "image/jpeg"},
    )
    return httpx_mock
