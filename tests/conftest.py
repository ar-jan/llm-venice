import pytest


@pytest.fixture
def mock_image_file():
    """Provide fake image data for testing.

    Returns:
        bytes: Fake image data
    """
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
