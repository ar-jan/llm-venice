"""Shared fixtures for llm-venice tests"""

import json
import shutil
from unittest.mock import patch
from typing import Any, Callable, Optional

import llm
from llm import get_key
import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def no_show_responses(monkeypatch):
    """Keep a developer's HTTP-logging environment out of captured test output."""
    monkeypatch.delenv("LLM_VENICE_SHOW_RESPONSES", raising=False)
    monkeypatch.delenv("LLM_OPENAI_SHOW_RESPONSES", raising=False)


@pytest.fixture(autouse=True)
def hermetic_llm_user_path(request, tmp_path_factory, monkeypatch):
    """Point LLM_USER_PATH at a per-test directory.

    Without this, any code path that is not patched would write into the real
    user directory (images, audio, logs.db, model cache). Integration tests are
    exempt: they read the real key store. A test can still override the path by
    setting LLM_USER_PATH itself, since that runs after this autouse fixture.
    """
    if request.node.get_closest_marker("integration"):
        return
    user_path = tmp_path_factory.mktemp("llm-user-path")
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    return user_path


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
def hermetic_venice_model(tmp_path, monkeypatch):
    """Provide a hermetic Venice model catalog for CLI tests.

    Points LLM_USER_PATH at a temporary directory containing a minimal
    venice_models.json with a single fake text model, so CLI tests do not
    depend on the live Venice model catalog (model IDs are deprecated
    upstream over time, which has repeatedly broken CLI tests).

    Returns:
        A factory callable: ``hermetic_venice_model(capabilities={...})``
        writes the catalog and returns the model ID (``venice/test-model``).
        Calling it with no arguments uses empty capabilities.
    """

    def make_model(capabilities=None):
        model_id = "test-model"
        (tmp_path / "venice_models.json").write_text(
            json.dumps(
                [
                    {
                        "id": model_id,
                        "type": "text",
                        "model_spec": {"capabilities": capabilities or {}},
                    }
                ]
            )
        )
        monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
        return f"venice/{model_id}"

    return make_model


@pytest.fixture(scope="session")
def venice_api_key():
    """Return the Venice API key or skip all Venice integration tests.

    Using a session-scoped fixture ensures we only check for the key once and
    keeps individual tests free of boilerplate skip logic.
    """
    try:
        key = get_key(None, "venice", "LLM_VENICE_KEY")
    except Exception:
        pytest.skip("No Venice API key available")
    if not key:
        pytest.skip("No Venice API key available")
    return key


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
def mocked_responses(httpx2_mock, mock_image_file):
    """Set up mocked HTTP responses for the upscale API endpoint"""
    httpx2_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        content=b"upscaled image data",
        headers={"Content-Type": "image/jpeg"},
    )
    return httpx2_mock


@pytest.fixture
def isolated_llm_dir(tmp_path, monkeypatch):
    """Create isolated dir with copies of real config files.

    This allows to call the API with live user configuration
    without adding test responses to the user's real logs.db.
    """
    test_llm_dir = tmp_path / "llm_test_dir"
    test_llm_dir.mkdir()

    # Copy real config files from actual user dir
    real_user_dir = llm.user_dir()  # Get before we change it

    for config_file in ["keys.json", "venice_models.json"]:
        src = real_user_dir / config_file
        if src.exists():
            shutil.copy(src, test_llm_dir / config_file)

    monkeypatch.setenv("LLM_USER_PATH", str(test_llm_dir))
    return test_llm_dir


@pytest.fixture
def request_json(
    httpx2_mock,
) -> Callable[..., Any]:
    """Return a helper that decodes the JSON body of a recorded request.

    Usage: ``payload = request_json()`` for the last request, or
    ``request_json(0)`` for the first.
    """

    def decode(index: int = -1) -> Any:
        request = httpx2_mock.get_requests()[index]
        return json.loads(request.read().decode())

    return decode


@pytest.fixture
def hermetic_venice_catalog(hermetic_llm_user_path):
    """Provide a factory that writes a Venice model catalog anywhere in the model list.

    Usage: ``install([{"id": "img", "type": "image", "model_spec": {...}}])``
    returns the model IDs prefixed with ``venice/``.
    """

    def install(models: Optional[list] = None) -> list[str]:
        models = models or [
            {"id": "test-model", "type": "text", "model_spec": {"capabilities": {}}}
        ]
        (hermetic_llm_user_path / "venice_models.json").write_text(json.dumps(models))
        return [f"venice/{model['id']}" for model in models]

    return install
