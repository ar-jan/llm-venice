import json
from llm import get_key

import llm
from llm.cli import cli
import pytest
import sqlite_utils
from llm_venice import VeniceChat


@pytest.mark.integration
def test_prompt_web_search(cli_runner, isolated_llm_dir):
    """Test that the 'web_search on' option includes web_search_citations.

    Uses isolated_llm_dir fixture to ensure test doesn't modify user's actual logs.db
    """

    result = cli_runner.invoke(
        cli,
        [
            "prompt",
            "-m",
            "venice/llama-3.3-70b",
            "--web-search",
            "on",
            "--no-stream",
            "What is VVV by Venice AI?",
        ],
    )

    assert result.exit_code == 0

    # Get the response from the isolated test logs database
    # The isolated_llm_dir fixture ensures llm.user_dir() returns the temp directory
    logs_db_path = llm.user_dir() / "logs.db"
    assert logs_db_path.parent == isolated_llm_dir  # Verify we're using the temp dir

    db = sqlite_utils.Database(logs_db_path)
    last_response = list(db["responses"].rows)[-1]

    response_json = json.loads(last_response["response_json"])
    assert "venice_parameters" in response_json
    assert "web_search_citations" in response_json["venice_parameters"]

    citations = response_json["venice_parameters"]["web_search_citations"]
    assert isinstance(citations, list)
    assert len(citations) > 0


@pytest.mark.integration
def test_thinking_parameters_with_real_api(isolated_llm_dir):
    """Test that thinking parameters work correctly with the real Venice API using qwen3-4b model.

    This test requires a valid Venice API key to be set in the environment.
    It uses qwen3-4b which is a small, fast model suitable for testing.
    """
    # Skip test if no API key is available
    try:
        api_key = get_key(None, "venice", "LLM_VENICE_KEY")
        if not api_key:
            pytest.skip("No Venice API key available")
    except Exception:
        pytest.skip("No Venice API key available")

    # Use qwen3-4b model for testing
    chat = VeniceChat(
        model_id="venice/qwen3-4b",
        model_name="qwen3-4b",
        api_base="https://api.venice.ai/api/v1",
    )

    # Test with strip_thinking_response=True
    response = chat.prompt(
        "What is 2+2? Think step by step.",
        extra_body={"venice_parameters": {"strip_thinking_response": True}},
    )

    # Verify we got a response
    assert response is not None
    response_text = response.text()
    assert len(response_text) > 0, "Should have received a response"
    # The response should not contain <think> tags since strip_thinking_response=True
    assert "<think>" not in response_text.lower(), (
        "Response should not contain thinking tags when strip_thinking_response=True"
    )

    # Test with disable_thinking=True
    response2 = chat.prompt(
        "What is 3+3?", extra_body={"venice_parameters": {"disable_thinking": True}}
    )

    # Verify we got a response
    assert response2 is not None
    response2_text = response2.text()
    assert len(response2_text) > 0, "Should have received a response"

    # Test with both parameters
    response3 = chat.prompt(
        "What is 4+4?",
        extra_body={
            "venice_parameters": {
                "strip_thinking_response": True,
                "disable_thinking": True,
            }
        },
    )

    # Verify we got a response
    assert response3 is not None
    response3_text = response3.text()
    assert len(response3_text) > 0, "Should have received a response"
    assert "<think>" not in response3_text.lower(), (
        "Response should not contain thinking tags"
    )
