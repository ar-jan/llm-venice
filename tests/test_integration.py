import json

import llm
from llm.cli import cli
import pytest
import sqlite_utils


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
