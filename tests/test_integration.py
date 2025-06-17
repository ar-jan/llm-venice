import json

from click.testing import CliRunner
import llm
from llm.cli import cli
import pytest
import sqlite_utils


@pytest.mark.integration
def test_prompt_web_search():
    """Test that the 'web_search on' option includes web_search_citations"""

    runner = CliRunner()
    result = runner.invoke(
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

    # Get the response from the logs database
    logs_db_path = llm.user_dir() / "logs.db"
    db = sqlite_utils.Database(logs_db_path)
    last_response = list(db["responses"].rows)[-1]

    response_json = json.loads(last_response["response_json"])
    assert "venice_parameters" in response_json
    assert "web_search_citations" in response_json["venice_parameters"]

    citations = response_json["venice_parameters"]["web_search_citations"]
    assert isinstance(citations, list)
    assert len(citations) > 0
