"""Exercise fixture ordering with synthetic configuration and no API calls."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("marker_scope", ["function", "class", "module"])
def test_integration_copies_config_and_unit_tests_stay_isolated(tmp_path, marker_scope):
    source = tmp_path / "configured-user-dir"
    source.mkdir()
    config = {
        "keys.json": json.dumps({"venice": "synthetic-fixture-key"}),
        "venice_models.json": json.dumps([{"id": "test-model", "type": "text"}]),
    }
    for name, contents in config.items():
        (source / name).write_text(contents)

    project = tmp_path / "probe"
    project.mkdir()
    # Use the actual fixtures so this catches ordering regressions as well as
    # missing marker handling. A separate pytest process cannot inherit a
    # session-scoped credential fixture from the parent suite.
    (project / "conftest.py").write_text(Path(__file__).with_name("conftest.py").read_text())
    (project / "pytest.ini").write_text("[pytest]\nmarkers =\n    integration: fixture probe\n")
    unit_test = """import llm

def test_unit_isolation(hermetic_llm_user_path):
    assert llm.user_dir() == hermetic_llm_user_path
    assert llm.get_key(None, "venice", "LLM_VENICE_KEY") is None
    assert not (llm.user_dir() / "venice_models.json").exists()
    marker = llm.user_dir() / "unit-marker"
    assert not marker.exists()
    marker.write_text("unit output")
"""
    for name in ("test_before.py", "test_after.py"):
        (project / name).write_text(unit_test)

    body = """def test_config(isolated_llm_dir, venice_api_key):
    source = Path(os.environ["FIXTURE_PROBE_SOURCE"])
    assert llm.user_dir() == isolated_llm_dir
    assert isolated_llm_dir != source
    assert venice_api_key == "synthetic-fixture-key"
    assert llm.get_key(None, "venice", "LLM_VENICE_KEY") == venice_api_key
    for name in ("keys.json", "venice_models.json"):
        assert (isolated_llm_dir / name).read_bytes() == (source / name).read_bytes()
    (isolated_llm_dir / "logs.db").write_bytes(b"isolated test output")
    (isolated_llm_dir / "keys.json").write_text("{}")
"""
    if marker_scope == "class":
        body = "@pytest.mark.integration\nclass TestConfig:\n" + textwrap.indent(
            body.replace("def test_config(", "def test_config(self, ", 1), "    "
        )
    elif marker_scope == "module":
        body = "pytestmark = pytest.mark.integration\n\n" + body
    else:
        body = "@pytest.mark.integration\n" + body
    (project / "test_config.py").write_text(
        "import os\nfrom pathlib import Path\nimport llm\nimport pytest\n\n" + body
    )

    env = os.environ.copy()
    env.pop("LLM_VENICE_KEY", None)
    env["LLM_USER_PATH"] = str(source)
    env["FIXTURE_PROBE_SOURCE"] = str(source)
    env["PYTEST_ADDOPTS"] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "test_before.py",
            "test_config.py",
            "test_after.py",
        ],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
    assert {path.name: path.read_text() for path in source.iterdir()} == config
