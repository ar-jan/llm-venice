import json
import os
import subprocess
import sys

import pytest
from llm.plugins import load_plugins, pm


def test_plugin_is_installed():
    load_plugins()
    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_venice" in names


# Loading the plugin and registering models must not import openai: llm (0.38+)
# defers that import because it slows down every llm command, even `llm --version`.
_NO_OPENAI_AT_LOAD = """
import sys
import llm.default_plugins.openai_models
if "openai" in sys.modules:
    print("llm imports openai itself")
    raise SystemExit(0)
import llm
models = {model.model_id for model in llm.get_models()}
assert "venice/test-model" in models, sorted(models)
assert "openai" not in sys.modules, "loading llm-venice imported openai"
print("ok")
"""


def test_loading_plugin_does_not_import_openai(tmp_path):
    (tmp_path / "venice_models.json").write_text(
        json.dumps([{"id": "test-model", "type": "text", "model_spec": {}}])
    )
    env = os.environ.copy()
    env.pop("LLM_VENICE_KEY", None)
    env["LLM_USER_PATH"] = str(tmp_path)
    # Other installed plugins may import openai themselves.
    env["LLM_LOAD_PLUGINS"] = "llm-venice"
    result = subprocess.run(
        [sys.executable, "-I", "-c", _NO_OPENAI_AT_LOAD],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if result.stdout.strip() == "llm imports openai itself":
        pytest.skip("llm imports openai at plugin load before 0.38")
    assert result.stdout.strip() == "ok"
