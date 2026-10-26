# Repository Guidelines

- Use the existing virtual environment: `source .venv/bin/activate`
- If there is no venv, create it first with uv: `uv venv`
- Install llm-venice in editable mode with test and development dependencies: `uv pip install -e '.[test,dev]'`
- Run the following checks before considering a task completed:

```sh
# Run the tests:
pytest
# Run the formatter:
ruff format
# Run the linter:
ruff check
# Run static type checks:
pyright
# Type-check the way CI does (no venv path needed):
pyright --pythonpath "$(command -v python)"
```

CI (`.github/workflows/test.yml`) runs the same four checks on Python 3.10 and
3.14, plus two jobs that are not part of the local gate:

- `type-check` runs pyright on Python 3.10. `[tool.pyright]` points `venvPath` at
the local `.venv`, so CI passes `--pythonpath` explicitly.
- `test-against-llm-main` installs llm from git `main` and runs the suite. It is
non-blocking and also runs weekly, to catch upstream internal API changes.

The integration tests call the real API and are excluded by default (`-m
'not integration'`); run them with `pytest -m integration` and a key, or trigger
the manual `test-integration` workflow.
