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
```
