# Repository Guidelines

## Project Structure & Module Organization
- `llm_venice/` is the main plugin package.
  - `llm_venice/api/` contains Venice API clients and helpers.
  - `llm_venice/models/` contains model wrappers (chat, image, audio).
  - `llm_venice/cli/` contains CLI command implementations and options.
- `tests/` holds the pytest suite (unit + integration).
- `pyproject.toml` defines dependencies and tooling configuration.
- `README.md` documents end-user usage and examples.

## Build, Test, and Development Commands
- `uv venv` and `source .venv/bin/activate`: create and activate a local virtualenv.
- `uv pip install -e '.[test,dev]'`: install the plugin in editable mode with test/dev deps.
- `uv pip install pre-commit` and `pre-commit install`: enable pre-commit hooks if you use them.
- `pytest`: run the default test suite (integration tests are excluded by default).
- After install, a quick smoke check is `llm models --query venice`.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; line length 100.
- Ruff is configured for linting/formatting (see `pyproject.toml`). Use `ruff format .` and `ruff check .` before submitting.
- Pyright is configured for static type checks (optional but encouraged).
- Use snake_case for modules/functions, PascalCase for classes, and keep CLI option names consistent with existing commands.

## Testing Guidelines
- Tests live in `tests/` and follow `test_*.py` naming.
- Integration tests are marked with `@pytest.mark.integration` and require a Venice API key.
- Provide `LLM_VENICE_KEY` or `llm keys set venice` before running integration tests.
- Run integration tests explicitly: `pytest -m integration`.

## Commit & Pull Request Guidelines
- Recent commit history uses short, imperative subject lines (e.g., "Add ...", "Fix ..."); keep messages concise and scoped.
- PRs should include a summary, test command(s) run, and any relevant screenshots/logs when behavior changes.
- Link related issues and update tests or README when you add or change user-facing behavior.

## Configuration & Secrets
- Store API keys in `LLM_VENICE_KEY` or the llm key store (`llm keys set venice`).
- Avoid committing credentials; rely on environment variables or local key storage.
