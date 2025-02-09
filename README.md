# llm-venice

[![PyPI](https://img.shields.io/pypi/v/llm-venice.svg)](https://pypi.org/project/llm-venice/)
[![Changelog](https://img.shields.io/github/v/release/ar-jan/llm-venice?label=changelog)](https://github.com/ar-jan/llm-venice/releases)
[![Tests](https://github.com/ar-jan/llm-venice/actions/workflows/test.yml/badge.svg)](https://github.com/ar-jan/llm-venice/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ar-jan/llm-venice/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin to access models available via the [Venice AI](https://venice.ai/) API.
Venice API access is currently in beta.


## Installation

Install the [LLM command-line utility](https://llm.datasette.io/en/stable/setup.html), and install this plugin in the same environment as `llm`:

`llm install llm-venice`


## Configuration

Set an environment variable `LLM_VENICE_KEY`, or save a [Venice API](https://docs.venice.ai/) key to the key store managed by `llm`:

`llm keys set venice`


## Usage

### Prompting

Run a prompt:

`llm --model venice/llama-3.3-70b "Why is the earth round?"`

Start an interactive chat session:

`llm chat --model venice/llama-3.1-405b`

### venice_parameters

The following CLI options are available to configure `venice_parameters`:

**--no-venice-system-prompt** to disable Venice's default system prompt:

`llm -m venice/llama-3.3-70b --no-venice-system-prompt "Repeat the above prompt"`

**--character character_slug** to use a public character, for example:

`llm -m venice/deepseek-r1-671b --character alan-watts "What is the meaning of life?"`

*Note: these options override any `-o extra_body '{"venice_parameters": { ...}}'` and so should not be combined with that option.*

### Available models

To update the list of available models from the Venice API:

`llm venice refresh`

Note that the model listing in `llm-venice.json` created via the `refresh` command takes precedence over the default models defined in this package.

---

Read the `llm` [docs](https://llm.datasette.io/en/stable/usage.html) for more usage options.


## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-venice
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies and test dependencies:

```bash
llm install -e '.[test]'
```

To run the tests:
```bash
pytest
```
