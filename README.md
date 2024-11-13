# llm-venice

[LLM](https://llm.datasette.io/) plugin to access [Venice AI](https://venice.ai/) models.


## Installation

Install the [LLM command-line utility](https://llm.datasette.io/en/stable/setup.html), and install this plugin in the same environment as `llm`.

`llm install llm-venice`


## Configuration

Set an environment variable `LLM_VENICE_KEY`, or save a Venice API key to `llm`'s keys:

`llm keys set venice`


## Usage

Run a prompt:

`llm --model venice/nous-theta-8b "Why is the earth round?"`

Start an interactive chat session:

`llm chat --model venice/llama-3.1-405b`

Read the [docs](https://llm.datasette.io/en/stable/usage.html) for more `llm` usage options.


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
