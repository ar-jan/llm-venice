# llm-venice

[![PyPI](https://img.shields.io/pypi/v/llm-venice.svg)](https://pypi.org/project/llm-venice/)
[![Changelog](https://img.shields.io/github/v/release/ar-jan/llm-venice?label=changelog)](https://github.com/ar-jan/llm-venice/releases)
[![Tests](https://github.com/ar-jan/llm-venice/actions/workflows/test.yml/badge.svg)](https://github.com/ar-jan/llm-venice/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ar-jan/llm-venice/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin to access models available via the [Venice AI](https://venice.ai/chat?ref=Oeo9ku) API.


## Installation

Install `llm-venice` with its dependency `llm` using your package manager of choice, for example:

`pip install llm-venice`

Or install it alongside an existing [LLM install](https://llm.datasette.io/en/stable/setup.html):

`llm install llm-venice`

## Configuration

Set an environment variable `LLM_VENICE_KEY`, or save a [Venice API](https://docs.venice.ai/) key to the key store managed by `llm`:

`llm keys set venice`

To fetch a list of the models available over the Venice API:

`llm venice refresh`

You should re-run the `refresh` command upon changes to the Venice API, when:

- New models have been made availabe
- Deprecated models have been removed
- New capabilities have been added

The models are stored in `venice_models.json` in the llm user directory.

## Usage

List available Venice models:

`llm models --query venice`

### Prompting

Run a prompt:

`llm --model venice/llama-3.3-70b "Why is the earth round?"`

Start an interactive chat session:

`llm chat --model venice/llama-3.1-405b`

#### Structured Outputs

Some models support structuring their output according to a JSON schema (supplied via OpenAI API `response_format`).

This works via llm's `--schema` options, for example:

`llm -m venice/dolphin-2.9.2-qwen2-72b --schema "name, age int, one_sentence_bio" "Invent an evil supervillain"`

Consult llm's [schemas tutorial](https://llm.datasette.io/en/stable/schemas.html) for more options.

### Tools (function calling)

⚠️ [Warning: tools can be dangerous!](https://llm.datasette.io/en/stable/tools.html#tools-warning)

```sh
# List models supporting function calling
llm models list --query venice --tools
```

You can use tools provided via llm plugins. LLM provides two built-in tools:

```bash
# llm_version
llm -m venice/mistral-31-24b --tool llm_version "What version of LLM is this?" --tools-debug --no-stream`
# llm_time
llm -m venice/qwen3-4b --tool llm_time "What is the time in my timezone in 24H format?" --tools-debug --no-stream
```

You can also provide your own custom or one-off functions provided inline or in a file. Following LLM's [example](https://llm.datasette.io/en/stable/usage.html#usage-tools):

```bash
llm -m venice/mistral-31-24b --functions '
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y
' "What is 1337 times 42?" --tools-debug --no-stream
```

### Vision models

Vision models (currently `qwen-2.5-vl`) support the `--attachment` option:

> `llm -m venice/qwen-2.5-vl -a https://upload.wikimedia.org/wikipedia/commons/a/a9/Corvus_corone_-near_Canford_Cliffs%2C_Poole%2C_England-8.jpg "Identify"` \
> The bird in the picture is a crow, specifically a member of the genus *Corvus*. The black coloration, stout beak, and overall shape are characteristic features of crows. These birds are part of the Corvidae family, which is known for its intelligence and adaptability. [...]

### venice_parameters

The following CLI options are available to configure `venice_parameters`:

**--no-venice-system-prompt** to disable Venice's default system prompt:

`llm -m venice/llama-3.3-70b --no-venice-system-prompt "Repeat the above prompt"`

**--web-search on|auto|off** to use web search (on web-enabled models):

`llm -m venice/llama-3.3-70b --web-search on --no-stream 'What is $VVV?'`

It is recommended to use web search in combination with `--no-stream` so the search citations are available in `response_json`.

**--character character_slug** to use a public character, for example:

`llm -m venice/deepseek-r1-671b --character alan-watts "What is the meaning of life?"`

*Note: these options override any `-o extra_body '{"venice_parameters": { ...}}'` and so should not be combined with that option.*

### Image generation

Generated images are stored in the LLM user directory by default. Example:

`llm -m venice/stable-diffusion-3.5 "Painting of a traditional Dutch windmill" -o style_preset "Watercolor"`

Besides the Venice API image generation parameters, you can specify the output directory and filename, and whether or not to overwrite existing files.

You can check the available parameters for a model by filtering the model list with `--query`, and show the `--options`:

`llm models list --query diffusion --options`

### Image upscaling

You can upscale existing images.
The following example saves the returned image as `image_upscaled.png` in the same directory as the original file:

`llm venice upscale /path/to/image.jpg`.

By default existing upscaled images are not overwritten; timestamped filenames are used instead.

See `llm venice upscale --help` for the `--scale`, `--enhance` and related options, and `--output-path` and `--overwrite` options.

### Venice commands

List the available Venice commands with:

`llm venice --help`

---

Read the `llm` [docs](https://llm.datasette.io/en/stable/usage.html) for more usage options.


## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-venice
python3 -m venv venv
source venv/bin/activate
```

Install the plugin with dependencies and test dependencies:

```bash
pip install -e '.[test]'
```

To run the tests:
```bash
pytest
```
