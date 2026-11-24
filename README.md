# llm-venice

[![PyPI](https://img.shields.io/pypi/v/llm-venice.svg)](https://pypi.org/project/llm-venice/)
[![Changelog](https://img.shields.io/github/v/release/ar-jan/llm-venice?label=changelog)](https://github.com/ar-jan/llm-venice/releases)
[![Tests](https://github.com/ar-jan/llm-venice/actions/workflows/test.yml/badge.svg)](https://github.com/ar-jan/llm-venice/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ar-jan/llm-venice/blob/develop/LICENSE)

[LLM](https://llm.datasette.io/) plugin for the [Venice AI](https://venice.ai/chat?ref=Oeo9ku) API — chat, vision, audio input, web & X search, image generation, upscaling, text-to-speech, and embeddings.

## Installation

Install `llm-venice` (and `llm` with it) using your package manager:

```bash
pip install llm-venice
```

Or install into an existing [LLM installation](https://llm.datasette.io/en/stable/setup.html):

```bash
llm install llm-venice
```

## Configuration

Set `LLM_VENICE_KEY`, or store a [Venice API](https://docs.venice.ai/) key in `llm`'s key store:

```bash
llm keys set venice
```

Fetch the current Venice model catalog:

```bash
llm venice refresh
```

Re-run `llm venice refresh` whenever Venice adds, updates, or deprecates models. Models are cached in `venice_models.json` in the `llm` user directory.

List available Venice models:

```bash
llm models --query venice
```

## Usage

### Prompting & chat

```bash
llm -m venice/qwen-3-8-flash "Why is the earth round?"
llm chat -m venice/minimax-m3-1-flash
```

#### Structured outputs

Models with schema support work with `llm`'s `--schema` option (see the [LLM schemas tutorial](https://llm.datasette.io/en/stable/schemas.html)):

```bash
llm -m venice/z-ai-glm-5-3 --schema "name, age int, one_sentence_bio" "Invent an evil supervillain"
```

#### Attachments (vision & audio)

Vision models (such as `qwen-3-8-27b`) accept image attachments via `-a/--attachment`:

```bash
llm -m venice/qwen-3-8-27b -a https://upload.wikimedia.org/wikipedia/commons/a/a9/Corvus_corone_-near_Canford_Cliffs%2C_Poole%2C_England-8.jpg "Identify"
```

Models with `supportsAudioInput` also accept WAV and MP3 attachments:

```bash
llm -m venice/<audio-model> -a recording.wav "Transcribe this"
```

### Venice options (`-o`)

Configure Venice-specific parameters with `llm`'s `-o/--option KEY VALUE` flag. Run `llm models --query <model> --options` to inspect what a model accepts.

```bash
# Disable Venice's default system prompt
llm -m venice/qwen-3-8-flash -o include_venice_system_prompt 0 "Repeat the above prompt"

# Enable web search (citations are kept in response_json)
llm -m venice/qwen-3-8-flash -o enable_web_search on 'What is $VVV?'

# Enable xAI native web + X/Twitter search on supporting models
llm -m venice/grok-4-7 -o enable_x_search true --no-stream "What is trending about Venice AI?"

# Let Venice scrape URLs in your prompt
llm -m venice/qwen-3-8-flash -o enable_web_scraping 1 "Summarize https://venice.ai"

# Use a public character
llm -m venice/google-gemma-4-26b-a4b-it -o character_slug alan-watts "What is the meaning of life?"

# Send API fields that do not have a dedicated option yet (also works on image & speech models)
llm -m venice/qwen-3-8-flash -o extra_body '{"venice_parameters": {"new_flag": true}}' "Hello"
```

See [CLI reference — Venice options](https://github.com/ar-jan/llm-venice/blob/develop/docs/cli.md#venice-options) for precedence rules and `extra_body` details.

### Tools & search

> ⚠️ [Warning: tools can be dangerous!](https://llm.datasette.io/en/stable/tools.html#tools-warning)

#### Function calling

List Venice models that support function calling and pass built-in, plugin, or inline `--functions` tools:

```bash
llm models list --query venice --tools
llm -m venice/qwen-3-8-flash --tool llm_version "What version of LLM is this?" --tools-debug --no-stream
llm -m venice/qwen-3-8-flash --functions '
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y
' "What is 1337 times 42?" --tools-debug --no-stream
```

#### Server-side web & X search tools

Supporting chat models expose `VeniceWebSearch` and `VeniceXSearch` as `llm` server-side tools (visible in `llm tools -m venice/<model>` and restored on `-c`):

```bash
llm -m venice/qwen-3-8-flash -T 'VeniceWebSearch(mode="on")' --no-stream 'What is $VVV?'
llm -m venice/grok-4-7 -T VeniceXSearch --no-stream "What is trending about Venice AI?"
```

#### Venice search tools for function calling

Venice's search API is also registered as function tools (`venice_augment_search`, `venice_augment_search_brave`, `venice_augment_search_google`, `venice_augment_search_exa`) that work with any tool-capable model, including models from other plugins:

```bash
llm -m venice/z-ai-glm-5-3 -T venice_augment_search --td "What is new in private AI inference?"
```

Each tool call is a separate billed search; use `--ta/--tools-approve` to confirm calls. See [CLI reference — Search & tools](https://github.com/ar-jan/llm-venice/blob/develop/docs/cli.md#search-and-tools) for provider selection and precedence details.

### Text-to-speech

Speech models are single-turn and save audio to the `llm` user directory by default. Voices and formats are model-specific (`llm models list --query tts-kokoro --options`).

```bash
# Default streaming (writes output as chunks arrive)
llm -m venice/tts-kokoro "Hello, welcome to Venice Voice." -o voice af_sky -o response_format mp3 -o speed 1.0

# Another TTS model with its own voice names
llm -m venice/tts-gemini-3-1-flash "Hello, welcome to Venice Voice." -o voice Kore -o response_format mp3

# Show progress, or wait for the full audio before writing
llm -m venice/tts-kokoro "First sentence. Second sentence." -o progress true
llm --no-stream -m venice/tts-kokoro "First sentence. Second sentence."
```

#### `llm venice tts` (pipelines & stdout)

Use `llm venice tts` to write audio to a chosen file, directory, or byte-exact to stdout (`-o -`):

```bash
llm venice tts "Hello." -o out.mp3
llm venice tts "Hello." --format mp3 -o - > out.mp3
echo "Hello." | llm venice tts --voice af_sky -o - | mpv -
llm venice tts "Hello." -o - | tee out.mp3 | mpv -
```

See [CLI reference — Text-to-speech](https://github.com/ar-jan/llm-venice/blob/develop/docs/cli.md#text-to-speech) for file safety, atomic writes, and options.

### Embeddings

Venice embedding models work with `llm`'s embedding commands (list them with `llm embed-models list`):

```bash
llm embed -m venice/text-embedding-bge-m3 -c "Hello world"
llm embed-multi notes --files docs '*.md' -m venice/text-embedding-qwen3-8b --store
llm similar notes -c "private inference"
```

See [CLI reference — Embeddings](https://github.com/ar-jan/llm-venice/blob/develop/docs/cli.md#embeddings) for batching, input limits, and shorter vectors.

### Image generation

Image models are single-turn and save images to the `llm` user directory by default. Check model-specific parameters with `llm models list --query <model> --options`.

```bash
llm -m venice/qwen-image-3 "Painting of a traditional Dutch windmill" -o style_preset "Watercolor"
llm -m venice/nano-banana-2-1 "Painting of a traditional Dutch windmill" -o aspect_ratio 16:9 -o resolution 4K
llm -m venice/nano-banana-2-1 "Current spring fashion street photography" -o enable_web_search true
llm -m venice/qwen-image-3 "Dutch windmill" -o variants 4 -o output_dir ./out -o output_filename windmill.png
```

List the style presets for `-o style_preset` with `llm venice image-styles`.

### Image upscaling

Upscale an image (`--scale` from `2` to `4`, optional `--creativity` from `0` to `0.02`, `--output-path`, `--overwrite`):

```bash
llm venice upscale /path/to/image.jpg
```

By default this saves `image_upscaled.<ext>` alongside the original without overwriting existing files.

### Image editing

Edit a local image or an image URL with a prompt, using one of the edit models (type `inpaint` in the catalog):

```bash
llm venice edit photo.jpg "change the sky to sunrise" -m qwen-image-2-1-turbo-edit
```

The result is saved as `photo_edited.<ext>`. See [CLI reference — Image editing](https://github.com/ar-jan/llm-venice/blob/develop/docs/cli.md#image-editing-llm-venice-edit) for aspect ratios, resolutions, transparency, and catalog checks.

### Background removal

Remove the background of a local image or an image URL, saving a transparent PNG (`image_nobg.png`):

```bash
llm venice background-remove /path/to/image.jpg
```

### Direct web search, scraping, and document parsing

Search the web through Venice without invoking a model (`-p brave|google|exa`, `-n 1..20`, `--json`), fetch a page as markdown, or extract the text of a document (PDF, DOCX, PPTX, XLSX, EPUB, text, and source code):

```bash
llm venice search -p exa -n 5 "private AI inference"
llm venice scrape https://venice.ai > venice.md
llm venice parse report.pdf | llm -m venice/qwen-3-8-flash "Summarize this"
```

Scraping is also available to models as the `venice_augment_scrape` function tool.

### Other `llm venice` commands

`llm venice` also includes commands for listing public characters and managing API keys and rate limits:

```bash
llm venice characters
llm venice api-keys list
llm venice api-keys rate-limits
llm venice --help
```

### Debugging HTTP requests

Set `LLM_VENICE_SHOW_RESPONSES=1` to print redacted Venice API requests and responses to stderr (covering CLI and Python API calls):

```bash
LLM_VENICE_SHOW_RESPONSES=1 llm -m venice/qwen-3-8-flash --no-stream 'Say hi'
```

See [CLI reference — Debugging HTTP requests](https://github.com/ar-jan/llm-venice/blob/develop/docs/cli.md#debugging-http-requests) for redaction, fallback to `LLM_OPENAI_SHOW_RESPONSES`, and streaming details.

## Documentation

- [CLI reference](https://github.com/ar-jan/llm-venice/blob/develop/docs/cli.md) — detailed notes on Venice options, search tools, `llm venice tts`, upscaling, characters, API keys, and HTTP request logging.
- [Python API](https://github.com/ar-jan/llm-venice/blob/develop/docs/python-api.md) — `VeniceClient`, `AsyncVeniceClient`, typed parameters/results, streaming, atomic saves, error handling, and catalog helpers.
- [Changelog & migration guide](https://github.com/ar-jan/llm-venice/blob/develop/CHANGELOG.md) — release notes and the 0.9-to-0.10 migration tables.
- [LLM documentation](https://llm.datasette.io/en/stable/usage.html) — general `llm` CLI and library usage.

## Development

Check out the repository and create a virtual environment:

```bash
cd llm-venice
uv venv
source .venv/bin/activate
uv pip install -e '.[test,dev]'
```

Optionally install pre-commit hooks:

```bash
uv pip install pre-commit
pre-commit install
```

Run tests, formatting, linting, and type checks:

```bash
pytest
ruff format --check .
ruff check .
pyright
```

The default `pytest` run excludes live API calls. Run `pytest -m integration` with a Venice key for catalog, account, chat, and short-speech checks; paid image generation and upscaling tests also require `LLM_VENICE_TEST_IMAGES=1`.
