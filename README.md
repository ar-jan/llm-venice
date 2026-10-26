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

`llm chat --model venice/mistral-small-3-2-24b-instruct`

#### Structured Outputs

Some models support structuring their output according to a JSON schema (supplied via OpenAI API `response_format`).

This works via llm's `--schema` options, for example:

`llm -m venice/zai-org-glm-4.6 --schema "name, age int, one_sentence_bio" "Invent an evil supervillain"`

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
llm -m venice/mistral-31-24b --tool llm_version "What version of LLM is this?" --tools-debug --no-stream
# llm_time
llm -m venice/minimax-m25 --tool llm_time "What is the time in my timezone in 24H format?" --tools-debug --no-stream
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

Vision models (currently `mistral-31-24b`) support the `--attachment` option:

> `llm -m venice/mistral-31-24b -a https://upload.wikimedia.org/wikipedia/commons/a/a9/Corvus_corone_-near_Canford_Cliffs%2C_Poole%2C_England-8.jpg "Identify"` \
> The bird in the image is a carrion crow (Corvus corone). [...]

### venice_parameters

Venice-specific parameters are configured with llm's standard `-o/--option key value` flag.
Use `llm models --options` to list the options that the selected model accepts.

Disable Venice's default system prompt:

`llm -m venice/llama-3.3-70b -o include_venice_system_prompt 0 "Repeat the above prompt"`

Use web search (on web-enabled models; image models accept `on` or `off`):

`llm -m venice/llama-3.3-70b -o enable_web_search on --no-stream 'What is $VVV?'`

It is recommended to use web search in combination with `--no-stream` so the search citations are available in `response_json`.

Let Venice scrape URLs in your latest message:

`llm -m venice/llama-3.3-70b -o enable_web_scraping 1 "Summarize https://venice.ai"`

Use a public character, for example:

`llm -m venice/google.gemma-4-26b-a4b-it -o character_slug alan-watts "What is the meaning of life?"`

### Web and X search tools

Models that advertise web search or X search expose them as llm server-side tools, so they also
show up in `llm tools -m venice/<model>` and are restored when you continue a conversation with
`-c`. Turn web search on or off per model with:

`llm -m venice/llama-3.3-70b -T 'VeniceWebSearch(mode="on")' --no-stream 'What is $VVV?'`

`VeniceWebSearch` accepts `mode` (`"auto"` or `"on"`), `citations` (on by default) and `scraping`
(off by default). Models with X search support also offer:

`llm -m venice/grok-4-3 -T VeniceXSearch --no-stream "What is trending about Venice AI?"`

`VeniceXSearch` replaces Venice web search, so any web-search parameters are dropped when it is
used. Use the `-o` parameters above for explicit web search modes and combinations that the tools
do not cover.

### Text-to-speech

Text-to-speech models (currently `tts-kokoro`) generate audio from text. Audio files are stored in the LLM user directory by default.

Basic usage:

`llm -m venice/tts-kokoro "Hello, welcome to Venice Voice." -o voice af_sky -o response_format mp3 -o speed 1.0`

Streaming (default; writes the output file immediately; useful for long outputs):

`llm -m venice/tts-kokoro "First sentence. Second sentence. Third sentence." -o progress true`

Disable streaming (wait for the full audio before writing the file):

`llm --no-stream -m venice/tts-kokoro "First sentence. Second sentence. Third sentence."`

To see all available options:

`llm models list --query tts-kokoro --options`

#### `llm venice tts`

The `llm venice tts` command writes audio to a file or, byte-exact, to stdout. Use it for pipelines: `llm -m venice/tts-...` always adds llm's own output (such as a trailing newline) to stdout, so it can only save files.

```
llm venice tts [TEXT] [-m MODEL] [--voice VOICE] [--format FORMAT] [--speed SPEED]
               [-o PATH | -o -] [--overwrite] [--progress]
```

- `TEXT` is read from stdin when it is omitted or `-`.
- `-m` takes the model ID with or without the `venice/` prefix. The default is the first text-to-speech model in the Venice model catalog.
- `--voice`, `--format` (the API's `response_format`) and `--speed` accept the same values as the `voice`, `response_format` and `speed` options of `llm -m venice/<tts-model>`.
- `-o` is this command's own output option, unrelated to llm's `-o/--option KEY VALUE`. `-o FILE` writes a file, `-o DIR` writes a generated file name into a directory, and `-o -` writes only the raw audio bytes to stdout (status and `--progress` go to stderr). Without `-o`, audio is saved under the LLM user directory in `audio/`. In file mode the saved path is printed on stdout.
- Existing files are not overwritten unless `--overwrite` is given; a timestamped file name is used instead.
- Audio is written to `<name>.part` and renamed when complete. If a download is interrupted, the partial audio is kept as `<name>.part` with a warning, and an existing file is left untouched. `llm -m venice/<tts-model>` behaves the same.
- The command does not log to llm's logs database.

Examples:

```sh
llm venice tts "Hello." --format mp3 -o - > out.mp3
llm venice tts "Hello." -o out.mp3
echo "Hello." | llm venice tts --voice af_sky -o - | mpv -
llm venice tts "Hello." -o - | tee out.mp3 | mpv -   # play and keep a copy
```

The former `-o stdout true` option of `llm -m venice/tts-...` has been removed:

| Old | New |
|---|---|
| `llm -m venice/tts-kokoro "Hello." -o stdout true -o response_format mp3 > out.mp3` | `llm venice tts "Hello." --format mp3 -o - > out.mp3` |
| `llm -m venice/tts-kokoro "Hello." -o stdout true -o output_dir . -o output_filename out.mp3` | `llm venice tts "Hello." -o - \| tee out.mp3`, or just `-o out.mp3` |

### Image generation

Generated images are stored in the LLM user directory by default. Example:

`llm -m venice/qwen-image "Painting of a traditional Dutch windmill" -o style_preset "Watercolor"`

Models that support them can also use API-native aspect-ratio and resolution presets:

`llm -m venice/nano-banana-2 "Painting of a traditional Dutch windmill" -o aspect_ratio 16:9 -o resolution 4K`

Web-enabled image models can also search the web for fresher visual context:

`llm -m venice/nano-banana-2 "Current spring fashion street photography" -o enable_web_search true`

Besides the Venice API image generation parameters, you can specify the output directory and filename, and whether or not to overwrite existing files.

When `return_binary` is `false`, you can also request up to four image variants with `-o variants 4`. Multiple returned images are saved as suffixed filenames such as `image_1.png`, `image_2.png`.

You can check the available parameters for a model by filtering the model list with `--query`, and show the `--options`:

`llm models list --query qwen-image --options`

### Image upscaling

You can upscale existing images.
The following example saves the returned image as `image_upscaled.<ext>` in the same directory as the original file, where `<ext>` matches the format the API returns (`png`, `jpg` or `webp`):

`llm venice upscale /path/to/image.jpg`.

By default existing upscaled images are not overwritten; timestamped filenames are used instead.

See `llm venice upscale --help` for the `--scale` (default `2.0`), `--enhance` and related options, and `--output-path` and `--overwrite` options.
A file name passed to `--output-path` is used as given, even if its extension differs from the returned format.

### Venice commands

List the available Venice commands with:

`llm venice --help`

---

Read the `llm` [docs](https://llm.datasette.io/en/stable/usage.html) for more usage options.

## Programmatic use

You can call the library helpers directly from Python (minimally tested):

- `fetch_models()` → list of model dicts, `persist_models(models)` writes to `venice_models.json`
- `list_characters()` → dict, `persist_characters(data)` writes to `venice_characters.json`
- API keys: `list_api_keys()`, `get_rate_limits()`, `get_rate_limits_log()`, `create_api_key()`, `delete_api_key()`
- `perform_image_upscale()` → `UpscaleResult` with bytes and a resolved output path; persist with `write_upscaled_image(result)`
- `generate_image_result()` → `ImageGenerationResult` with image byte lists/metadata/output paths and structured `notices` for image generation; persist with `save_image_result(result)`
- `generate_speech_result()` → `SpeechGenerationResult` with bytes/metadata/output path for TTS generation; persist with `save_speech_result(result)`
- `stream_speech_result()` (context manager) yields `SpeechStreamResult` with an iterator of audio chunks and a resolved output path

The helpers take the API key in one of three ways:

| Argument | Helpers |
|---|---|
| optional `key=`; falls back to the stored key or `LLM_VENICE_KEY` | `fetch_models()`, `list_characters()`, `perform_image_upscale()` |
| required `headers` dict as the first argument | `list_api_keys()`, `get_rate_limits()`, `get_rate_limits_log()`, `create_api_key()`, `delete_api_key()` |
| required keyword `api_key=` | `generate_image_result()`, `generate_speech_result()`, `stream_speech_result()` |

The `persist_*`, `save_*` and `write_upscaled_image()` helpers only write to disk and take no key.

To build `headers` or resolve the stored key yourself:

```python
from llm_venice.api.client import get_auth_headers
from llm_venice.utils import get_venice_key

headers = get_auth_headers()  # or get_auth_headers("sk-...")
api_key = get_venice_key()  # raises llm.NeedsKeyException if no key is set
```

### Async usage

Async chat models are registered alongside the sync ones; fetch them with `llm.get_async_model("venice/<id>")`:

```python
import asyncio
import llm


async def main():
    model = llm.get_async_model("venice/llama-3.3-70b")
    response = await model.prompt("Hello Venice")
    print(await response.text())


asyncio.run(main())
```

Async image generation is also available via `llm.get_async_model("venice/<image-model-id>")`, which returns an `AsyncVeniceImage` instance.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-venice
uv venv
source .venv/bin/activate
```

Install the plugin with dependencies (including test and dev):

```bash
uv pip install -e '.[test,dev]'
```

Preferably also install and enable pre-commit hooks:
```bash
uv pip install pre-commit
pre-commit install
```

To run the tests:
```bash
pytest
```
