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

API fields that have no dedicated option yet can be sent with `extra_body`, as a JSON object.
Put new Venice parameters under `venice_parameters`; they are merged with the options above:

`llm -m venice/llama-3.3-70b -o extra_body '{"venice_parameters": {"new_flag": true}}' "Hello"`

`extra_body` cannot override fields that have an option or that llm sets itself (such as
`min_p`, `temperature`, `messages`, or `venice_parameters.character_slug`); use the option instead.
Image and speech models accept `-o extra_body '{...}'` in the same way.

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
- Use a trailing separator for a directory that does not exist yet: `-o new-clips/` creates it and saves audio under a generated file name.
- Existing files are not overwritten unless `--overwrite` is given; a timestamped file name is used instead.
- Audio is written to an exclusively created `<name>.part`. If that staging name is taken, a unique `.<name>.<random>.part` is used; existing partial files and symlinks are left alone. The destination appears only when the download is complete. Without `--overwrite`, publication atomically refuses to replace an existing path and retries with a timestamped name on collision. A failed download keeps any partial audio with a warning naming its path, leaving completed output from other writers untouched. With `--overwrite`, an existing file is replaced only after a successful download. `llm -m venice/<tts-model>` behaves the same.
- Staging and new audio files request private permissions (`0600`, subject to umask); overwrites preserve existing regular-file Unix access permissions where supported. FAT32/exFAT do not store per-file Unix permissions, so access protection depends on the platform and mount settings.
- Saving without overwrite prefers the platform's rename-without-replacement operation, with hard links as a fallback when that operation is unavailable or unsupported. Hard links are not required when native exclusive rename works (including FAT32/exFAT with compatible OS drivers). If neither safe publication method works, the completed audio remains in the `.part` file named in the warning.
- A request the API rejects (for example an unknown model or voice) fails before any audio arrives and writes nothing at all, not even an empty `.part` file.
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

You can request up to four image variants with `-o variants 4`. Multiple returned images are saved as suffixed filenames such as `image_1.png`, `image_2.png`.

You can check the available parameters for a model by filtering the model list with `--query`, and show the `--options`:

`llm models list --query qwen-image --options`

### Image upscaling

You can upscale existing images.
The following example saves the returned image as `image_upscaled.<ext>` in the same directory as the original file, where `<ext>` matches the format the API returns (`png`, `jpg` or `webp`):

`llm venice upscale /path/to/image.jpg`.

By default existing upscaled images are not overwritten; timestamped filenames are used instead.

See `llm venice upscale --help` for the `--scale` (`2` or `4`, default `2`), `--enhance` and related options, and `--output-path` and `--overwrite` options.
A file name passed to `--output-path` is used as given, even if its extension differs from the returned format.

### Venice commands

List the available Venice commands with:

`llm venice --help`

### Debugging HTTP requests

Set `LLM_VENICE_SHOW_RESPONSES=1` to print each Venice API request and response
(headers and bodies) to stderr. This covers chat, image, speech, upscale and the
`llm venice` commands, from both the CLI and the Python API:

```bash
LLM_VENICE_SHOW_RESPONSES=1 llm -m venice/llama-3.3-70b --no-stream 'Say hi'
```

- If `LLM_VENICE_SHOW_RESPONSES` is unset or blank, llm's `LLM_OPENAI_SHOW_RESPONSES`
  is used instead. Set the Venice variable to `0` to log other OpenAI-compatible
  models but not Venice.
- `0`, `false`, `off` and `no` turn logging off; any other value turns it on. (llm
  itself treats any non-empty `LLM_OPENAI_SHOW_RESPONSES`, even `0`, as on.)
- Authorization headers, cookies and returned API keys are redacted. Long base64
  data is shortened and binary bodies are summarised by type and size.
- Streamed chat responses are printed chunk by chunk as they arrive; use
  `--no-stream` for a single JSON body. Streamed speech audio is not printed.

---

Read the `llm` [docs](https://llm.datasette.io/en/stable/usage.html) for more usage options.

## Python API (0.10)

`VeniceClient` and `AsyncVeniceClient` cover catalogs, characters, image generation,
upscaling, speech, and API-key management. Chat continues to use llm's model API.
Python 3.10 and later are supported; the installed package includes typing information.

```python
from llm_venice import VeniceClient, ImageParams, SpeechParams

with VeniceClient() as client:  # or VeniceClient(key="...")
    models = client.list_models(type="image")
    result = client.generate_image(
        models[0]["id"],
        "A Dutch windmill at dusk",
        ImageParams(image_format="webp", variants=2),
    )
    paths = result.save("out", filename="windmill.webp")
    # out/windmill_1.webp, out/windmill_2.webp

    voices = client.list_models(type="tts")
    speech = client.generate_speech(voices[0]["id"], "Hello", SpeechParams())
    speech.save(f"hello.{speech.format}")
    client.upscale_image("photo.jpg", scale=2).save("photo_2x.png")
```

Keys resolve lazily from `key=`, the llm keystore, or `LLM_VENICE_KEY`. Each client
reuses one HTTP connection pool; close it or use a context manager. An injected
`http_client=` is borrowed and is never closed or mutated. The wrapper's base URL,
authentication, user agent and timeout apply to every request. Closed wrappers
cannot be reused. Generation and key creation are never automatically retried.

Parameter defaults are omitted so Venice chooses its API defaults. Image params
accept `image_format=`; dictionary validation also accepts the wire alias `format`.
`extra_body` carries future JSON fields, including explicit nulls, but cannot override
named parameters, positional inputs, output options, or streaming/binary controls.
Clients never fetch a catalog implicitly during generation. Model adapters use
their existing catalog snapshot for model-specific validation and retain plugin
defaults (PNG, `safe_mode=False`, `hide_watermark=True`; MP3 speech when supported).
Use a registered model's `.Options` for its published limits and choices; class
`.Options` remains the base endpoint schema. Unknown catalog metadata adds no limits.

JSON methods return ordinary typed dictionaries and unwrap `data` envelopes:
`list_models()`, `list_characters(web_enabled=True, adult=False)`, `list_api_keys()`,
`get_rate_limits()`, and `get_rate_limits_log()`. Unknown JSON fields and wire-format
timestamps are preserved. Key management requires an ADMIN key; `create_api_key()`
takes `key_type=`, `description=`, optional `limit_usd=`/`limit_vcu=`, and an optional
timezone-aware `expires_at` datetime. `delete_api_key(id)` returns `None`.
Resource annotations are available from `llm_venice.types`.

### Media and explicit saves

Buffered image/speech values expose `.data`. `ImageResult.images` is a tuple of
`GeneratedImage`; upscale returns that same image type. Results report the returned
format and never transcode bytes to match a filename suffix.

Single images, speech values, and streams save to an **exact file path**. Image
batches save to an **explicit directory**; `filename` must be a basename. A missing
suffix is inferred, and multiple images insert `_1`, `_2`, etc. before the suffix.
Existing destinations raise `FileExistsError` unless `overwrite=True`; library saves
never invent a different name. Requested parents are created without a prompt.
CLI adapters retain their automatic names and race-safe collision retries.

Publication is atomic per file, not across a batch: earlier successful files survive
a later failure. Private staging avoids truncating existing files or following
planted staging symlinks. Overwrite preserves existing regular-file permissions.
Errors/cancellation retain nonempty staging and may attach `error.partial_path`;
empty staging is removed. Cancellation that arrives once the file is already
published attaches `error.published_path` to the complete file instead.
Library saves print nothing.
To inspect this metadata on cancellation, catch `asyncio.CancelledError` where
`stream.save()` is awaited inside the task. Python 3.10 wraps that exception when
another task awaits the cancelled task.

`content_violation`, `content_policy_violation_count`, and `blurred` are independent
facts. The library preserves all returned images, including moderated images.
Adapters suppress whole-response content violations but save successful partial
outputs when only some variants were omitted.

### Async and streaming

```python
import asyncio
from llm_venice import AsyncVeniceClient, SpeechParams


async def main():
    async with AsyncVeniceClient() as client:
        models = await client.list_models(type="tts")
        async with client.stream_speech(
            models[0]["id"], "Hello from Venice", SpeechParams()
        ) as stream:
            await stream.save(f"hello.{stream.format}")
            # Alternatively: async for chunk in stream.aiter_bytes(): ...


asyncio.run(main())
```

The synchronous equivalent is `with client.stream_speech(...) as stream`, followed
by `stream.iter_bytes()` or `stream.save(path)`. A stream is context-bound and
single-use: choose one iterator or one save, and consume it before leaving the
context and client lifetime. Repeated consumption or saving after partial iteration
raises `ValueError`. Responses close on exhaustion, context exit, failure or
cancellation. Async file cleanup waits for in-flight writes; network I/O is native async.
Caller-supplied upscale files are read from their current position and remain open.

### Errors and logging

Catch `VeniceError` for service failures, or its specific subclasses:
`AuthenticationError` (401/403), `RateLimitError` (429, with `retry_after`),
`VeniceAPIError` (HTTP status, headers, code and detail), `VeniceConnectionError`,
`VeniceResponseError` (malformed responses), and `MissingKeyError` (also an llm
`NeedsKeyException`). `VeniceAPIError` no longer inherits `RuntimeError`.
Invalid parameters raise Pydantic `ValidationError`; local file failures remain
`OSError`. Original causes and cancellation are preserved.

Clients created without `http_client` log as described in
[Debugging HTTP requests](#debugging-http-requests), reading the setting on every
request. They never consume streams. Logging for a client you inject with
`http_client` is up to you.

### Catalogs

```python
from llm_venice import load_catalog, refresh_catalog, VeniceClient
from llm_venice.catalog import catalog_path, save_catalog

cached = load_catalog()  # disk only; None if missing or unusable
with VeniceClient() as client:
    models = refresh_catalog(client)  # explicit fetch and atomic save
save_catalog(models, catalog_path())
```

Registration uses any usable cache regardless of age, even without a key. A missing
cache can bootstrap with a five-second timeout; failed attempts are remembered for
the same path/key until explicit successful refresh. Offline models are skipped;
deprecation notices appear when a model is used. There is no automatic TTL refresh,
trait routing, or trait alias registration.

See [the 0.9-to-0.10 migration guide](CHANGELOG.md) for removed helpers and options.

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

The default test run excludes live API calls. Run `pytest -m integration` manually
with a Venice key for catalog, account, chat and short-speech checks. Paid image
generation/upscale additionally requires `LLM_VENICE_TEST_IMAGES=1`.
