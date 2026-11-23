# Python API

`llm-venice` provides typed synchronous and asynchronous clients (`VeniceClient` and `AsyncVeniceClient`) for Venice's catalogs, characters, image generation, upscaling, speech, web search, and API key management. Chat continues to use `llm`'s model API (`llm.get_model("venice/...")` or `VeniceChat` / `AsyncVeniceChat`).

Python 3.11+ is supported and the installed package ships inline type information (`py.typed`).

## Contents

- [Quick start](#quick-start)
- [Clients, authentication, and lifecycle](#clients-authentication-and-lifecycle)
- [Parameters and `extra_body`](#parameters-and-extra_body)
- [JSON resources and search](#json-resources-and-search)
- [Chat and search tools via `llm`](#chat-and-search-tools-via-llm)
- [Media values and saving files](#media-values-and-saving-files)
- [Async and streaming](#async-and-streaming)
- [Errors and HTTP logging](#errors-and-http-logging)
- [Catalog helpers](#catalog-helpers)
- [Migration from 0.9](#migration-from-09)

## Quick start

```python
from llm_venice import ImageParams, SpeechParams, VeniceClient

with VeniceClient() as client:  # or VeniceClient(key="...")
    models = client.list_models(type="image")
    result = client.generate_image(
        models[0]["id"],
        "A Dutch windmill at dusk",
        ImageParams(image_format="webp", variants=2),
    )
    paths = result.save("out", filename="windmill.webp")
    # [Path('out/windmill_1.webp'), Path('out/windmill_2.webp')]

    tts_models = client.list_models(type="tts")
    speech = client.generate_speech(tts_models[0]["id"], "Hello", SpeechParams())
    speech.save(f"hello.{speech.format}")

    upscaled = client.upscale_image("photo.jpg", scale=2)
    upscaled.save("photo_2x.png")
```

## Clients, authentication, and lifecycle

Both `VeniceClient` and `AsyncVeniceClient` accept:

```python
VeniceClient(
    key: str | None = None,
    *,
    base_url: str = VENICE_API_BASE,
    timeout: float | httpx2.Timeout = DEFAULT_TIMEOUT,
    http_client: httpx2.Client | None = None,  # httpx2.AsyncClient for AsyncVeniceClient
)
```

- **Lazy key resolution**: if `key` is omitted, the client resolves a key on the first request from `llm`'s key store (`venice`) or the `LLM_VENICE_KEY` environment variable.
- **Connection reuse & closing**: each client owns one HTTP connection pool; use it as a context manager (`with` / `async with`) or call `.close()` / `await .aclose()`. Closed wrappers cannot be reused.
- **Borrowed HTTP clients**: an injected `http_client=` is borrowed and never closed or mutated. The wrapper's `base_url`, authentication headers, user agent, and `timeout` still apply to every request.
- **No implicit side effects**: clients never fetch a model catalog implicitly during generation, and generation or API key creation calls are never automatically retried.

## Parameters and `extra_body`

`ImageParams` and `SpeechParams` omit unset fields so Venice applies its own API defaults:

- `ImageParams` accepts `image_format=` (dictionary validation also accepts the wire alias `format`).
- `ImageParams.width` and `height` accept 1 to 4096. Models with aspect ratios map them to the nearest aspect ratio and resolution tier; models without aspect ratios allow at most 1280 per side, which Venice checks.
- `ImageParams.anon_user_id` (and the chat model option of the same name) must be 1 to 128 printable ASCII characters without `||`. `enhance_prompt` and `disable_prompt_optimization_thinking` are plain booleans; Venice ignores the latter on models without a prompt-optimization step.
- `extra_body` carries arbitrary future JSON fields (including explicit `None`/nulls), but cannot override named parameters, positional inputs, output options, or streaming/binary controls.
- **Plugin adapters vs. direct client**: `llm`'s registered model adapters (`llm.get_model("venice/...")`) validate against their cached catalog snapshot and keep plugin defaults (PNG, `safe_mode=False`, `hide_watermark=True`; MP3 speech when supported). Inspect a registered model's `.Options` for its published limits and choices; the base class `.Options` holds the endpoint schema. Unknown catalog metadata adds no limits.

## JSON resources and search

JSON methods unwrap Venice's `data` envelope and return ordinary typed dictionaries (`TypedDict` annotations live in `llm_venice.types`). Unknown JSON fields and wire-format timestamps are preserved.

| Method | Returns | Notes |
|---|---|---|
| `list_models(type="all")` | `list[ModelInfo]` | Filters by model type (e.g. `"text"`, `"image"`, `"tts"`). Skips malformed entries. |
| `list_characters(web_enabled=None, adult=None, limit=None, offset=None)` | `list[Character]` | Returns one page (Venice default 50, max 100). Stop when a page is shorter than `limit`. (`llm venice characters` fetches all pages.) |
| `list_api_keys()` | `list[ApiKey]` | Requires an `ADMIN` key. |
| `create_api_key(key_type=..., description="", expires_at=None, limit_usd=None, limit_diem=None, limit_vcu=None, limit_period=None, model_privacy=None, ip_allowlist=None)` | `CreatedApiKey` | Requires an `ADMIN` key. Unset options are not sent. `expires_at`, if given, must be timezone-aware (sent in UTC). `limit_period` is `"EPOCH"`, `"MONTH"` or `"LIFETIME"`; `model_privacy` is `"ALL"`, `"PRIVATE_TEXT"` or `"PRIVATE_ONLY"`; `ip_allowlist` takes up to 100 IP addresses or CIDR subnets. `limit_vcu` is deprecated by Venice in favour of `limit_diem`. |
| `delete_api_key(key_id)` | `None` | Requires an `ADMIN` key. |
| `get_rate_limits()` | `RateLimits` | Current access status, balances, and rate limits. |
| `get_rate_limits_log()` | `list[RateLimitLogEntry]` | Recent rate-limit events. |
| `augment_search(query, limit=None, provider=None)` | `AugmentSearchResponse` | Returns `{"query": ..., "results": [...]}`. |

### Web search via `augment_search()`

```python
with VeniceClient() as client:
    response = client.augment_search("private AI inference", provider="exa", limit=5)
    for item in response["results"]:
        print(item["title"], item["url"])
```

- `provider` is `"brave"`, `"google"`, or `"exa"`. Omitted options use Venice's defaults (Brave, up to 10 results).
- Each result has `title` and `url`, plus optional `content` (snippet) and `date` which may be missing, `None`, or empty.

## Chat and search tools via `llm`

Use `llm.get_model()` (or `llm.get_async_model()`) for chat, structured outputs, and tool calling:

```python
import llm
from llm_venice import VeniceWebSearch, VeniceXSearch, venice_augment_search_exa

# Chat with Venice options
model = llm.get_model("venice/grok-4-7")
response = model.prompt("What is trending about Venice AI?", enable_x_search=True)
print(response.text())

# Server-side tools
chat = llm.get_model("venice/qwen-3-8-flash")
response = chat.prompt("What is $VVV?", tools=[VeniceWebSearch(mode="on")])

# Function-calling search tools (work with any tool-capable model in llm)
chain = chat.chain("Find recent papers on private inference", tools=[venice_augment_search_exa])
print(chain.text())
```

- The four function search tools (`venice_augment_search`, `venice_augment_search_brave`, `venice_augment_search_google`, `venice_augment_search_exa`) are plain synchronous functions exported from `llm_venice`.
- Calling `venice_augment_search` directly requires an explicit `provider`:

  ```python
  from llm_venice import venice_augment_search

  results = venice_augment_search("private AI inference", provider="exa")
  ```

- See [CLI reference — Venice options](cli.md#venice-options) for how `enable_x_search`, `VeniceXSearch`, and `VeniceWebSearch` interact.
- `response.response_json` keeps Venice's response fields (such as `venice_parameters.web_search_citations`, `cost` and `reasoning_content`) for streamed responses too; see [CLI reference — Response metadata](cli.md#response-metadata).

## Media values and saving files

Buffered media results expose raw bytes on `.data` and report the actual format returned by Venice on `.format` — saving never transcodes bytes to match a filename extension.

- `ImageResult.images` is a `tuple[GeneratedImage, ...]`. `upscale_image()` returns a single `GeneratedImage`.
- `upscale_image()` accepts a file path, `bytes`, or an open binary file object; caller-supplied file objects are read from their current position and remain open.
- **Moderation flags**: `ImageResult.content_violation`, `content_policy_violation_count`, and `blurred` are independent facts. The library preserves every returned image, including moderated ones. (Plugin adapters suppress whole-response content violations, while still saving partial outputs when only some variants were omitted.)
- **Response headers**: `ImageResult.enhanced_prompt` is the rewritten prompt, decoded from `x-venice-enhanced-prompt`, when `enhance_prompt` produced one. `ImageResult.deprecation` is a `llm_venice.results.ModelDeprecation` (`deprecated`, `warning`, `date`, `replacement`) when Venice sent model deprecation headers, otherwise `None`. Clients only report these; the image model adapters print the deprecation warning. Unexpected values in these headers never fail a generation.

### Save semantics

- **Exact file path**: `GeneratedImage.save(path)`, `SpeechResult.save(path)`, `SpeechStream.save(path)`, and `AsyncSpeechStream.save(path)` write to the exact `path` given.
- **Explicit directory for batches**: `ImageResult.save(directory, filename="image")` writes into `directory`; `filename` must be a bare basename (no path separators). A missing extension is inferred from each image's `.format`, and batches with multiple images insert `_1`, `_2`, … before the suffix.
- **No silent renames**: existing destinations raise `FileExistsError` unless `overwrite=True` (library saves never invent timestamped names, unlike CLI adapters). Missing parent directories are created automatically.
- **Atomic per-file publication**:
  - Each file is staged privately before replacing or creating the destination, avoiding truncation of existing files or following planted staging symlinks. Overwrites preserve existing regular-file Unix permissions where supported.
  - Batch saves are atomic per file, not across the entire batch: earlier published files remain if a later save fails.
  - On error or cancellation, non-empty staging files are retained and the exception may carry `error.partial_path` (`Path`); empty staging files are removed. If cancellation arrives after publication has already succeeded, the exception carries `error.published_path` instead.
  - Library saves never print to stdout or stderr.
  - To inspect `partial_path` / `published_path` on async cancellation, catch `asyncio.CancelledError` where `await stream.save(...)` is awaited inside the task.

## Async and streaming

```python
import asyncio
from llm_venice import AsyncVeniceClient, SpeechParams


async def main() -> None:
    async with AsyncVeniceClient() as client:
        models = await client.list_models(type="tts")
        async with client.stream_speech(
            models[0]["id"], "Hello from Venice", SpeechParams()
        ) as stream:
            await stream.save(f"hello.{stream.format}")
            # Or iterate chunks directly:
            # async for chunk in stream.aiter_bytes():
            #     ...


asyncio.run(main())
```

The synchronous counterpart is `with client.stream_speech(...) as stream:`, followed by `stream.iter_bytes()` or `stream.save(path)`.

- **Context-bound and single-use**: choose either iteration or `.save()`, and consume the stream before leaving both the stream context and the client lifetime. Re-consuming a stream or saving after partial iteration raises `ValueError`.
- **Clean shutdown**: HTTP responses close on exhaustion, context exit, error, or cancellation. In `AsyncSpeechStream.save()`, network I/O is native async and file cleanup waits for in-flight writes before returning.

## Errors and HTTP logging

### Exception hierarchy

Catch `VeniceError` for all Venice service failures, or handle specific subclasses:

- `MissingKeyError` — no API key supplied or configured (also inherits `llm.NeedsKeyException`).
- `VeniceAPIError` — unsuccessful HTTP status; exposes `status_code`, `reason`, `error_code`, `detail`, and `headers`. (No longer inherits `RuntimeError`.)
  - `ContentViolationError` — HTTP 422 content-policy refusal from image generation or upscaling; exposes `suggested_prompt` and `recommended_model` (`str | None`) and `retryable` (`bool | None`, whether the same request may succeed on retry). Every upscale 422 raises it; a generation 422 raises it when Venice sends these policy fields. Other 422s, such as an oversized style reference image, remain `VeniceAPIError`. Its message is `VeniceAPIError`'s line followed by one line per hint (`Suggested prompt: ...`, `Recommended model: chroma`, and whether retrying may succeed); `describe(model_prefix="venice/")` returns the same text with the model named as `llm` knows it, which the `llm` model adapters use. (Unrelated to `ImageResult.content_violation`, which flags an image Venice did return.)
  - `AuthenticationError` — HTTP 401 or 403.
  - `RateLimitError` — HTTP 429; exposes `retry_after` (`float | None`, parsed from numeric or HTTP-date `Retry-After` headers).
- `VeniceConnectionError` — transport failure (with the underlying HTTP exception as `__cause__`).
- `VeniceResponseError` — HTTP 2xx response whose body was missing or malformed.

Invalid inputs raise Pydantic's `ValidationError`; local filesystem errors remain `OSError`. Original causes and task cancellation are preserved.

Venice exceptions can be pickled and copied with their message and attributes, so they reach the caller from `concurrent.futures.ProcessPoolExecutor` and `multiprocessing` workers. As with any exception, `__cause__` and the traceback are not pickled.

### HTTP request logging

Clients created without a custom `http_client=` respect `LLM_VENICE_SHOW_RESPONSES` (and fall back to `LLM_OPENAI_SHOW_RESPONSES`), checking the environment on every request. Logging never consumes streams. See [CLI reference — Debugging HTTP requests](cli.md#debugging-http-requests) for redaction and formatting details.

If you inject your own `http_client=`, request logging is under your control.

## Catalog helpers

```python
from llm_venice import VeniceClient, load_catalog, refresh_catalog
from llm_venice.catalog import catalog_path, save_catalog

cached = load_catalog()  # disk only; returns None if missing or unusable
with VeniceClient() as client:
    models = refresh_catalog(client)  # explicit fetch + atomic save
save_catalog(models, catalog_path())
```

- `llm` plugin registration uses any usable cached catalog regardless of age, even when no API key is configured.
- If the cache is missing, registration can bootstrap it with a 5-second timeout; failed bootstrap attempts are remembered for the same path/key until an explicit `refresh_catalog()` succeeds.
- Offline models are skipped during registration; deprecation notices are emitted when a deprecated model is used, and image models also warn when Venice's response headers report a deprecation.
- There is no automatic TTL refresh, trait routing, or trait alias registration.

## Migration from 0.9

See the [0.10.0 migration guide in CHANGELOG.md](../CHANGELOG.md#python-migration) for the full table of removed 0.9 helpers, result classes, and CLI options.
