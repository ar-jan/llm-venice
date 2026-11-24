# Changelog

## Unreleased

### New features

- Chat option `responses_api` (`auto`, `native` or `compat`) sends a prompt to Venice's
  `/responses` endpoint instead of Chat Completions, for example
  `llm -m venice/openai-gpt-6-luna -o responses_api native`. Chat Completions stays the
  default, and the Responses models take the same options class, so the prompt is
  handed over unchanged; options that `/responses` lacks are refused.
- The Responses models read Venice's compat lane, which serves non-OpenAI models through
  Chat Completions: its `response.reasoning.delta` events and plain-string reasoning
  summaries become `llm` reasoning, streamed or not, and summaries are replayed on later
  turns. An incomplete response (`max_output_tokens`, `content_filter`) keeps its partial
  text, usage, `status` and `incomplete_details`, where `llm` alone recorded nothing, and
  a failed response or stream error raises `llm.ModelError`.
- `VeniceResponses` and `AsyncVeniceResponses` models for Venice's `/responses` endpoint,
  built on `llm`'s Responses support. They take the chat options plus `responses_api`
  (`auto`, `native` or `compat`, sent as `x-venice-responses-mode`), send
  `anon_user_id`, `fallbacks`, `prompt_cache_retention` and the Venice parameters, and
  record the `x-venice-responses-lane` and `x-venice-responses-compat-reason` headers in
  `response_json["venice_headers"]`. Chat-only options and E2EE models are refused.
- `llm venice billing` with `balance`, `history` and `analytics` commands. `history`
  takes the first-page filters (`--currency`, `--start`, `--end`, `--page-size`) or a
  `--cursor`, prints the cursor for the next page to stderr, and with `--all-pages`
  follows cursors to the end; `--csv` prints Venice's CSV export with one header row.
- Billing client methods, which need an ADMIN key: `get_billing_balance()` (including
  bundled and earned credits), `list_usage_history()` and `export_usage_history()` (CSV)
  for cursor-paginated usage, with first-page filters for currency (including
  `EARNED_CREDITS`), time range and page size, and `get_usage_analytics()` for totals by
  day, model and key. A cursor combined with filters is rejected before the request, as
  Venice would.
- `list_model_traits()` and `get_model_compatibility_mapping()` client methods, with
  `llm venice traits` and `llm venice compatibility-mapping`, show which model each trait
  selects and which Venice model serves another provider's model name, optionally for
  one model type (`type=`, `--type`).
- `multi_edit_image()` client methods and the `llm venice multi-edit IMAGE... -p PROMPT`
  command for `/image/multi-edit`, editing a base image with layers or masks. Local
  images are uploaded as files, or sent inline when any image is a URL. It takes the
  edit options plus `quality`, and the command checks the image count against the
  model's `maxInputImages` in the cached catalog.
- `edit_image()` client methods and the `llm venice edit IMAGE PROMPT` command for
  `/image/edit`. The image is a path, bytes or binary file, uploaded, or an `http(s)`
  URL that Venice fetches. Options cover the model (Venice's default edit model when
  omitted), aspect ratio (`auto` or any ratio such as `19.5:9`), resolution, output
  format, safe mode, transparent background, prompt enhancement, skipping
  prompt-optimization thinking and `anon_user_id`. The result, an `ImageEditResult`,
  reports the model ID and name, moderation flags, the enhanced prompt and deprecation
  headers. The command checks options against the model's cached catalog entry, keeps
  safe mode off unless `--safe-mode` is given, and does not save an image flagged as a
  content violation.
- `remove_image_background()` client methods and the `llm venice background-remove`
  command remove an image's background, returning a transparent PNG. The image is a
  path, bytes or binary file, uploaded, or an `http(s)` URL that Venice fetches, with
  an optional `anon_user_id`.
- `list_image_styles()` client methods and the `llm venice image-styles` command list
  the style presets for `style_preset` (`-o style_preset`).
- `augment_parse_text()` client methods and the `llm venice parse FILE` command for
  `/augment/text-parser`, extracting the text of PDF, DOCX, PPTX, XLSX, EPUB, text and
  source-code files. A path, bytes or binary file is uploaded under a filename whose
  extension tells Venice the format (`filename=` sets one for bytes).
  `response_format="text"` returns a string; the default JSON adds the token count
  (`--format json` on the command line).
- `augment_scrape()` client methods, the `llm venice scrape URL` command (`--json` for
  the raw response) and the `venice_augment_scrape` function tool, for fetching a web
  page as markdown through `/augment/scrape`. Pages Venice cannot fetch raise
  `VeniceAPIError` (400 for sites that block automated access, 502 when scraping
  fails), which the command shows as a one-line error.
- Venice embedding models are registered with `llm` as `venice/<model>`, so
  `llm embed`, `llm embed-multi` and `llm similar` work with them, 100 texts per request
  by default. `VeniceEmbeddingModel(..., dimensions=N)` returns shorter vectors on models
  whose catalog entry has `supportsCustomDimensions`, under its own model ID
  (`venice/<model>-<N>`) so collections never mix vector sizes.
- `create_embeddings()` client methods for `/embeddings`, embedding a string or a list of
  up to 2048 strings with optional `dimensions` and `encoding_format` (`float` or
  `base64`). The result holds one embedding per input, in input order. Token-ID arrays,
  which Venice does not accept, are rejected before the request.
- `get_character()` and `list_character_reviews()` client methods, with
  `llm venice characters get SLUG` and `llm venice characters reviews SLUG` commands,
  for one character's details and a page of its public reviews with paging totals and
  the rating summary. `llm venice characters` with its options still lists and saves
  every character as before.
- `list_characters()` and `llm venice characters` filter by Pro model use (`pro`,
  `--pro`), categories, tags and model IDs (repeatable `--category`, `--tag` and
  `--model-id`) and a search term (`search`, `--search`), and sort with `sort_by` /
  `--sort-by` and `sort_order` / `--sort-order`.
- `get_api_key()` and `update_api_key()` client methods, with `llm venice api-keys get`
  and `llm venice api-keys update` commands, for `GET /api_keys/{id}` and
  `PATCH /api_keys`. An update changes only the settings given: description, expiry,
  limits, limit period, model privacy and IP allowlist. `None` (`--no-expiration`,
  `--limits-usd none`) removes the expiry or a limit, and an empty allowlist
  (`--clear-ip-allowlist`) allows any IP address.
- `create_api_key()` and `llm venice api-keys create` set Diem limits
  (`limit_diem`, `--limits-diem`), the limit period (`limit_period`, `--limit-period`:
  `EPOCH`, `MONTH` or `LIFETIME`), model privacy (`model_privacy`, `--model-privacy`:
  `ALL`, `PRIVATE_TEXT` or `PRIVATE_ONLY`) and an IP allowlist (`ip_allowlist`,
  repeatable `--ip-allowlist`, up to 100 addresses or CIDR subnets). VCU limits are
  still accepted, though Venice deprecates them in favour of Diem.
- Chat options `reasoning`, a JSON object with `effort`, `summary` and `enabled`;
  `verbosity` (`low`, `medium`, `high` or `auto`); and `fallbacks`, up to 10 model IDs
  to try when an Anthropic model refuses the request. `reasoning.effort` is checked
  against the model's published choices, as `reasoning_effort` is. As declared options,
  these fields are now rejected in `extra_body`.
- Chat `image_detail` accepts `original`, which Venice passes to OpenAI models and
  treats as `high` elsewhere.
- `ContentViolationError`, a `VeniceAPIError` subclass for content-policy refusals
  (HTTP 422) from image generation and upscaling, with Venice's `suggested_prompt`,
  `recommended_model` and `retryable` fields. Other 422s, such as an oversized style
  reference image, remain `VeniceAPIError`. Image model and `llm venice upscale`
  errors list these hints on lines after the unchanged error line, naming the
  recommended model as `venice/<id>`; exit codes are unchanged.
- Image options `enhance_prompt`, `disable_prompt_optimization_thinking` and
  `anon_user_id`, also available on `ImageParams`, and an `anon_user_id` chat option.
  `anon_user_id` is checked locally against the API's rules: 1 to 128 printable ASCII
  characters, without `||`. These fields are now rejected in `extra_body`; use the
  options instead.
- When `enhance_prompt` rewrites the prompt, the rewritten prompt is decoded from the
  `x-venice-enhanced-prompt` header into `ImageResult.enhanced_prompt` and the image
  model's `response_json` (shown by `llm logs --json`).
- Image generation reads Venice's model deprecation headers into
  `ImageResult.deprecation`. Image models warn on stderr, once per model, when Venice
  reports that the model is deprecated or scheduled for deprecation, with the
  replacement model when one is published. A model the catalog already marks as
  deprecated still gets a single warning.

### Other changes and fixes

- Creating an API key no longer sends an empty `expiresAt` and null limits when they
  are not set; only the given settings are sent.
- `prompt_cache_retention` is checked locally against `default`, `extended` and `24h`.
- Constructing `VeniceChat` or `AsyncVeniceChat` with llm's `reasoning`, `verbosity`,
  `image_detail_original` or `service_tier` flags no longer replaces the Venice chat
  options with llm's generic OpenAI ones.
- Streamed chat responses keep Venice's response fields in `response_json`, as
  non-streamed responses already did: `venice_parameters` (including web search
  citations), `cost`, `stop_reason`, `prompt_logprobs`, `reasoning_content`,
  `reasoning_details`, `thought_signature`, `extra_content` and `cache_control`.
  `llm`'s own keys and values are unchanged. `--no-stream` is no longer needed to keep
  citations.
- Venice exceptions can be pickled and copied. `VeniceResponseError`, `VeniceAPIError`
  and its subclasses failed to unpickle, so an error raised in a
  `concurrent.futures.ProcessPoolExecutor` or `multiprocessing` worker broke the pool
  with `BrokenProcessPool` instead of reaching the caller.
- Image `width` and `height` accept up to 4096 pixels, as the API now does (previously
  1280). Models that list aspect ratios use them to pick the nearest aspect ratio and a
  resolution tier: a long edge above 1536 selects 2K and above 3072 selects 4K, where
  offered. Models without aspect ratios render an exact canvas of at most 1280 per side,
  which Venice checks rather than the plugin.

## 0.10.0 (2026-11-21)

### Breaking changes

- Requires `llm>=0.37` and Python 3.11 or later.
- The 0.9 Python helper functions, result classes and `llm_venice.api` package are
  removed. See Python migration below.
- The Venice-specific `llm prompt` and `llm chat` flags, such as `--web-search` and
  `--character`, are removed in favour of `-o` options. See CLI migration below.
- The speech `-o stdout true` option is removed. Use `llm venice tts` to write audio
  to stdout.
- Image and speech models are single-turn. `llm -c`, `--cid`, `llm chat` and Python
  conversation replies fail with `llm.ConversationNotSupported` before any request is
  sent. Previously only the latest prompt was sent, but it was logged as part of the
  conversation.
- The image `return_binary` option is removed. Images are always requested as base64
  JSON; use `variants` for multiple images.
- `llm venice upscale` no longer offers `--enhance`, `--enhance-creativity`,
  `--enhance-prompt` and `--replication`, which the API dropped and silently ignored.
  `--scale` now accepts 2 to 4 (previously 1 to 4).

### New features

- Typed Python client, `VeniceClient` and `AsyncVeniceClient`, with typed parameters
  and results such as `ImageParams` and `ImageResult`. The plugin's models and
  `llm venice` commands use the same clients.
- `llm venice search` command and `augment_search()` client method, for web search
  through Venice without a model, using Brave, Google or Exa and 1 to 20 results.
- Web search tools for any model that supports tool calls, including models from other
  plugins: `venice_augment_search`, where the model picks the provider on each call,
  and `venice_augment_search_brave`, `venice_augment_search_google` and
  `venice_augment_search_exa`, which each use one provider.
- `enable_x_search` chat option (`-o enable_x_search true`) for xAI's native web and
  X/Twitter search on models with `supportsXSearch`. When enabled, it replaces the
  Venice web-search settings, including those from `VeniceWebSearch`. The
  `VeniceXSearch` tool overrides `-o enable_x_search false`.
- `extra_body` option on chat, image and speech models, a JSON object for API fields
  without a dedicated option. Chat `venice_parameters` entries are merged with those
  set by options. Fields that an option or llm already sets are rejected rather than
  overwritten.
- `llm venice upscale --creativity` (`upscale_image(creativity=...)`) sets how much
  detail and texture the upscaler adds, from 0 to 0.02. Values outside that range fail
  locally, since the API would clamp them silently.
- Chat models whose catalog entry lists `supportsAudioInput` accept WAV and MP3
  attachments, for example `-a recording.wav`.
- `list_characters()` takes `limit` (1 to 100) and `offset` for paging, and
  `llm venice characters` saves every page instead of only the first 50 characters,
  and warns when paging stops before the last page.

### Other changes and fixes

- Options are checked locally against the API's bounds and each model's published
  limits and choices, which `llm models --options` lists. Where Venice publishes no
  metadata, the API validates as before. Prompt and text length limits are checked
  too. Speech accepts cloned and custom voice IDs.
- `LLM_VENICE_SHOW_RESPONSES` logs every Venice request, including chat, upscale,
  characters, API keys and model refreshes. In 0.9 it covered only image and speech,
  and chat was logged only with `LLM_OPENAI_SHOW_RESPONSES`.
  - If it is unset or blank, `LLM_OPENAI_SHOW_RESPONSES` is used. Setting it to `0`
    also silences Venice chat.
  - `0`, `false`, `off` and `no` disable logging. In 0.9, any non-empty value,
    including `0`, enabled it.
  - Logs include request and response headers and bodies. Credential headers and
    `apiKey`/`api_key` fields are redacted, base64 data is shortened and binary bodies
    are summarised. Proxy settings from the environment still apply.
  - Fixed: logging binary speech responses crashed in 0.9, and async chat failed with
    `TypeError: Invalid http_client argument` when `LLM_OPENAI_SHOW_RESPONSES` was set.
- Image and speech logs record `output_paths`. Image logs also record moderation
  details. When the content policy omits some image variants, the remaining images are
  saved, with a warning.
- Speech file extensions follow the format the API returned, including a model's
  default format.
- Saving to a directory that doesn't exist creates it, instead of asking for
  confirmation.
- API responses are checked for required fields; other fields pass through unchanged.
  Model refreshes skip malformed entries and fail only if no usable model remains.
- `VeniceChat` and `AsyncVeniceChat` built directly in Python use the Venice API by
  default. Previously, without `api_base`, they sent prompts and the Venice key to the
  OpenAI API.
- Speech fails when a successful response is not audio: a content type other than
  `audio/*` or generic binary, such as JSON, text or an image, or an empty body.
  Previously such a response was saved as the audio file, and `--overwrite` replaced
  an existing file with it. Upscales with no image data fail the same way.
- `delete_api_key()` and `llm venice api-keys delete` fail when an HTTP 200 response
  does not confirm the deletion with `success: true`. HTTP 204 still counts as success.
- Structured output fails locally, naming the field, when the schema has a map with
  arbitrary keys, such as a Pydantic `dict[str, int]` field. Venice's strict schema
  mode would otherwise only accept an empty object for it.
- Image and speech models accept a prompt passed as `messages=[llm.user(...)]`.
- Buffered AAC speech returned as `application/octet-stream` is identified as AAC, not
  MP3. Streamed speech still takes its format from the content type or the requested
  format.

### Python migration

The 0.9 helper functions, result classes, `llm_venice.api` package and public `notices`
module are removed, with no compatibility shims or deprecation warnings.
`VeniceWebSearch` and `VeniceXSearch` remain public. Plugin hooks are still module
attributes but are no longer in `__all__`.

| Old (0.9) | New |
|---|---|
| `fetch_models(key)` | `with VeniceClient(key=key) as client: models = client.list_models()` |
| `persist_models(models, path)` | `from llm_venice.catalog import save_catalog`; `save_catalog(models, path)` / `refresh_catalog()` |
| `list_characters(key, web_enabled="true")` → envelope dict | `client.list_characters(web_enabled=True)` → `list[Character]` |
| `persist_characters(data)` | removed (now CLI-internal) |
| `list_api_keys(headers)` / `get_rate_limits(headers)` / `get_rate_limits_log(headers)` | `client.list_api_keys()` / `client.get_rate_limits()` / `client.get_rate_limits_log()` |
| `create_api_key(headers, description, key_type, "2026-…Z", vcu, usd)` | `client.create_api_key(key_type=…, description=…, expires_at=datetime, limit_vcu=…, limit_usd=…)` |
| `delete_api_key(headers, id)` | `client.delete_api_key(id)` |
| `perform_image_upscale(...)` + `write_upscaled_image(r)` | `client.upscale_image(...).save(path)` |
| `generate_image_result(prompt=, options=, model_id=, model_name=, api_key=, …)` + `save_image_result(r)` | `client.generate_image(model, prompt, ImageParams(...)).save(dir)` |
| `generate_speech_result(...)` + `save_speech_result(r)` | `client.generate_speech(model, text, SpeechParams(...)).save(path)` |
| `stream_speech_result(...)` | `client.stream_speech(model, text, params)` |
| `ImageGenerationResult` / `SpeechGenerationResult` / `SpeechStreamResult` | `ImageResult` / `SpeechResult` / `SpeechStream` |
| `UpscaleResult` | `GeneratedImage`, also used for generated image items |
| Mixed binary attributes on results | `.data` on all buffered media results |
| Automatic file paths in save helpers | `.save(path)` writes a single image or speech result to exactly `path`; `ImageResult.save(dir)` saves all images into `dir`. CLI file naming is unchanged |
| `VeniceChatOptions` / `VeniceImageOptions` / `VeniceSpeechOptions` | Base class `.Options`, or `llm.get_model(...).Options` for model-specific validation |
| `VeniceNotice` | internal |
| `VeniceAPIError` / catching `RuntimeError` | `VeniceAPIError` under `VeniceError`, no longer a `RuntimeError` |
| Raw transport/response-decoding failures | `VeniceConnectionError` / `VeniceResponseError`; local file failures remain `OSError` |
| `-o return_binary true` | removed; the client always requests JSON/base64 images |
| Interactive output-directory confirmation | parent directories created automatically on an explicit save |

### CLI migration

The Venice-specific flags that 0.9 added to `llm prompt` and `llm chat` are removed,
and llm rejects them with `No such option`. Each has an `-o` option with the same
effect:

| Old (0.9) | New |
|---|---|
| `--no-venice-system-prompt` | `-o include_venice_system_prompt 0` |
| `--web-search MODE` | `-o enable_web_search MODE` (`auto`, `on` or `off`; image models accept `on` or `off`) |
| `--web-scraping` | `-o enable_web_scraping 1` |
| `--web-citations` | `-o enable_web_citations 1` |
| `--include-search-results-in-stream` | `-o include_search_results_in_stream 1` |
| `--character SLUG` | `-o character_slug SLUG` |
| `--strip-thinking-response` | `-o strip_thinking_response 1` |
| `--disable-thinking` | `-o disable_thinking 1` |

For web and X search you can also use the `VeniceWebSearch` and `VeniceXSearch` tools
described in the README.

The speech `-o stdout true` option is removed. Use `llm venice tts` to write audio to
stdout:

| Old | New |
|---|---|
| `llm -m venice/tts-kokoro "Hello" -o stdout true > out.mp3` | `llm venice tts "Hello" -o - > out.mp3` |
| Save a copy while using `-o stdout true` | `llm venice tts "Hello" -o - \| tee out.mp3`, or `-o out.mp3` |

`llm venice tts` writes only the audio bytes to stdout, and progress and diagnostics
to stderr. With `llm -m`, llm adds its own text to stdout after generation.

## 0.9.1 (2026-08-24)

- Update to `llm>=0.33`, and various test and development changes
  ([#66](https://github.com/ar-jan/llm-venice/pull/66)):
  - Simplify AGENTS.md.
  - Use a test fixture with model id: decouples tests from live model catalog IDs,
    which can get stale.
  - pyproject.toml: Resolve `missingDefaultExcludes` warning.
  - Make `supports_web_search` a constructor parameter.
  - Redeclare `VeniceChatOptions` inherited fields for pyright compatibility.
  - pyproject.toml: `tool.ruff.lint`: use legacy rules for now with newer ruff versions.
  - Update to `llm>=0.33`, httpx2, httpx2-pytest, and transitively OpenAI 3.
  - Tests: Avoid legacy responses table by using `prompt --json`.

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.9.0...0.9.1

## 0.9.0 (2026-04-04)

- Use `llm.KeyModel` ([#56](https://github.com/ar-jan/llm-venice/pull/56))
- Redesign for programmatic use as Python library
  ([#57](https://github.com/ar-jan/llm-venice/pull/57))
- Add HTTP status error handling for API, CLI
  ([#58](https://github.com/ar-jan/llm-venice/pull/58))
- Add async classes ([#59](https://github.com/ar-jan/llm-venice/pull/59))
- Add some more checks and tests ([#60](https://github.com/ar-jan/llm-venice/pull/60))
- Add text-to-speech (TTS) ([#61](https://github.com/ar-jan/llm-venice/pull/61))
- Fix tests ([#62](https://github.com/ar-jan/llm-venice/pull/62))
- Add `aspect_ratio` and `resolution` support for image generation
  ([#63](https://github.com/ar-jan/llm-venice/pull/63))
- Add `x-venice-is-blurred` response header handling
  ([#64](https://github.com/ar-jan/llm-venice/pull/64))
- Add image `web_search` and `variants` options
  ([#65](https://github.com/ar-jan/llm-venice/pull/65))
- Add jpeg output format to image generation

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.8.2...0.9.0

## 0.8.2 (2025-12-07)

- Add support for more generation parameters
  ([#47](https://github.com/ar-jan/llm-venice/pull/47))
- Handle `venice_parameters` as Pydantic fields in `VeniceChatOptions`
  ([#48](https://github.com/ar-jan/llm-venice/pull/48))
- Add more integration tests; deduplicate options
  ([#49](https://github.com/ar-jan/llm-venice/pull/49))
- Add web citations, search results options
  ([#50](https://github.com/ar-jan/llm-venice/pull/50))
- Update project config: Python versions; add Ruff, pre-commit config, GH workflows
  ([#51](https://github.com/ar-jan/llm-venice/pull/51))
- Various minor fixes ([#52](https://github.com/ar-jan/llm-venice/pull/52))
- Add support for `enable_web_scraping` parameter
  ([#53](https://github.com/ar-jan/llm-venice/pull/53))

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.8.1...0.8.2

## 0.8.1 (2025-10-07)

- Add more unit tests, and check `output_dir` before making API call
  ([#43](https://github.com/ar-jan/llm-venice/pull/43))
- Refactor into multiple modules ([#44](https://github.com/ar-jan/llm-venice/pull/44))

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.8.0...0.8.1

## 0.8.0 (2025-09-22)

- Add and reorganize tests ([#40](https://github.com/ar-jan/llm-venice/pull/40))
- Add tool support (function calling) ([#41](https://github.com/ar-jan/llm-venice/pull/41))
- Add thinking-related `venice_parameters`
  ([#42](https://github.com/ar-jan/llm-venice/pull/42))

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.7.0...0.8.0

## 0.7.0 (2025-05-24)

- Add new image generation options for image format, `embed_exif_metadata`
  ([#32](https://github.com/ar-jan/llm-venice/pull/32))
- Use `supportsVision` model spec for image `--attachment` capability
- Add logging client option to `VeniceImage`
  ([#33](https://github.com/ar-jan/llm-venice/pull/33))
- Do not save empty black image in case of content violation response header
- Add image upscaling functionality ([#35](https://github.com/ar-jan/llm-venice/pull/35))
- Add `output_dir` option for image generation
  ([#36](https://github.com/ar-jan/llm-venice/pull/36))
- Add new upscaling image enhancement options; support 1x and fractional scaling; add
  more extensive tests ([#37](https://github.com/ar-jan/llm-venice/pull/37))

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.6.0...0.7.0

## 0.6.0 (2025-03-06)

- Use Accept-Encoding to allow gzip compression (for images)
  ([#25](https://github.com/ar-jan/llm-venice/pull/25))
- Add characters command ([#11](https://github.com/ar-jan/llm-venice/pull/11))
- Use fetched model capabilities to check web search option
  ([#29](https://github.com/ar-jan/llm-venice/pull/29))
- Add support for Structured Outputs (JSON schemas) using OpenAI `response_format`
  ([#30](https://github.com/ar-jan/llm-venice/pull/30))
- Add consumption limits options (VCU, USD) for API key creation
  ([#31](https://github.com/ar-jan/llm-venice/pull/31))
- Add `venice api-keys rate-limits-log` command (experimental API endpoint)
- Require `llm>=0.23` for structured output support and remove use of Pydantic 1

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.5.0...0.6.0

## 0.5.0 (2025-02-23)

- Add expiration date option to `api-keys create` command
  ([#15](https://github.com/ar-jan/llm-venice/pull/15))
- Add image generation ([#16](https://github.com/ar-jan/llm-venice/pull/16))
- Add web search option ([#22](https://github.com/ar-jan/llm-venice/pull/22))
- Various maintenance chores

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.4.1...0.5.0

## 0.4.1 (2025-02-18)

- Fix rate-limit command and add test; option to run API-calling tests also in GH
  Actions ([#14](https://github.com/ar-jan/llm-venice/pull/14))

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.4.0...0.4.1

## 0.4.0 (2025-02-16)

- Add `api-keys create`/`delete` commands
  ([#12](https://github.com/ar-jan/llm-venice/pull/12))
- Always use fetched available models, prepares for using model traits (stored as
  `venice_models.json`)
- Enable attachments for vision models ([#13](https://github.com/ar-jan/llm-venice/pull/13))

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.3.0...0.4.0

## 0.3.0 (2025-02-09)

- Add CLI options `--no-venice-system-prompt` and `--character character_slug`, to
  avoid having to write the `venice_parameters` JSON string manually
  ([#9](https://github.com/ar-jan/llm-venice/pull/9))
- Add provisional `api-keys` commands ([#10](https://github.com/ar-jan/llm-venice/pull/10))

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.2.0...0.3.0

## 0.2.0 (2025-02-05)

- Add `extra_body` parameter to options, allowing to disable Venice default system
  prompt, and enabling use of public characters
  ([#6](https://github.com/ar-jan/llm-venice/pull/6))
- Updated currently available models

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.1.1...0.2.0

## 0.1.1 (2025-01-06)

- Update supported models
- Enable streaming responses

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.1.0...0.1.1

## 0.1.0 (2024-11-26)

- Add `refresh` command to fetch available models via API

**Full Changelog**: https://github.com/ar-jan/llm-venice/compare/0.0.1...0.1.0

## 0.0.1 (2024-11-14)

- Initial release

**Full Changelog**: https://github.com/ar-jan/llm-venice/commits/0.0.1
