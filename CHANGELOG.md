# Changelog

## 0.10.0 (unreleased)

- Requirements raised to `llm>=0.37` and Python 3.11+. llm 0.37 dropped Python 3.10,
  and the suite already needed llm 0.36 for `Prompt.options` to default to the model's
  options object.
- Chat models accept `-o enable_x_search true/false` and the Python `enable_x_search`
  option for xAI native web and X/Twitter search. True requires `supportsXSearch`;
  false is forwarded even on unsupported models, while unset/`None` is omitted.
  `VeniceXSearch` remains available for discovery and conversation continuation and
  overrides an explicit false option. After tool settings override options, effective
  X search replaces all Venice web-search settings, including `VeniceWebSearch`.
  False/unset retains existing web-search validation and behavior. Reserved
  `extra_body.venice_parameters.enable_x_search` overrides remain forbidden; use the
  dedicated option instead.
- Media adapters and Venice commands use reusable synchronous/asynchronous clients.
- Image and speech models are single-turn. `llm -c`, `llm --cid`, `llm chat` and
  Python conversation replies now fail with "does not support conversations"
  (`llm.ConversationNotSupported`) before any request. Previously they silently sent
  only the latest prompt and logged it as part of the conversation. Start a new
  prompt without the continuation options instead; for scripted speech, use
  `llm venice tts`. `llm models list --json` reports `supports_conversation: false`.
- Image and speech logs include `output_paths`; image logs preserve moderation and
  omitted-variant metadata. Successful partial image responses are saved.
- Image output always uses JSON/base64. The `return_binary` option has been removed;
  omit it and use `variants` for multi-image output.
- Saving creates requested parent directories without an interactive confirmation.
- Parameter validation follows endpoint bounds. Upscale `scale` and
  `llm venice upscale --scale` accept any value from 2 to 4.
- Upscaling no longer offers `--enhance`, `--enhance-creativity`, `--enhance-prompt`
  and `--replication` (or the matching `upscale_image()` arguments). The endpoint
  dropped these fields and silently ignores them.
- Upscaling accepts `--creativity` (and `upscale_image(creativity=...)`), from 0 to
  0.02, for how much detail and texture the upscaler adds. Out-of-range values fail
  locally because the endpoint would clamp them silently; unset uses the API default.
- JSON resources are checked for their required fields; optional fields pass through
  unchanged. Catalog refreshes skip malformed model entries and fail only when no
  usable model remains.
- Speech filenames follow the returned media format, including model-specific defaults.
- Model Options now expose published limits and choices. Explicit unsupported
  values fail locally; missing metadata retains endpoint validation. Prompt/text
  limits also apply. Speech accepts cloned and custom voice IDs.
- Chat models accept an `extra_body` option for API fields without a dedicated option,
  including new `venice_parameters` entries, which are merged with option-derived ones.
  Fields that would override an option or request field are rejected rather than
  silently overridden. `-o extra_body '{...}'` JSON strings now also work for image
  and speech models.
- `LLM_VENICE_SHOW_RESPONSES` now logs every Venice request: chat, image, speech
  (including streaming), upscale, characters, API keys and catalog refreshes.
  Previously it covered only image generation and speech, and chat responded only
  to `LLM_OPENAI_SHOW_RESPONSES`.
  - When the Venice variable is unset or blank, `LLM_OPENAI_SHOW_RESPONSES` is used.
    A Venice value of `0` now also silences Venice chat.
  - `0`, `false`, `off` and `no` now disable logging. In 0.9, any non-empty value,
    including `0`, enabled it.
  - Logs show request and response headers and bodies, with credential headers and
    `apiKey`/`api_key` fields redacted, base64 data shortened and binary bodies
    summarised. Enabling logging keeps environment proxy settings. This fixes the 0.9 crash
    when logging binary speech responses.
  - Async chat no longer fails with `TypeError: Invalid http_client argument` when
    `LLM_OPENAI_SHOW_RESPONSES` is set.
- New `llm venice search` command and `augment_search()` client methods search the web
  through Venice's `/augment/search` endpoint without a model, with Brave, Google or Exa
  as the provider and 1-20 results.
- New function tools let models that support function calling search through Venice:
  `venice_augment_search`, where the model must explicitly choose the provider on
  each call using guidance in the tool description, and
  `venice_augment_search_brave`, `venice_augment_search_google` and
  `venice_augment_search_exa`, which each use one provider. They also work with models
  from other plugins.

### Python migration

The legacy names, result classes, `llm_venice.api` package and public `notices`
module are removed outright. There are no compatibility shims or deprecation warnings.
`VeniceWebSearch` and `VeniceXSearch` remain public. Plugin hooks remain module
attributes but are excluded from `__all__`.

| Old (0.9) | New |
|---|---|
| `fetch_models(key)` | `with VeniceClient(key=key) as client: models = client.list_models()` |
| `persist_models(models, path)` | `from llm_venice.catalog import save_catalog`; `save_catalog(models, path)` / `refresh_catalog()` |
| `list_characters(key, web_enabled="true")` → envelope dict | `client.list_characters(web_enabled=True)` → `list[Character]` |
| `persist_characters(data)` | removed from public API (CLI-internal) |
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
| Automatic paths in library generation/save helpers | Exact-path single-result saves; directory-only batch image saves; CLI naming policy unchanged |
| `VeniceChatOptions` / `VeniceImageOptions` / `VeniceSpeechOptions` | Base class `.Options`, or `llm.get_model(...).Options` for model-specific validation |
| `VeniceNotice` | internal (adapters render notices from result facts) |
| `VeniceAPIError` / catching `RuntimeError` | `VeniceAPIError` under `VeniceError`, no longer a `RuntimeError` |
| Raw transport/response-decoding failures | `VeniceConnectionError` / `VeniceResponseError`; local file failures remain `OSError` |
| `-o return_binary true` | removed; the client always requests JSON/base64 images |
| Interactive output-directory confirmation | parent directories created automatically on an explicit save |


### CLI migration included in this release

The Venice-specific `llm prompt`/`llm chat` flag overrides are removed. Use the
`-o` equivalents in the README or the public `VeniceWebSearch`/`VeniceXSearch` tools.
Those replacements ship together in 0.10, so no released version loses first-class
search support between changes.

Binary speech stdout now belongs to `llm venice tts`:

| Old | New |
|---|---|
| `llm -m venice/tts-kokoro "Hello" -o stdout true > out.mp3` | `llm venice tts "Hello" -o - > out.mp3` |
| Save a copy while using `-o stdout true` | `llm venice tts "Hello" -o - \| tee out.mp3`, or `-o out.mp3` |

The speech `stdout` option is removed. llm owns stdout on its model paths and adds
text after generation; the dedicated command emits exact audio bytes. Diagnostic
and progress output goes to stderr. Existing CLI JSON envelopes remain intact.

L9 reasoning/stream metadata, embeddings, additional media endpoints, automatic
catalog TTL refresh and trait handling remain deferred.

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
