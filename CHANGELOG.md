# Changelog

## 0.10.0 (unreleased)

- Media adapters and Venice commands use reusable synchronous/asynchronous clients.
- Image and speech logs include `output_paths`; image logs preserve moderation and
  omitted-variant metadata. Successful partial image responses are saved.
- Image output always uses JSON/base64. The `return_binary` option has been removed;
  omit it and use `variants` for multi-image output.
- Saving creates requested parent directories without an interactive confirmation.
- Parameter validation follows endpoint bounds; upscale scale must be 2 or 4, and
  `llm venice upscale --scale` accepts only `2` or `4`.
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
