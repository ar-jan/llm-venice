# Changelog

## 0.10.0 (unreleased)

- Media adapters and Venice commands use reusable synchronous/asynchronous clients.
- Image and speech logs include `output_paths`; image logs preserve moderation and
  omitted-variant metadata. Successful partial image responses are saved.
- Image output always uses JSON/base64. The `return_binary` option has been removed;
  omit it and use `variants` for multi-image output.
- Saving creates requested parent directories without an interactive confirmation.
- Parameter validation follows endpoint bounds; upscale scale must be 2 or 4.
- Speech filenames follow the returned media format, including model-specific defaults.
