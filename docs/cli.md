# CLI reference

Detailed behaviour, flags, and edge cases for `llm-venice` on the command line. For quick examples, see the [README](../README.md).

## Contents

- [Model catalog (`llm venice refresh`)](#model-catalog-llm-venice-refresh)
- [Venice options](#venice-options)
- [Search and tools](#search-and-tools)
- [Embeddings](#embeddings)
- [Text-to-speech](#text-to-speech)
- [Image generation and editing](#image-generation-and-editing)
- [Characters and API keys](#characters-and-api-keys)
- [Debugging HTTP requests](#debugging-http-requests)

## Model catalog (`llm venice refresh`)

`llm venice refresh` fetches the current model list from the Venice API and stores it in `venice_models.json` inside the `llm` user directory.

Re-run it whenever:

- New models have been made available.
- Deprecated models have been removed.
- Model capabilities, voices, or parameter limits have changed.

### Model traits and compatibility mappings

```bash
llm venice traits                         # trait -> model ID, such as default or most_uncensored
llm venice traits --type image
llm venice compatibility-mapping          # other providers' model names -> Venice model IDs
llm venice compatibility-mapping --type embedding
```

Both print Venice's JSON response. `-t/--type` picks the model type (`text` when omitted); `traits` also accepts `all` and `code`.

## Venice options

Venice-specific parameters are passed with `llm`'s standard `-o/--option KEY VALUE` flag. Run `llm models --query <model> --options` to see every option and limit accepted by a given model.

### Web search and X search

- **Web search (`enable_web_search`)** accepts `auto`, `on`, or `off` on web-enabled chat models (image models accept `on` or `off`). Citations are kept in the response JSON under `venice_parameters.web_search_citations`, whether the response streams or not (see `llm logs --json`).
- **Web citations and scraping**: `enable_web_citations` requests superscript source citations (`^index^`), `include_search_results_in_stream` emits search results as the first stream chunk, and `enable_web_scraping` scrapes URLs found in the latest user message.
- **xAI native search (`enable_x_search`)** enables both native web **and** X/Twitter search (not X alone) on chat models advertising `supportsXSearch`.
  - Omitting `enable_x_search` (or passing `None` in Python) leaves the field out of the request; Venice's API default is `false`.
  - Passing `-o enable_x_search false` (or `enable_x_search=False` in Python) sends an explicit `false` and is allowed even on models without X search support.
  - When effective X search is `true`, it replaces Venice search augmentation: `enable_web_search`, `enable_web_citations`, `enable_web_scraping`, and `include_search_results_in_stream` are dropped, including settings from `VeniceWebSearch`.
  - When `enable_x_search` is `false` or unset, the usual Venice web-search settings and validation apply.
  - Tool settings override `-o` options: selecting `VeniceXSearch` enables X search even with `-o enable_x_search false`, including when the tool is restored by `-c` conversation continuation.

### Request options

- **Reasoning**: `-o reasoning_effort LEVEL` sets the effort, and `-o reasoning '{"effort": "high", "summary": "auto", "enabled": true}'` sets the full `reasoning` object. `summary` is `auto`, `concise`, or `detailed`; `"enabled": false` turns reasoning off on supporting models. Both efforts are checked against the model's published choices, and when both are given Venice uses `reasoning_effort`.
- **Verbosity**: `-o verbosity` accepts `low`, `medium`, `high`, or `auto`.
- **Prompt caching**: `-o prompt_cache_key KEY` groups related requests, and `-o prompt_cache_retention` accepts `default`, `extended`, or `24h`.
- **Refusal fallbacks**: `-o fallbacks '["claude-opus-4-8"]'` lists up to 10 models to try if an Anthropic model refuses the request; Venice forwards it on direct Anthropic routes only. IDs may carry the `venice/` prefix, and the API's `[{"model": "claude-opus-4-8"}]` form works too.
- **End-user attribution**: `-o anon_user_id ID` identifies your end user to upstream providers, with 1 to 128 printable ASCII characters and no `||`.
- **Image detail**: `-o image_detail` accepts `original` as well as `auto`, `low`, and `high`. Venice passes `original` to OpenAI models and treats it as `high` elsewhere.

### Response metadata

Venice adds fields to chat responses that `llm`'s generic handling of streamed chunks leaves out. Streamed responses keep them too, in the response JSON logged by `llm` (`llm logs --json`), alongside `llm`'s usual `content`, `finish_reason` and `usage` keys:

- `venice_parameters`, including `web_search_citations`, and `cost` (`usd`, `diem`).
- `stop_reason` and `prompt_logprobs`.
- `reasoning_content` (joined across chunks), `reasoning_details` (items in the order received), `thought_signature`, `extra_content` (such as `extra_content.google.thought_signature`) and `cache_control`.

These fields are recorded but not sent back on later conversation turns.

### `extra_body`

API fields that do not have a dedicated option yet can be passed as a JSON object via `-o extra_body '{...}'`:

```bash
llm -m venice/qwen-3-8-flash -o extra_body '{"venice_parameters": {"new_flag": true}}' "Hello"
```

- On chat models, entries under `venice_parameters` are merged with the Venice options set via `-o`.
- `extra_body` cannot override fields that already have a dedicated option or that `llm` sets itself (such as `min_p`, `temperature`, `messages`, `venice_parameters.character_slug`, or `venice_parameters.enable_x_search`); use the option instead.
- Image and speech models accept `-o extra_body '{...}'` in the same way.

## Search and tools

### Server-side tools (`VeniceWebSearch` and `VeniceXSearch`)

Chat models that advertise web search or X search expose them as `llm` server-side tools. They appear in `llm tools -m venice/<model>` and are automatically restored when continuing a conversation with `-c`.

- **`VeniceWebSearch`** accepts `mode` (`"auto"` or `"on"`, default `"auto"`), `citations` (`True` by default), and `scraping` (`False` by default):

  ```bash
  llm -m venice/qwen-3-8-flash -T 'VeniceWebSearch(mode="on")' --no-stream 'What is $VVV?'
  ```

- **`VeniceXSearch`** enables the same xAI native web and X/Twitter search as `-o enable_x_search true` and takes precedence over `-o enable_x_search false`:

  ```bash
  llm -m venice/grok-4-7 -T VeniceXSearch --no-stream "What is trending about Venice AI?"
  ```

- X search replaces Venice web search even when both tools are selected, regardless of order.
- `llm` requires every selected server-side tool to be supported by the chosen model.
- Use `-o` options directly for explicit combinations not covered by the tools (for example `enable_web_search off` or `include_search_results_in_stream`).

### Function tools (`venice_augment_search*`)

Venice's `/augment/search` endpoint is registered as function tools that work with any tool-capable model, including models from other `llm` plugins. Unlike the server-side tools above, the model writes each query and `llm` executes the search.

| Tool | Search provider |
|---|---|
| `venice_augment_search` | Required choice by the model on each call: `brave`, `google`, or `exa` |
| `venice_augment_search_brave` | Brave |
| `venice_augment_search_google` | Google |
| `venice_augment_search_exa` | Exa |

- Each tool takes `query` (1–400 characters) and `limit` (1–20 results, default 10), returning the `title`, `url`, `content` snippet, and `date` of each result.
- Offer only one of these tools per prompt; passing multiple gives the model duplicate tools.
- The generic `venice_augment_search` tool asks the model to choose a provider for each query. Its description suggests Brave for general keyword searches and news (ZDR), Google as an alternative keyword engine or when requested, and Exa for natural-language discovery and research (ZDR). These are starting heuristics; the model should follow your provider and privacy preferences and may switch providers if results are inadequate. Pass `--td/--tools-debug` to see the chosen provider and arguments.
- **Billing and safety**: every tool call is a separate, billed search. `--cl/--chain-limit` caps response rounds, not total searches (one round may contain multiple tool calls). Use `--ta/--tools-approve` to confirm calls before execution. Search results are untrusted web content.

Optional live checks using `llm -m venice/z-ai-glm-5-3 -T venice_augment_search --td`:

- *"Find research comparing approaches to private AI inference."* — check how the model applies the research-search heuristic.
- *"Find today's AI news using only search providers with zero data retention."* — calls should use Brave or Exa.
- *"Use Google to find the Venice API documentation."* — calls should use Google.

### Direct search (`llm venice search`)

Search the web without invoking a model:

```bash
llm venice search -p exa -n 5 "private AI inference"
```

- `-p/--provider` chooses `brave`, `google`, or `exa` (Venice defaults to Brave when omitted).
- `-n/--limit` sets the maximum number of results from `1` to `20` (Venice defaults to up to 10).
- `--json` outputs the raw JSON response.
- Each search is billed.

### Web page scraping (`llm venice scrape`)

Fetch one web page through Venice's `/augment/scrape` endpoint and print it as markdown:

```bash
llm venice scrape https://venice.ai > venice.md
```

- `URL` must be an absolute `http` or `https` URL.
- Sites that block automated access, such as X/Twitter and Reddit, are refused, and pages Venice cannot fetch fail with its error (for example `502 Bad Gateway - Could not resolve hostname`).
- Control characters other than newlines and tabs are removed from the printed markdown; `--json` outputs the raw JSON response (`url`, `content`, `format`).
- Each scrape is billed.

The same endpoint is registered as the `venice_augment_scrape` function tool, which takes a `url` and returns that JSON to the model, so a model can read a page it found with a search tool:

```bash
llm -m venice/z-ai-glm-5-3 -T venice_augment_search_brave -T venice_augment_scrape --td \
  "Find the Venice API documentation and summarize its authentication page"
```

Each tool call is a separate, billed request; use `--ta/--tools-approve` to confirm them. Scraped pages are untrusted web content, and a long page can fill much of the model's context.

### Document text extraction (`llm venice parse`)

Extract the text of a document through Venice's `/augment/text-parser` endpoint, without a model:

```bash
llm venice parse report.pdf
llm venice parse --format json slides.pptx
llm venice parse report.pdf | llm -m venice/qwen-3-8-flash "Summarize this"
```

- Venice reads PDF, DOCX, PPTX, XLSX, EPUB, plain text, Markdown, CSV, JSON and most source-code files up to 25 MB, and tells the format from the file extension. It parses the file in memory and keeps no copy.
- `--format text` (the default) prints the extracted text, with control characters other than newlines and tabs removed, so PDF page breaks (form feeds) are dropped. `--format json` prints the exact text and its token count as JSON.
- Unsupported files fail with Venice's error, which lists the supported formats.
- Each request is billed.

## Embeddings

Every online embedding model in the cached catalog is registered for `llm`'s embedding commands as `venice/<model>`; `llm embed-models list` shows them.

```bash
llm embed -m venice/text-embedding-bge-m3 -c "Hello world"
llm embed-multi notes --files docs '*.md' -m venice/text-embedding-qwen3-8b --store
llm similar notes -c "private inference"
```

- **Size**: each model returns vectors of its native size, `embeddingDimensions` in the catalog (`llm venice refresh` updates it).
- **Batching**: `llm embed-multi` sends up to 100 texts per request by default; change it with `--batch-size` (Venice accepts up to 2048).
- **Input limits**: Venice rejects a text longer than the model's `maxInputTokens`; texts are not truncated. Embedding models take text only, not attachments or binary content.
- **Shorter vectors**: models with `supportsCustomDimensions` can return fewer dimensions. `llm embed` has no option for this, so use `VeniceEmbeddingModel(..., dimensions=N)` from Python (see the [Python API](python-api.md#embeddings)).
- **Deprecations**: using a model the catalog marks as deprecated prints a warning to stderr, once per model.

## Text-to-speech

### Model invocation (`llm -m venice/tts-...`)

- Voices and output formats are model-specific (for example `tts-kokoro` uses voice IDs like `af_sky`, while `tts-gemini-3-1-flash` uses names like `Kore`). Inspect options with `llm models list --query <model> --options`.
- Speech models are single-turn: `llm -c`, `--cid`, and `llm chat` are rejected before any request is sent.
- Streaming is enabled by default and writes chunks as they arrive (`-o progress true` reports progress on stderr). Pass `--no-stream` to wait for the complete response before writing.
- Files are saved under `<llm user dir>/audio/` unless `output_dir` or `output_filename` is specified.

### Pipeline command (`llm venice tts`)

`llm venice tts` writes audio to a file, directory, or byte-exact to stdout. Use it for shell pipelines: `llm -m venice/tts-...` always appends `llm`'s own output (such as a trailing newline) to stdout, so it can only save files.

```text
llm venice tts [TEXT] [-m MODEL] [--voice VOICE] [--format FORMAT] [--speed SPEED]
               [-o PATH | -o -] [--overwrite] [--progress]
```

- **Input**: `TEXT` is read from stdin when omitted or `-`.
- **Model**: `-m/--model` accepts a model ID with or without the `venice/` prefix. When omitted, it uses the first online `tts` model in the cached catalog.
- **Speech options**: `--voice`, `--format` (maps to `response_format`), and `--speed` (`0.25`–`4.0`) accept the same values as `llm -m venice/<tts-model>`.
- **Output (`-o/--output`)**: this flag belongs to `llm venice tts` (unrelated to `llm`'s `-o/--option KEY VALUE`):
  - `-o FILE` writes to that file path.
  - `-o DIR` (or a non-existent directory with a trailing separator, e.g. `-o new-clips/`) creates the directory if needed and writes a generated filename inside it.
  - `-o -` writes only the raw audio bytes to stdout (progress and status messages go to stderr) and refuses to write binary audio directly to a TTY.
  - Without `-o`, audio is saved in `<llm user dir>/audio/`. In file mode, the saved path is printed to stdout.
- **Logging**: `llm venice tts` does not log to `llm`'s SQLite logs database.

### Safe file publication

Both `llm venice tts` and `llm -m venice/<tts-model>` publish audio safely:

- Existing files are never overwritten unless `--overwrite` (`-o overwrite_files true`) is passed; a timestamped filename is generated instead.
- Audio streams into an exclusively created `<name>.part` staging file (falling back to a unique `.<name>.<random>.part` if taken), leaving existing partial files and symlinks untouched.
- The final destination appears only once the download succeeds. Without `--overwrite`, publication atomically refuses to replace an existing path (preferring the OS rename-without-replacement operation, falling back to hard links when unavailable) and retries with a timestamped name on collision. If neither safe publication method works, completed audio stays in the `.part` file named in the warning.
- Failed downloads keep any partial audio and emit a warning with its path without affecting completed files from other writers. API rejections (such as an invalid model or voice) fail before any audio arrives and write nothing, not even an empty `.part` file.
- With `--overwrite`, an existing file is replaced only after a complete download, preserving existing regular-file Unix permissions where supported. New and staging files request `0600` permissions (subject to umask); on FAT32/exFAT, access protection depends on mount settings.

### Migration from `-o stdout true`

The `-o stdout true` option on `llm -m venice/tts-...` was removed in 0.10:

| Old (0.9) | New (0.10+) |
|---|---|
| `llm -m venice/tts-kokoro "Hello." -o stdout true -o response_format mp3 > out.mp3` | `llm venice tts "Hello." --format mp3 -o - > out.mp3` |
| `llm -m venice/tts-kokoro "Hello." -o stdout true -o output_dir . -o output_filename out.mp3` | `llm venice tts "Hello." -o - \| tee out.mp3`, or `-o out.mp3` |

## Image generation and editing

### Image generation

- Image models are single-turn (`llm -c`, `--cid`, and `llm chat` are rejected).
- Generated images are stored in `<llm user dir>/images/` by default; override with `-o output_dir`, `-o output_filename`, and `-o overwrite_files true`.
- Request up to 4 variants with `-o variants 4`. Multiple returned images are saved with numbered suffixes (`image_1.png`, `image_2.png`, …).
- Supporting models accept `-o aspect_ratio`, `-o resolution`, and `-o enable_web_search`. Inspect every model's published limits and choices with `llm models list --query <model> --options`.
- `-o width` and `-o height` accept up to 4096 pixels. On models that list aspect ratios they pick the nearest aspect ratio and a resolution tier (a long edge above 1536 selects 2K, above 3072 selects 4K, where offered); an explicit `aspect_ratio` or `resolution` overrides only its own value. Models without aspect ratios render an exact canvas of at most 1280 per side, which Venice enforces.
- `-o enhance_prompt true` has Venice rewrite the prompt with more visual detail before generating, at extra cost and up to about 30 seconds of delay. The rewritten prompt is saved as `enhanced_prompt` in the response JSON (see `llm logs --json`). `-o disable_prompt_optimization_thinking true` skips the prompt-optimization thinking step on models that have one.
- `-o anon_user_id ID` identifies your end user to upstream providers, on image and chat models. It takes 1 to 128 printable ASCII characters, without `||`.
- When Venice reports that an image model is deprecated or scheduled for deprecation, a warning is printed to stderr once per model, naming the replacement model when there is one.
- When Venice refuses a prompt under a content policy, the error keeps its usual first line and exit code, followed by Venice's suggested prompt, its recommended model (as `venice/<id>`), and whether retrying may succeed, when Venice sends them. Other 422 errors, such as an oversized style reference image, are reported as before.

### Image styles (`llm venice image-styles`)

List the style presets that image models accept with `-o style_preset`, one per line (`--json` for the raw response):

```bash
llm venice image-styles
llm -m venice/qwen-image-3 -o style_preset "Watercolor" "A Dutch windmill"
```

### Image editing (`llm venice edit`)

Edit an image as a prompt directs:

```bash
llm venice edit photo.jpg "change the sky to sunrise" -m qwen-image-2-1-turbo-edit
llm venice edit https://example.com/logo.png "remove the text" --format png --transparent-background -m mai-image-2-7-flash-edit
```

```text
llm venice edit IMAGE PROMPT... [-m MODEL] [--aspect-ratio RATIO] [--resolution TIER]
                [--format png|jpeg|webp] [--safe-mode] [--transparent-background]
                [--enhance-prompt] [--disable-prompt-optimization-thinking]
                [--anon-user-id ID] [-o/--output-path PATH] [--overwrite]
```

- `IMAGE` is a local file, uploaded to Venice, or an `http`/`https` URL, which Venice fetches. Multiple `PROMPT` words are joined with spaces.
- `-m/--model` takes an edit model ID, with or without the `venice/` prefix; Venice uses its default edit model when it is omitted. Edit models are listed as type `inpaint` in the catalog (`llm venice refresh`), and are not `llm` models.
- When the model is in the cached catalog, its prompt length limit and its aspect ratios, resolutions and transparent-background support are checked before anything is sent. Otherwise Venice checks them.
- `--aspect-ratio` is `auto` or a ratio such as `16:9`; `auto`, or no ratio, follows the input image. `--resolution` picks a tier such as `1K`, `2K` or `4K`.
- Without `--format`, Venice returns PNG at 1K and JPEG at higher resolutions, or PNG for a transparent background, which needs PNG or WebP.
- Safe Venice blurring of adult content is off unless `--safe-mode` is given, as for image models. A blurred result prints a warning.
- `--enhance-prompt` has Venice rewrite the prompt first, at extra cost, and prints the rewritten prompt to stderr.
- Saves `<name>_edited.<ext>` next to a local image, or in the current directory for a URL, where `<ext>` matches the returned format. `-o` takes a file name, or a directory for the default name; existing files are kept unless `--overwrite` is given.
- An image that Venice marks as a content violation is not saved, and the command fails. Model deprecation warnings from Venice are printed to stderr.
- Each edit is billed, by model and resolution.

### Multi-image editing (`llm venice multi-edit`)

Edit layered images with one prompt. The first image is the base; the others are layers or masks:

```bash
llm venice multi-edit room.jpg sofa.png -p "place the sofa in the room" -m qwen-image-2-1-turbo-edit
```

- Each `IMAGE` is a local file or an `http`/`https` URL. When every image is local they are uploaded as files; when any is a URL, local images are sent inline (base64) alongside it.
- `-p/--prompt` is required. `--quality` (`low`, `medium` or `high`) applies on models that offer it; the other options are those of `llm venice edit`.
- How many images a model takes is `maxInputImages` in its catalog constraints. When the model is in the cached catalog, the count, quality and the `edit` checks are applied before anything is sent.
- The result is saved as `<name>_edited.<ext>` after the first image.
- Each edit is billed; some models charge per additional input image.

### Background removal (`llm venice background-remove`)

```text
llm venice background-remove IMAGE [-o/--output-path PATH] [--overwrite] [--anon-user-id ID]
```

- `IMAGE` is a local file, uploaded to Venice, or an `http`/`https` URL, which Venice fetches.
- Saves a PNG with a transparent background as `<name>_nobg.png` next to a local image, or in the current directory for a URL. `-o` takes a file name, or a directory for the default name.
- Existing files are not overwritten unless `--overwrite` is given; a timestamped filename is used instead.
- Each request is billed.

### Image upscaling (`llm venice upscale`)

```text
llm venice upscale IMAGE_PATH [--scale FLOAT] [--creativity FLOAT]
                   [-o/--output-path PATH] [--overwrite]
```

- Saves `<stem>_upscaled.<ext>` in the source image's directory by default, where `<ext>` matches the format returned by the API (`png`, `jpg`, or `webp`).
- Existing files are not overwritten unless `--overwrite` is given; a timestamped filename is used instead.
- `--scale` accepts `2` to `4` (default `2`).
- `--creativity` accepts `0` to `0.02` (API default `0.01`); higher values add more detail and texture.
- Images refused by content moderation fail with the usual error line, followed by a suggested prompt if Venice sends one.
- Passing a filename to `-o/--output-path` uses that exact name even if its extension differs from the returned image format. Passing a directory saves the default filename inside it.

## Characters and API keys

### Public characters (`llm venice characters`)

Fetches every page of public characters and saves them to `<llm user dir>/venice_characters.json`:

```bash
llm venice characters
llm venice characters --web-enabled true --adult false
llm venice characters --tag philosophy --search "#zen" --sort-by highlyRated
```

- Filters: `--web-enabled`, `--adult`, and `--pro` (characters using a Pro model) take `true` or `false`. `--category`, `--tag`, and `--model-id` can each be repeated, up to 20 values, and match characters with any of the values given. `--search` matches names, descriptions, and tags (`#tag` searches tags).
- Sorting: `--sort-by` accepts `featured`, `highestRating`, `highlyRated`, `highlyRatedAndRecent`, `imports`, `mostRecent`, or `ratingCount`, and `--sort-order` accepts `asc` or `desc` (Venice's default).
- The saved file holds the characters matching the filters, replacing any earlier list.

Show one character, or a page of its public reviews, as JSON on stdout:

```bash
llm venice characters get alan-watts
llm venice characters reviews alan-watts --page 2 --page-size 50
```

`reviews` returns the reviews with paging totals (`pagination.total`, `pagination.totalPages`) and the rating summary; pages start at 1, with 20 reviews per page by default and at most 100. The list filters apply only to `llm venice characters` itself, so combining them with `get` or `reviews` is an error.

Use a character's `slug` in chat prompts with `-o character_slug <slug>`.

### API keys and rate limits (`llm venice api-keys`)

```bash
llm venice api-keys list            # default when no subcommand is given
llm venice api-keys rate-limits
llm venice api-keys rate-limits-log
llm venice api-keys get <API_KEY_ID>
llm venice api-keys create --type INFERENCE --description "ci" --expiration-date 2026-12-31 --limits-usd 5.0
llm venice api-keys update <API_KEY_ID> --limits-diem 5 --limit-period MONTH
llm venice api-keys delete <API_KEY_ID>
```

Listing, getting, creating, updating, and deleting API keys require a Venice `ADMIN` key. Run `llm venice api-keys --help` for full flag details.

`create` sends only the settings you give; Venice applies its defaults for the rest:

- `--limits-usd` and `--limits-diem` cap consumption per limit period. `--limits-vcu` is still accepted but deprecated by Venice in favour of Diem.
- `--limit-period` sets when the limits reset: `EPOCH` (each UTC day, Venice's default), `MONTH` (the 1st of each UTC month), or `LIFETIME` (never).
- `--model-privacy` restricts which models the key may call: `ALL`, `PRIVATE_TEXT` (text and embedding models must be Private, TEE, or E2EE), or `PRIVATE_ONLY`.
- `--ip-allowlist ADDRESS` limits use to an IP address or CIDR subnet; repeat it for up to 100 entries. Without it, the key works from any IP address.

```bash
llm venice api-keys create --type INFERENCE --limits-diem 2 --limit-period MONTH \
  --model-privacy PRIVATE_ONLY --ip-allowlist 11.22.33.0/24 --ip-allowlist 2a00:1::/48
```

`update` takes the same settings plus `--description`, and changes only the ones you give; the others keep their current values. To remove a setting, use `--no-expiration`, `none` as a limit (`--limits-usd none`), or `--clear-ip-allowlist` to let any IP address use the key. `--ip-allowlist` replaces the whole list.

```bash
llm venice api-keys update <API_KEY_ID> --no-expiration --limits-usd none --ip-allowlist 11.22.33.44
```

## Debugging HTTP requests

Set `LLM_VENICE_SHOW_RESPONSES=1` to print each Venice API request and response (headers and bodies) to stderr. This covers chat, image, speech, upscale, search tools, and all `llm venice` commands, from both the CLI and the Python API:

```bash
LLM_VENICE_SHOW_RESPONSES=1 llm -m venice/qwen-3-8-flash --no-stream 'Say hi'
```

- **Fallback**: if `LLM_VENICE_SHOW_RESPONSES` is unset or blank, `llm`'s `LLM_OPENAI_SHOW_RESPONSES` is used instead. Set `LLM_VENICE_SHOW_RESPONSES=0` to log other OpenAI-compatible models without logging Venice.
- **Boolean parsing**: `0`, `false`, `off`, and `no` turn logging off; any other non-empty value turns it on. (`llm` itself treats any non-empty `LLM_OPENAI_SHOW_RESPONSES`, including `0`, as on.)
- **Redaction**: `Authorization`, `Proxy-Authorization`, `Cookie`, `Set-Cookie`, and `X-API-Key` headers are redacted, as are `apiKey` / `api_key` fields in JSON bodies (such as a newly created API key). Other fields, such as logprobs `token`, are shown as-is.
- **Binary and large payloads**: long base64 strings are shortened and binary bodies are summarised by content type and byte size.
- **Streaming**: streamed chat responses are logged event by event as they arrive; pass `--no-stream` for a single pretty-printed JSON response body. Streamed speech audio is not printed.
- **Proxies**: logging does not alter transport configuration, so `HTTP(S)_PROXY`, `ALL_PROXY`, and `NO_PROXY` continue to apply.
