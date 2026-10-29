"""Constants for the LLM Venice plugin."""

VENICE_API_BASE = "https://api.venice.ai/api/v1"

# Image generation defaults
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_IMAGE_HIDE_WATERMARK = True
DEFAULT_IMAGE_SAFE_MODE = False

# API endpoints
ENDPOINT_MODELS = f"{VENICE_API_BASE}/models"
ENDPOINT_IMAGE_GENERATE = f"{VENICE_API_BASE}/image/generate"
ENDPOINT_IMAGE_UPSCALE = f"{VENICE_API_BASE}/image/upscale"
ENDPOINT_AUDIO_SPEECH = f"{VENICE_API_BASE}/audio/speech"
ENDPOINT_API_KEYS = f"{VENICE_API_BASE}/api_keys"
ENDPOINT_API_KEYS_RATE_LIMITS = f"{VENICE_API_BASE}/api_keys/rate_limits"
ENDPOINT_API_KEYS_RATE_LIMITS_LOG = f"{VENICE_API_BASE}/api_keys/rate_limits/log"
ENDPOINT_CHARACTERS = f"{VENICE_API_BASE}/characters"

# Venice-specific parameters for _ChatOptions
# Field names as expected by the API formatted as extra_body.venice_parameters
VENICE_PARAMETERS = {
    "include_venice_system_prompt",
    "enable_web_search",
    "enable_web_scraping",
    "enable_web_citations",
    "enable_x_search",
    "include_search_results_in_stream",
    "character_slug",
    "strip_thinking_response",
    "disable_thinking",
}

NON_OPENAI_COMPATIBLE_PARAMS = [
    "min_p",
    "top_k",
    "repetition_penalty",
    "stop_token_ids",
]
