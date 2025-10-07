"""Constants for the LLM Venice plugin."""

VENICE_API_BASE = "https://api.venice.ai/api/v1"

# Image generation defaults
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_IMAGE_HIDE_WATERMARK = True
DEFAULT_IMAGE_SAFE_MODE = False

# API endpoints
ENDPOINT_MODELS = f"{VENICE_API_BASE}/models"
ENDPOINT_IMAGE_GENERATE = f"{VENICE_API_BASE}/image/generate"
ENDPOINT_IMAGE_UPSCALE = f"{VENICE_API_BASE}/image/upscale"
ENDPOINT_API_KEYS = f"{VENICE_API_BASE}/api_keys"
ENDPOINT_API_KEYS_RATE_LIMITS = f"{VENICE_API_BASE}/api_keys/rate_limits"
ENDPOINT_API_KEYS_RATE_LIMITS_LOG = f"{VENICE_API_BASE}/api_keys/rate_limits/log"
ENDPOINT_CHARACTERS = f"{VENICE_API_BASE}/characters"

# Venice-specific option names (for CLI filtering)
VENICE_OPTION_NAMES = {
    "no_venice_system_prompt",
    "web_search",
    "character",
    "strip_thinking_response",
    "disable_thinking",
}
