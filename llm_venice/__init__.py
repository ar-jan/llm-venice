"""Venice clients, media values, catalog helpers and llm plugin adapters."""

import llm

from .client import VeniceClient, AsyncVeniceClient
from .params import ImageParams, SpeechParams
from .results import GeneratedImage, ImageResult, SpeechResult, SpeechStream, AsyncSpeechStream
from .errors import (
    VeniceError,
    VeniceAPIError,
    AuthenticationError,
    RateLimitError,
    VeniceConnectionError,
    VeniceResponseError,
    MissingKeyError,
)
from .catalog import load_catalog, refresh_catalog
from .models.chat import VeniceChat, AsyncVeniceChat, VeniceWebSearch, VeniceXSearch
from .models.image import VeniceImage, AsyncVeniceImage
from .models.audio import VeniceSpeech, AsyncVeniceSpeech
from .tools import (
    AUGMENT_SEARCH_TOOLS as _AUGMENT_SEARCH_TOOLS,
    venice_augment_search,
    venice_augment_search_brave,
    venice_augment_search_google,
    venice_augment_search_exa,
)
from .models import register_venice_models as _register_models
from .cli import register_venice_commands as _register_commands


@llm.hookimpl
def register_commands(cli):
    """Register Venice commands with llm's plugin system."""
    _register_commands(cli)


@llm.hookimpl
def register_models(register):
    """Register cached Venice models with llm's plugin system."""
    _register_models(register)


@llm.hookimpl
def register_tools(register):
    """Register Venice function tools with llm's plugin system."""
    for tool in _AUGMENT_SEARCH_TOOLS:
        register(tool)


__all__ = [
    "VeniceClient",
    "AsyncVeniceClient",
    "ImageParams",
    "SpeechParams",
    "GeneratedImage",
    "ImageResult",
    "SpeechResult",
    "SpeechStream",
    "AsyncSpeechStream",
    "VeniceError",
    "VeniceAPIError",
    "AuthenticationError",
    "RateLimitError",
    "VeniceConnectionError",
    "VeniceResponseError",
    "MissingKeyError",
    "load_catalog",
    "refresh_catalog",
    "VeniceChat",
    "AsyncVeniceChat",
    "VeniceImage",
    "AsyncVeniceImage",
    "VeniceSpeech",
    "AsyncVeniceSpeech",
    "VeniceWebSearch",
    "VeniceXSearch",
    "venice_augment_search",
    "venice_augment_search_brave",
    "venice_augment_search_google",
    "venice_augment_search_exa",
]
