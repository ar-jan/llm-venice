"""Temporary internal bridge while consumers migrate to the client."""

from llm_venice.errors import VeniceAPIError as VeniceAPIError
from llm_venice._http import raise_api_error as raise_api_error
