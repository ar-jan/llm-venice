"""API key management functions for Venice API."""

import json
from typing import Dict, Optional

import httpx

from llm_venice.constants import (
    ENDPOINT_API_KEYS,
    ENDPOINT_API_KEYS_RATE_LIMITS,
    ENDPOINT_API_KEYS_RATE_LIMITS_LOG,
)


def list_api_keys(headers: Dict[str, str]) -> dict:
    """
    List all API keys.

    Args:
        headers: Authentication headers

    Returns:
        JSON response with API keys
    """
    response = httpx.get(ENDPOINT_API_KEYS, headers=headers)
    response.raise_for_status()
    return response.json()


def get_rate_limits(headers: Dict[str, str]) -> dict:
    """
    Get current rate limits for the API key.

    Args:
        headers: Authentication headers

    Returns:
        JSON response with rate limits
    """
    response = httpx.get(ENDPOINT_API_KEYS_RATE_LIMITS, headers=headers)
    response.raise_for_status()
    return response.json()


def get_rate_limits_log(headers: Dict[str, str]) -> dict:
    """
    Get the last 50 rate limit logs for the account.

    Args:
        headers: Authentication headers

    Returns:
        JSON response with rate limit logs
    """
    response = httpx.get(ENDPOINT_API_KEYS_RATE_LIMITS_LOG, headers=headers)
    response.raise_for_status()
    return response.json()


def create_api_key(
    headers: Dict[str, str],
    description: str,
    key_type: str,
    expiration_date: Optional[str] = None,
    limits_vcu: Optional[float] = None,
    limits_usd: Optional[float] = None,
) -> dict:
    """
    Create a new API key.

    Args:
        headers: Authentication headers
        description: Description for the new API key
        key_type: Type of API key (ADMIN or INFERENCE)
        expiration_date: ISO format expiration date
        limits_vcu: VCU consumption limit per epoch
        limits_usd: USD consumption limit per epoch

    Returns:
        JSON response with created API key
    """
    payload = {
        "description": description,
        "apiKeyType": key_type,
        "expiresAt": expiration_date or "",
        "consumptionLimit": {
            "vcu": limits_vcu,
            "usd": limits_usd,
        },
    }
    response = httpx.post(ENDPOINT_API_KEYS, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def delete_api_key(headers: Dict[str, str], api_key_id: str) -> dict:
    """
    Delete an API key by ID.

    Args:
        headers: Authentication headers
        api_key_id: ID of the API key to delete

    Returns:
        JSON response confirming deletion
    """
    params = {"id": api_key_id}
    response = httpx.delete(ENDPOINT_API_KEYS, headers=headers, params=params)
    response.raise_for_status()
    return response.json()
