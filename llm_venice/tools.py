"""Function tools that let a model search the web through Venice's /augment/search.

``venice_augment_search`` lets the model choose the provider; the other tools
always use one provider. Each call is a separate, billed search. The Venice key
resolves on every call, from llm's key store or ``LLM_VENICE_KEY``. The tools are
synchronous, so they work in sync and async chains; async chains wait for them.
Failures raise, and llm reports them to the model as ``Error: ...`` tool results.
"""

import inspect
from typing import get_args

from pydantic import ValidationError

from .client import VeniceClient
from .types import AugmentSearchProvider, AugmentSearchResponse


def _augment_search(
    query: str, provider: AugmentSearchProvider, limit: int
) -> AugmentSearchResponse:
    # llm passes tool arguments directly to Python; annotations do not validate
    # them. In particular, None would let the client use the API's default.
    if provider not in get_args(AugmentSearchProvider):
        raise ValueError(
            "Invalid search input: provider: choose 'brave', 'google' or 'exa' explicitly"
        )
    try:
        with VeniceClient() as client:
            return client.augment_search(query, limit=limit, provider=provider)
    except ValidationError as exc:
        # A short message the model can act on, instead of pydantic's report.
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ValueError(f"Invalid search input: {details}") from exc


def venice_augment_search(
    query: str, provider: AugmentSearchProvider, limit: int = 10
) -> AugmentSearchResponse:
    """Search the web through Venice. Returns JSON with the query and results
    (title, url, content snippet, and date when available).

    Choose provider explicitly for each query, respecting the user's provider
    and privacy preferences. These are starting heuristics:
    - "brave": general keyword searches and news; Zero Data Retention (ZDR).
    - "google": alternative keyword searches or when requested; anonymized through Venice.
    - "exa": natural-language discovery, semantic and research queries; ZDR.
    If results are inadequate, consider a different provider or reformulate the
    query. Each call is a separate, billed search.
    query: 1-400 characters. limit: 1-20 results.
    """
    return _augment_search(query, provider, limit)


def venice_augment_search_brave(query: str, limit: int = 10) -> AugmentSearchResponse:
    """Search the web with Brave through Venice; suits keyword, news and general
    web searches. Returns JSON with the query and results (title, url, content
    snippet, and date when available). query: 1-400 characters. limit: 1-20 results.
    """
    return _augment_search(query, "brave", limit)


def venice_augment_search_google(query: str, limit: int = 10) -> AugmentSearchResponse:
    """Search the web with Google through Venice; suits keyword, news and general
    web searches. Returns JSON with the query and results (title, url, content
    snippet, and date when available). query: 1-400 characters. limit: 1-20 results.
    """
    return _augment_search(query, "google", limit)


def venice_augment_search_exa(query: str, limit: int = 10) -> AugmentSearchResponse:
    """Search the web with Exa through Venice; suits natural-language, semantic and
    research queries. Returns JSON with the query and results (title, url, content
    snippet, and date when available). query: 1-400 characters. limit: 1-20 results.
    """
    return _augment_search(query, "exa", limit)


AUGMENT_SEARCH_TOOLS = (
    venice_augment_search,
    venice_augment_search_brave,
    venice_augment_search_google,
    venice_augment_search_exa,
)

for _tool in AUGMENT_SEARCH_TOOLS:
    # llm sends __doc__ to the model as the tool description. Python 3.13+ dedents
    # docstrings itself; earlier versions would keep the indentation.
    _tool.__doc__ = inspect.cleandoc(_tool.__doc__ or "")
