"""Function tools that let a model search the web through Venice's /augment/search.

``venice_augment_search`` lets the model choose the provider; the other tools
always use one provider. Each call is a separate, billed search. The Venice key
resolves on every call, from llm's key store or ``LLM_VENICE_KEY``. The tools are
synchronous, so they work in sync and async chains; async chains wait for them.
Failures raise, and llm reports them to the model as ``Error: ...`` tool results.
"""

import inspect

from pydantic import ValidationError

from .client import VeniceClient
from .types import AugmentSearchProvider, AugmentSearchResponse


def _augment_search(
    query: str, provider: AugmentSearchProvider, limit: int
) -> AugmentSearchResponse:
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
    query: str, provider: AugmentSearchProvider = "brave", limit: int = 10
) -> AugmentSearchResponse:
    """Search the web through Venice. Returns JSON with the query and results
    (title, url, content snippet, and date when available).

    provider: "brave" (default) or "google" suit keyword, news and general web
    searches; "exa" suits natural-language, semantic and research queries.
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
