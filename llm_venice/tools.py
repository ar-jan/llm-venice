"""Function tools that let a model search the web through Venice's /augment/search.

``venice_augment_search`` requires the model to choose the provider; the other tools
always use one provider. Each call is a separate, billed search. The Venice key
resolves on every call, from llm's key store or ``LLM_VENICE_KEY``. The tools are
synchronous, so they work in sync and async chains; async chains wait for them.
Failures raise, and llm reports them to the model as ``Error: ...`` tool results.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any, ParamSpec, cast, get_args

from pydantic import ValidationError

from ._endpoints import AugmentSearchInput
from .client import VeniceClient
from .types import AugmentSearchProvider, AugmentSearchResponse

_P = ParamSpec("_P")
_PROVIDERS = get_args(AugmentSearchProvider)


def _input_problems(arguments: dict[str, Any], parameters: list[str]) -> list[str]:
    """Every problem with a tool call's bound arguments, in parameter order."""
    problems: dict[str, list[str]] = {}
    fields = {name: arguments[name] for name in ("query", "limit") if name in arguments}
    try:
        AugmentSearchInput.model_validate(fields)
    except ValidationError as exc:
        # Short messages the model can act on, instead of pydantic's report.
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"])
            problems.setdefault(str(error["loc"][0]), []).append(f"{location}: {error['msg']}")
    # A missing or null provider would let Venice apply its default, Brave.
    if "provider" in parameters and arguments.get("provider") not in _PROVIDERS:
        problems["provider"] = ["provider: choose 'brave', 'google' or 'exa' explicitly"]
    return [problem for name in parameters for problem in problems.get(name, [])]


def _validate_tool_call(
    function: Callable[_P, AugmentSearchResponse],
) -> Callable[_P, AugmentSearchResponse]:
    """Report all invalid arguments in one error before the tool searches.

    llm calls a tool with the model's arguments as keywords and checks neither
    them nor the annotations. Missing, unknown and invalid arguments all become
    one ``ValueError`` that the model can correct in a single retry, instead of
    Python's ``TypeError`` or only the first problem.
    """
    signature = inspect.signature(function)
    parameters = list(signature.parameters)

    @functools.wraps(function)
    def tool(*args: _P.args, **kwargs: _P.kwargs) -> AugmentSearchResponse:
        known = {name: value for name, value in kwargs.items() if name in parameters}
        # Extra positional arguments come only from Python callers; keep their TypeError.
        bound = signature.bind_partial(*args, **known)
        bound.apply_defaults()
        problems = _input_problems(dict(bound.arguments), parameters)
        problems += [f"{name}: unexpected argument" for name in kwargs if name not in known]
        if problems:
            raise ValueError(f"Invalid search input: {'; '.join(problems)}")
        return cast(Callable[..., AugmentSearchResponse], function)(**bound.arguments)

    return tool


def _augment_search(
    query: str, provider: AugmentSearchProvider, limit: int
) -> AugmentSearchResponse:
    with VeniceClient() as client:
        return client.augment_search(query, limit=limit, provider=provider)


@_validate_tool_call
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


@_validate_tool_call
def venice_augment_search_brave(query: str, limit: int = 10) -> AugmentSearchResponse:
    """Search the web with Brave through Venice; suits keyword, news and general
    web searches. Returns JSON with the query and results (title, url, content
    snippet, and date when available). query: 1-400 characters. limit: 1-20 results.
    """
    return _augment_search(query, "brave", limit)


@_validate_tool_call
def venice_augment_search_google(query: str, limit: int = 10) -> AugmentSearchResponse:
    """Search the web with Google through Venice; suits keyword, news and general
    web searches. Returns JSON with the query and results (title, url, content
    snippet, and date when available). query: 1-400 characters. limit: 1-20 results.
    """
    return _augment_search(query, "google", limit)


@_validate_tool_call
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
