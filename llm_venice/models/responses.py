"""Venice /responses models: llm's Responses classes adapted to Venice.

Venice stores no responses, so every request carries the full conversation, as
llm's Responses classes already build it from the prompt's messages; they never
send ``previous_response_id``. Venice's compat lane, which serves models through
Chat Completions, streams some events in its own form; the client adapts them.
"""

from contextvars import ContextVar
import copy
from collections.abc import AsyncGenerator, Generator
from itertools import islice
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional

import llm
from llm.default_plugins.openai_models import AsyncResponses, Responses
from pydantic import Field

from llm_venice._options import CatalogOptionsMixin, normalize_policy
from llm_venice.constants import VENICE_API_BASE
from ._common import initialize_model, model_error_boundary, warn_deprecated
from .chat import (
    VeniceWebSearch,
    _ChatOptions,
    _openai_client,
    _split_server_side_tools,
    _wire_value,
)

MODE_HEADER = "x-venice-responses-mode"
# Response headers that say how Venice served the request.
_LANE_HEADERS = ("x-venice-responses-lane", "x-venice-responses-compat-reason")

# Chat options with no /responses equivalent; setting one is an error.
_CHAT_ONLY_OPTIONS = (
    "seed",
    "stop",
    "logit_bias",
    "frequency_penalty",
    "presence_penalty",
    "min_p",
    "top_k",
    "repetition_penalty",
    "stop_token_ids",
    "prompt_cache_key",
    "verbosity",
    "enable_x_search",
    "strip_thinking_response",
    "disable_thinking",
)
# Top-level /responses fields that llm's OpenAI SDK call does not take as arguments.
_TOP_LEVEL_OPTIONS = ("anon_user_id", "fallbacks", "prompt_cache_retention")
_VENICE_PARAMETERS = (
    "character_slug",
    "include_venice_system_prompt",
    "enable_web_search",
    "enable_web_scraping",
    "enable_web_citations",
    "include_search_results_in_stream",
)
# Request fields that llm or this module set; extra_body must not override them.
# Its venice_parameters merge with the options', which the chat options check.
_RESERVED_REQUEST_KEYS = {
    "model",
    "input",
    "instructions",
    "stream",
    "store",
    "include",
    "text",
    "tools",
    "tool_choice",
    "reasoning",
    "max_output_tokens",
}


class _ResponsesOptions(_ChatOptions):
    """The chat options, of which the chat-only ones are refused, and the routing mode."""

    responses_api: Optional[Literal["auto", "native", "compat"]] = Field(
        description=(
            "How Venice serves /responses: 'native' forwards to OpenAI's Responses API "
            "(most OpenAI models), 'compat' translates through Chat Completions, and "
            "'auto' lets Venice choose."
        ),
        default=None,
    )


class _LlmOptions(llm.Options):
    """The options llm's Responses code reads; Venice's are applied separately."""

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    json_object: Optional[bool] = None


def _check_options(model: Any, options: Any) -> None:
    unsupported = [name for name in _CHAT_ONLY_OPTIONS if getattr(options, name, None) is not None]
    if unsupported:
        raise llm.ModelError(
            f"The Venice Responses API does not take {', '.join(unsupported)}; "
            "leave them out, or use Chat Completions without responses_api"
        )
    capabilities = model.model_spec.get("capabilities") or {}
    if isinstance(capabilities, dict) and capabilities.get("supportsE2EE") is True:
        raise llm.ModelError(
            f"{model.model_id} is an E2EE model, which the Venice Responses API does not "
            "serve; use Chat Completions without responses_api"
        )
    collisions = sorted(_RESERVED_REQUEST_KEYS.intersection(options.extra_body or {}))
    if collisions:
        raise llm.ModelError(
            f"extra_body conflicts with fields set by the request: {', '.join(collisions)}"
        )


def _venice_kwargs(
    model: Any, options: Any, server_side_tools: List[Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Add Venice's request fields and mode header to llm's Responses kwargs."""
    reasoning: Dict[str, Any] = dict(kwargs.pop("reasoning", None) or {})
    if options.reasoning is not None:
        reasoning.update(options.reasoning.model_dump(exclude_none=True))
    # As in Chat Completions, reasoning_effort takes precedence over reasoning.effort.
    if options.reasoning_effort is not None:
        reasoning["effort"] = options.reasoning_effort
    if reasoning:
        kwargs["reasoning"] = reasoning

    extra_body: Dict[str, Any] = copy.deepcopy(options.extra_body or {})
    for name in _TOP_LEVEL_OPTIONS:
        value = getattr(options, name, None)
        if value is not None:
            extra_body[name] = _wire_value(name, value)

    venice_parameters = {
        name: getattr(options, name)
        for name in _VENICE_PARAMETERS
        if getattr(options, name, None) is not None
    }
    for tool in server_side_tools:
        if not isinstance(tool, VeniceWebSearch):
            raise llm.ModelError(
                f"{tool.name} is not available with the Venice Responses API; "
                "use Chat Completions without responses_api"
            )
        venice_parameters.update(tool.venice_parameters())
    searching = venice_parameters.get("enable_web_search") in ("on", "auto")
    if (
        searching
        or venice_parameters.get("enable_web_citations")
        or venice_parameters.get("include_search_results_in_stream")
    ) and not model.supports_web_search:
        raise llm.ModelError(f"Model {model.model_id} does not support web search")
    if venice_parameters:
        extra_body["venice_parameters"] = {
            **extra_body.get("venice_parameters", {}),
            **venice_parameters,
        }
    if extra_body:
        kwargs["extra_body"] = extra_body

    if options.responses_api is not None:
        kwargs["extra_headers"] = {MODE_HEADER: options.responses_api}
    return kwargs


# Header dict for the execute() call that is creating its client: set only for
# the first step of llm's execute(), which builds the client, as in chat.py.
_captured_headers: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "venice_responses_headers", default=None
)


def _capture_headers(raw: Any, sink: Optional[Dict[str, str]]) -> None:
    if sink is None:
        return
    for name in _LANE_HEADERS:
        value = raw.headers.get(name)
        if value:
            sink[name] = value


def _summary_parts(item: Any) -> None:
    """Make a reasoning item's summary the parts llm reads.

    Venice's compat lane sends each summary as a plain string, where OpenAI sends
    ``{"type": "summary_text", "text": ...}``; llm would drop the strings.
    """
    summary: Any = getattr(item, "summary", None)
    if getattr(item, "type", None) != "reasoning" or not isinstance(summary, list):
        return
    if any(isinstance(part, str) for part in summary):
        from openai.types.responses.response_reasoning_item import Summary

        item.summary = [
            Summary(type="summary_text", text=part) if isinstance(part, str) else part
            for part in summary
        ]


def _normalize_response(response: Any) -> Any:
    for item in getattr(response, "output", None) or []:
        _summary_parts(item)
    return response


def _failure(event: Any) -> str:
    error: Any = getattr(getattr(event, "response", None), "error", None) or event
    message = getattr(error, "message", None) or "no reason given"
    code = getattr(error, "code", None)
    return f"Venice response failed: {message}" + (f" ({code})" if code else "")


def _translate(event: Any) -> Any:
    """Turn a Venice stream event into the one llm's Responses loop handles."""
    kind = getattr(event, "type", None)
    if kind == "response.reasoning.delta":
        # The compat lane's reasoning text; OpenAI's event for it is reasoning_text.delta.
        return SimpleNamespace(
            type="response.reasoning_text.delta",
            delta=getattr(event, "delta", None) or "",
            item_id=getattr(event, "item_id", None),
        )
    if kind in ("response.output_item.added", "response.output_item.done"):
        _summary_parts(getattr(event, "item", None))
    elif kind in ("response.completed", "response.incomplete"):
        _normalize_response(event.response)
        if kind == "response.incomplete":
            # llm reads the final payload only from response.completed. An incomplete
            # response (max_output_tokens, content_filter) has the same payload, with
            # its status and incomplete_details, and the partial output and usage.
            return SimpleNamespace(type="response.completed", response=event.response)
    elif kind in ("response.failed", "error"):
        raise llm.ModelError(_failure(event))
    return event


def _events(stream: Any) -> Generator[Any, None, None]:
    try:
        for event in stream:
            yield _translate(event)
    finally:
        stream.close()


async def _aevents(stream: Any) -> AsyncGenerator[Any, None]:
    try:
        async for event in stream:
            yield _translate(event)
    finally:
        await stream.close()


def _adapt_client(client: Any, *, async_: bool) -> Any:
    """Have this client's responses.create() read Venice's replies as llm expects.

    It records the routing headers, adapts compat-lane stream events and
    reasoning summaries, and finishes on an incomplete response too.
    """
    sink = _captured_headers.get()
    responses = client.responses
    raw_create = responses.with_raw_response.create

    def create(*args: Any, **kwargs: Any) -> Any:
        raw = raw_create(*args, **kwargs)
        _capture_headers(raw, sink)
        parsed = raw.parse()
        return _events(parsed) if kwargs.get("stream") else _normalize_response(parsed)

    async def acreate(*args: Any, **kwargs: Any) -> Any:
        raw = await raw_create(*args, **kwargs)
        _capture_headers(raw, sink)
        parsed = raw.parse()
        return _aevents(parsed) if kwargs.get("stream") else _normalize_response(parsed)

    # The client is new for this call, so patching it affects nothing else.
    responses.create = acreate if async_ else create
    return client


def _add_headers(response: Any, headers: Dict[str, str]) -> None:
    """Record how Venice served the request (its lane, and why compat) in response_json."""
    if headers and isinstance(response.response_json, dict):
        response.response_json = {**response.response_json, "venice_headers": headers}


class _VeniceResponsesMixin:
    """Venice configuration shared by the sync and async Responses models."""

    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    supports_web_search = False
    model_spec: dict

    def _configure(self, supports_web_search: bool, model_spec: Any) -> None:
        model: Any = self
        # llm's Responses.__init__ assigns its OpenAI Options class to the instance.
        vars(self).pop("Options", None)
        initialize_model(self, model.model_id, model.model_name, model_spec)
        # Without a base URL the OpenAI SDK would send the Venice key to OpenAI.
        model.api_base = model.api_base or VENICE_API_BASE
        model.catalog_policy = normalize_policy("text", model.model_spec)
        model.supports_web_search = supports_web_search

    def __str__(self) -> str:
        return f"Venice Responses: {getattr(self, 'model_id')}"

    @property
    def supported_server_side_tools(self) -> tuple[type[llm.ServerSideTool], ...]:
        return (VeniceWebSearch,) if self.supports_web_search else ()

    def get_client(self, key: Any, *, async_: bool = False) -> Any:
        return _adapt_client(_openai_client(self, key, async_=async_), async_=async_)  # type: ignore[arg-type]

    def _finalize_responses_kwargs(
        self, prompt: Any, stream: bool, instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        model: Any = self
        warn_deprecated(self)
        options = prompt.options
        with model_error_boundary():
            model.Options.model_validate(options.model_dump())
        _check_options(self, options)
        plain, server_side_tools = _split_server_side_tools(prompt)
        plain = copy.copy(plain)
        plain.options = _LlmOptions.model_validate(
            {name: getattr(options, name, None) for name in _LlmOptions.model_fields}
        )
        kwargs = super()._finalize_responses_kwargs(plain, stream, instructions)  # type: ignore[misc]
        return _venice_kwargs(self, options, server_side_tools, kwargs)


def _responses_init_kwargs(model_spec: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    capabilities = (model_spec or {}).get("capabilities") or {}
    if isinstance(capabilities, dict):
        # llm asks for reasoning summaries and encrypted reasoning only for reasoning models.
        kwargs.setdefault("reasoning", capabilities.get("supportsReasoning") is True)
    return kwargs


class VeniceResponses(_VeniceResponsesMixin, CatalogOptionsMixin, Responses):  # type: ignore[reportIncompatibleMethodOverride]
    """Venice model served by /responses; takes the chat options, minus chat-only ones."""

    def __init__(
        self,
        model_id: str,
        *args: Any,
        supports_web_search: bool = False,
        model_spec: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, *args, **_responses_init_kwargs(model_spec, kwargs))
        self._configure(supports_web_search, model_spec)

    class Options(_ResponsesOptions):  # type: ignore[reportIncompatibleVariableOverride]
        pass

    def execute(self, prompt, stream, response, conversation=None, key=None):
        """Run llm's Responses execution, recording Venice's routing headers."""
        events: Any = super().execute(prompt, stream, response, conversation, key)
        headers: Dict[str, str] = {}
        try:
            token = _captured_headers.set(headers)
            try:
                first = list(islice(events, 1))
            finally:
                _captured_headers.reset(token)
            yield from first
            yield from events
        finally:
            events.close()
        _add_headers(response, headers)


class AsyncVeniceResponses(_VeniceResponsesMixin, CatalogOptionsMixin, AsyncResponses):  # type: ignore[reportIncompatibleMethodOverride]
    """Async Venice model served by /responses."""

    def __init__(
        self,
        model_id: str,
        *args: Any,
        supports_web_search: bool = False,
        model_spec: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, *args, **_responses_init_kwargs(model_spec, kwargs))
        self._configure(supports_web_search, model_spec)

    Options = VeniceResponses.Options  # type: ignore[reportIncompatibleVariableOverride]

    async def execute(self, prompt, stream, response, conversation=None, key=None):
        """Run llm's Responses execution, recording Venice's routing headers."""
        events: Any = super().execute(prompt, stream, response, conversation, key)
        headers: Dict[str, str] = {}
        try:
            first: List[Any] = []
            token = _captured_headers.set(headers)
            try:
                async for event in events:
                    first.append(event)
                    break
            finally:
                _captured_headers.reset(token)
            for event in first:
                yield event
            async for event in events:
                yield event
        finally:
            await events.aclose()
        _add_headers(response, headers)
