"""Venice chat model implementation."""

import copy
import json
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

import llm
from llm_venice._options import CatalogOptionsMixin, normalize_policy
from ._common import model_error_boundary
from ._common import normalize_model_id, warn_deprecated
from llm.default_plugins.openai_models import AsyncChat, Chat
from llm_venice.constants import VENICE_PARAMETERS, NON_OPENAI_COMPATIBLE_PARAMS
from pydantic import Field, field_validator


class _ChatOptions(Chat.Options):
    """Options for Venice chat models."""

    if TYPE_CHECKING:
        # llm assembles Chat.Options dynamically, so type checkers cannot see
        # the inherited parameters. Declare only the names here; the real
        # fields and their descriptions come from llm at runtime.
        temperature: Optional[float] = None
        max_tokens: Optional[int] = None

    # Non-standard generation parameters (top-level in extra_body)
    min_p: Optional[float] = Field(
        description=(
            "Sets a minimum probability threshold for token selection. "
            "Tokens with probabilities below this value are filtered out."
        ),
        ge=0,
        le=1,
        default=None,
    )
    top_k: Optional[int] = Field(
        description=(
            "The number of highest probability vocabulary tokens to keep for top-k-filtering."
        ),
        ge=0,
        default=None,
    )
    repetition_penalty: Optional[float] = Field(
        description=(
            "The parameter for repetition penalty. 1.0 means no penalty. "
            "Values > 1.0 discourage repetition."
        ),
        ge=0,
        default=None,
    )
    stop_token_ids: Optional[Union[List[int], str]] = Field(
        description=(
            "Array of token IDs where the API will stop generating further tokens. "
            "When provided via CLI, pass as JSON array string: '[151643, 151645]'"
        ),
        default=None,
    )

    # OpenAI-compatible parameters sent at the top level of the request body
    reasoning_effort: Optional[str] = Field(
        description=(
            "Reasoning effort for models that support it: one of 'none', 'minimal', "
            "'low', 'medium', 'high', 'xhigh' or 'max'."
        ),
        default=None,
    )
    prompt_cache_key: Optional[str] = Field(
        description="Key used to group related requests for prompt caching.",
        default=None,
    )
    prompt_cache_retention: Optional[str] = Field(
        description="Prompt cache retention: 'default', 'extended' or '24h'.",
        default=None,
    )

    # Venice-specific parameters (extra_body.venice_parameters)
    include_venice_system_prompt: Optional[bool] = Field(
        description=(
            "Whether to include Venice-supplied default system prompts alongside your own."
        ),
        default=None,
    )
    enable_web_search: Optional[str] = Field(
        description=("Enable web search: one of 'on', 'off', or 'auto'."),
        default=None,
    )
    enable_web_scraping: Optional[bool] = Field(
        description=("Enable scraping of URLs in the latest user message using."),
        default=None,
    )
    enable_web_citations: Optional[bool] = Field(
        description=(
            "When web search is enabled, request that the LLM cite its sources using ^index^ or ^i,j^ superscript format."
        ),
        default=None,
    )
    include_search_results_in_stream: Optional[bool] = Field(
        description=(
            "Experimental: Include search results in the stream as the first emitted chunk."
        ),
        default=None,
    )
    character_slug: Optional[str] = Field(
        description=("Public character slug to use (e.g. 'alan-watts')."),
        default=None,
    )
    strip_thinking_response: Optional[bool] = Field(
        description=("Strip <think></think> blocks from the response (reasoning models)."),
        default=None,
    )
    disable_thinking: Optional[bool] = Field(
        description=("Disable thinking and strip <think></think> blocks (reasoning models)."),
        default=None,
    )

    @field_validator("enable_web_search")
    def validate_enable_web_search(cls, value):
        if value is None:
            return None
        if value not in ("on", "off", "auto"):
            raise ValueError("enable_web_search must be one of 'on', 'off', or 'auto'")
        return value

    @field_validator("stop_token_ids")
    def validate_stop_token_ids(cls, stop_token_ids):
        """Validate and parse stop_token_ids parameter."""
        if stop_token_ids is None:
            return None

        if isinstance(stop_token_ids, str):
            try:
                parsed = json.loads(stop_token_ids)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in stop_token_ids string")

            if not isinstance(parsed, list):
                raise ValueError("stop_token_ids must be an array")

            # Validate all elements are integers
            if not all(isinstance(x, int) for x in parsed):
                raise ValueError("All elements in stop_token_ids must be integers")

            return parsed

        if isinstance(stop_token_ids, list):
            # Validate all elements are integers
            if not all(isinstance(x, int) for x in stop_token_ids):
                raise ValueError("All elements in stop_token_ids must be integers")
            return stop_token_ids

        raise ValueError("stop_token_ids must be a list or JSON array string")


_WEB_SEARCH_KEYS = (
    "enable_web_search",
    "enable_web_citations",
    "enable_web_scraping",
    "include_search_results_in_stream",
)


class VeniceWebSearch(llm.ServerSideTool):
    """Search the web with Venice's hosted web search.

    ``mode`` is passed through as ``enable_web_search``: ``"auto"`` (default)
    lets Venice decide when to search, ``"on"`` always searches. ``citations``
    asks the model to cite results with ``^index^`` markers, and ``scraping``
    lets Venice scrape URLs in the latest user message. Advanced combinations
    remain available through llm's ``-o`` options.
    """

    name = "venice_web_search"

    def __init__(
        self,
        mode: Literal["auto", "on"] = "auto",
        citations: bool = True,
        scraping: bool = False,
    ):
        super().__init__()
        if mode not in ("auto", "on"):
            raise ValueError("mode must be 'auto' or 'on'")
        if not isinstance(citations, bool) or not isinstance(scraping, bool):
            raise TypeError("citations and scraping must be booleans")
        self.mode = mode
        self.citations = citations
        self.scraping = scraping

    def venice_parameters(self) -> Dict[str, object]:
        """Return the ``venice_parameters`` entries for this tool.

        Both booleans are emitted even when false so that tool configuration
        always takes precedence over conflicting ``-o`` options.
        """
        return {
            "enable_web_search": self.mode,
            "enable_web_citations": self.citations,
            "enable_web_scraping": self.scraping,
        }


class VeniceXSearch(llm.ServerSideTool):
    """Use xAI native search (web plus X/Twitter) on supported Grok models.

    X search replaces Venice web search, as in the official venice-cli, so any
    web-search parameters are dropped when this tool is used.
    """

    name = "venice_x_search"

    def __init__(self):
        super().__init__()

    def venice_parameters(self) -> Dict[str, object]:
        """Return the ``venice_parameters`` entries for this tool."""
        return {"enable_x_search": True}


def _split_server_side_tools(prompt):
    """Split server-side tools out of a prompt so only function tools are sent.

    llm only calls ``prepare_request()`` for Responses-API models, so Venice
    applies server-side tool configuration in ``build_kwargs`` instead.
    """
    server_side_tools = [tool for tool in prompt.tools if isinstance(tool, llm.ServerSideTool)]
    if not server_side_tools:
        return prompt, []
    prompt_without_server_side_tools = copy.copy(prompt)
    prompt_without_server_side_tools.tools = [
        tool for tool in prompt.tools if not isinstance(tool, llm.ServerSideTool)
    ]
    return prompt_without_server_side_tools, server_side_tools


def _apply_server_side_tools(kwargs: Dict, server_side_tools: List) -> Dict:
    """Expose server-side tool configuration as Venice request parameters.

    Tool values are merged before ``_build_venice_kwargs`` so they get the same
    normalization and capability validation as ``-o`` parameters, with the tool
    configuration taking precedence.
    """
    if not server_side_tools:
        return kwargs

    tool_parameters: Dict[str, object] = {}
    x_search_requested = False
    for tool in server_side_tools:
        tool_parameters.update(tool.venice_parameters())
        if isinstance(tool, VeniceXSearch):
            x_search_requested = True

    if x_search_requested:
        # X search replaces Venice web search, as in the official venice-cli.
        for key in _WEB_SEARCH_KEYS:
            kwargs.pop(key, None)
            tool_parameters.pop(key, None)

    kwargs.update(tool_parameters)
    return kwargs


def _make_json_schema_strict(schema: Dict) -> Dict:
    """Recursively set ``additionalProperties: false`` on every object schema.

    Venice (like OpenAI strict mode) requires every object in a JSON schema to
    disallow additional properties, including nested objects, objects in arrays
    and objects under ``$defs``.
    """
    schema_maps = (
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    )
    schema_lists = ("anyOf", "oneOf", "allOf", "prefixItems")
    schema_values = ("items", "not", "contains", "if", "then", "else", "propertyNames")

    for key in schema_maps:
        subschemas = schema.get(key)
        if isinstance(subschemas, dict):
            for subschema in subschemas.values():
                if isinstance(subschema, dict):
                    _make_json_schema_strict(subschema)

    for key in schema_lists:
        subschemas = schema.get(key)
        if isinstance(subschemas, list):
            for subschema in subschemas:
                if isinstance(subschema, dict):
                    _make_json_schema_strict(subschema)

    for key in schema_values:
        subschema = schema.get(key)
        if isinstance(subschema, dict):
            _make_json_schema_strict(subschema)

    schema_type = schema.get("type")
    if (
        schema_type == "object"
        or (isinstance(schema_type, list) and "object" in schema_type)
        or (schema_type is None and "properties" in schema)
    ):
        schema["additionalProperties"] = False

    return schema


def _build_venice_kwargs(model_id: str, supports_web_search: bool, kwargs: Dict) -> Dict:
    """Apply Venice-specific request shaping to kwargs."""
    # Venice requires strict mode and no additional properties for JSON schema.
    # The schema is copied so the caller's prompt.schema is never mutated.
    if "response_format" in kwargs and kwargs["response_format"].get("type") == "json_schema":
        json_schema = kwargs["response_format"]["json_schema"]
        json_schema["strict"] = True
        if isinstance(json_schema.get("schema"), dict):
            json_schema["schema"] = _make_json_schema_strict(copy.deepcopy(json_schema["schema"]))

    # Move non-openai-compatible generation parameters into extra_body
    non_openai_compatible_params = {}

    for param in NON_OPENAI_COMPATIBLE_PARAMS:
        if param in kwargs:
            non_openai_compatible_params[param] = kwargs.pop(param)

    venice_parameters = {}
    for key in VENICE_PARAMETERS:
        if key in kwargs and kwargs[key] is not None:
            venice_parameters[key] = kwargs.pop(key)

    web_search_requested = venice_parameters.get("enable_web_search")
    web_search_enabled = web_search_requested in ("on", "auto")
    web_citations_requested = venice_parameters.get("enable_web_citations")
    include_results_requested = venice_parameters.get("include_search_results_in_stream")

    # Capability guard for web-search-related features
    if (
        web_search_enabled or web_citations_requested or include_results_requested
    ) and not supports_web_search:
        raise llm.ModelError(f"Model {model_id} does not support web search")

    if (web_citations_requested or include_results_requested) and not web_search_enabled:
        raise llm.ModelError(
            "enable_web_search must be set to 'on' or 'auto' when using web citations or including search results in stream"
        )

    # Build extra_body
    if non_openai_compatible_params or venice_parameters:
        kwargs["extra_body"] = {}
        # Non-standard parameters go top-level inside extra_body
        if non_openai_compatible_params:
            kwargs["extra_body"].update(non_openai_compatible_params)
        # Venice-specific parameters go in extra_body.venice_parameters
        if venice_parameters:
            kwargs["extra_body"]["venice_parameters"] = venice_parameters

    return kwargs


class VeniceChat(CatalogOptionsMixin, Chat):
    """Venice AI chat model."""

    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    supports_web_search = False
    supports_x_search = False

    def __init__(
        self,
        *args,
        supports_web_search: bool = False,
        supports_x_search: bool = False,
        model_spec=None,
        **kwargs,
    ):
        if "model_id" in kwargs:
            kwargs["model_id"] = f"venice/{normalize_model_id(kwargs['model_id'])}"
        if "model_name" in kwargs:
            kwargs["model_name"] = normalize_model_id(kwargs["model_name"])
        super().__init__(*args, **kwargs)
        self.model_id = f"venice/{normalize_model_id(self.model_id)}"
        if self.model_name:
            self.model_name = normalize_model_id(self.model_name)
        self.model_spec = model_spec or {}
        self.catalog_policy = normalize_policy("text", self.model_spec)
        self.supports_web_search = supports_web_search
        self.supports_x_search = supports_x_search

    def __str__(self):  # type: ignore[invalid-argument-type]
        return f"Venice Chat: {self.model_id}"

    class Options(_ChatOptions):
        pass

    @property
    def supported_server_side_tools(self) -> tuple[type[llm.ServerSideTool], ...]:
        """Server-side tools allowed by this model's catalog capabilities."""
        tools: list[type[llm.ServerSideTool]] = []
        if self.supports_web_search:
            tools.append(VeniceWebSearch)
        if self.supports_x_search:
            tools.append(VeniceXSearch)
        return tuple(tools)

    def build_kwargs(self, prompt, stream):
        warn_deprecated(self)
        with model_error_boundary():
            self.Options.model_validate(prompt.options.model_dump())
        prompt, server_side_tools = _split_server_side_tools(prompt)
        base_kwargs = super().build_kwargs(prompt, stream)
        base_kwargs = _apply_server_side_tools(base_kwargs, server_side_tools)
        return _build_venice_kwargs(self.model_id, self.supports_web_search, base_kwargs)


class AsyncVeniceChat(CatalogOptionsMixin, AsyncChat):
    """Venice AI async chat model."""

    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    supports_web_search = False
    supports_x_search = False

    def __init__(
        self,
        *args,
        supports_web_search: bool = False,
        supports_x_search: bool = False,
        model_spec=None,
        **kwargs,
    ):
        if "model_id" in kwargs:
            kwargs["model_id"] = f"venice/{normalize_model_id(kwargs['model_id'])}"
        if "model_name" in kwargs:
            kwargs["model_name"] = normalize_model_id(kwargs["model_name"])
        super().__init__(*args, **kwargs)
        self.model_id = f"venice/{normalize_model_id(self.model_id)}"
        if self.model_name:
            self.model_name = normalize_model_id(self.model_name)
        self.model_spec = model_spec or {}
        self.catalog_policy = normalize_policy("text", self.model_spec)
        self.supports_web_search = supports_web_search
        self.supports_x_search = supports_x_search

    def __str__(self):  # type: ignore[invalid-argument-type]
        return f"Venice Chat: {self.model_id}"

    Options = VeniceChat.Options

    @property
    def supported_server_side_tools(self) -> tuple[type[llm.ServerSideTool], ...]:
        """Server-side tools allowed by this model's catalog capabilities."""
        tools: list[type[llm.ServerSideTool]] = []
        if self.supports_web_search:
            tools.append(VeniceWebSearch)
        if self.supports_x_search:
            tools.append(VeniceXSearch)
        return tuple(tools)

    def build_kwargs(self, prompt, stream):
        warn_deprecated(self)
        with model_error_boundary():
            self.Options.model_validate(prompt.options.model_dump())
        prompt, server_side_tools = _split_server_side_tools(prompt)
        base_kwargs = super().build_kwargs(prompt, stream)
        base_kwargs = _apply_server_side_tools(base_kwargs, server_side_tools)
        return _build_venice_kwargs(self.model_id, self.supports_web_search, base_kwargs)
