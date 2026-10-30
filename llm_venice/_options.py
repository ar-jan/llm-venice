"""Normalize catalog constraints once and lazily specialize adapter Options."""

from copy import deepcopy
from functools import lru_cache
import json
import math
from typing import Annotated, Any, Callable, Literal, cast

from pydantic import AfterValidator, BaseModel, Field, TypeAdapter, create_model
from pydantic.fields import FieldInfo


class _CatalogOptions:
    """Return base Options on the class and catalog-specialized Options on instances."""

    def __init__(self, base: type[BaseModel]) -> None:
        self.base = base

    def __get__(self, obj: Any, owner: type | None = None) -> type[BaseModel]:
        if obj is None:
            return self.base
        policy: dict[str, Any] = obj.catalog_policy
        # Policies are replaced, never mutated, so identity detects a new policy
        # and avoids re-serializing on every access.
        cached: tuple[dict[str, Any], type[BaseModel]] | None = obj.__dict__.get("_catalog_options")
        if cached is not None and cached[0] is policy:
            return cached[1]
        options = options_class(self.base, json.dumps(policy, sort_keys=True))
        obj.__dict__["_catalog_options"] = (policy, options)
        return options


class CatalogOptionsMixin:
    """Wrap each subclass's own Options so instances see catalog-specialized Options."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        options = cls.__dict__.get("Options")
        if isinstance(options, type) and issubclass(options, BaseModel):
            setattr(cls, "Options", _CatalogOptions(options))


def parse_json_object(value: Any, name: str) -> Any:
    """Parse a JSON object passed as a CLI string; leave other values to field validation.

    llm passes ``-o key value`` options as raw strings, so dict-typed options need
    this as a ``mode="before"`` validator to be usable from the command line.
    """
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError(f"{name} must be a JSON object") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], parsed)


def _object(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _choices(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    values = cast(list[Any], value)
    if not all(isinstance(item, str) and item for item in values):
        return None
    return sorted(set(cast(list[str], values)))


def _positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _bounds(
    value: Any, floor: float | None = None, ceiling: float | None = None
) -> dict[str, float]:
    raw = _object(value)
    bounds: dict[str, float] = {}
    for key, alias in (("min", "ge"), ("max", "le")):
        item: Any = raw.get(key)
        if type(item) in {int, float} and math.isfinite(item):
            bounds[alias] = float(item)
    if "ge" in bounds and floor is not None:
        bounds["ge"] = max(bounds["ge"], floor)
    if "le" in bounds and ceiling is not None:
        bounds["le"] = min(bounds["le"], ceiling)
    if bounds.get("ge", floor or -math.inf) > bounds.get("le", ceiling or math.inf):
        return {}
    return bounds


def normalize_policy(kind: str, spec: Any, image_constraints: Any = None) -> dict[str, Any]:
    metadata = _object(spec)
    constraints = _object(
        image_constraints if image_constraints is not None else metadata.get("constraints")
    )
    capabilities = _object(metadata.get("capabilities"))
    fields: dict[str, Any] = {}
    policy: dict[str, Any] = {"kind": kind, "fields": fields}
    if kind == "image":
        fields["steps"] = _bounds(constraints.get("steps"))
        divisor = _positive_int(constraints.get("widthHeightDivisor"))
        if divisor and divisor <= 1280:
            for name in ("width", "height"):
                fields[name] = {"multiple_of": divisor}
        for name, key in (
            ("aspect_ratio", "aspectRatios"),
            ("resolution", "resolutions"),
            ("quality", "qualities"),
        ):
            values = _choices(constraints.get(key))
            if values is not None:
                if name == "quality" and any(
                    value not in {"low", "medium", "high"} for value in values
                ):
                    continue
                fields[name] = {"choices": values}
        if constraints.get("supportsTransparentBackground") is False:
            fields["transparent_background"] = {"choices": [False]}
        policy["text_limit"] = _positive_int(constraints.get("promptCharacterLimit"))
    elif kind == "tts":
        formats = _choices(metadata.get("supported_formats"))
        known = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
        if formats and all(value in known for value in formats):
            format_rules: dict[str, Any] = {"choices": formats}
            fields["response_format"] = format_rules
            if "mp3" not in formats:
                default = metadata.get("default_format")
                format_rules["default"] = default if default in formats else formats[0]
        speed_rules: dict[str, Any] = dict(_bounds(constraints.get("speed"), 0.25, 4.0))
        if not speed_rules.get("ge", 0.25) <= 1.0 <= speed_rules.get("le", 4.0):
            speed_rules["default"] = None
        fields["speed"] = speed_rules
        policy["text_limit"] = _positive_int(constraints.get("inputCharacterLimit"))
        # Stock voice lists do not establish a closed set: cloned handles and
        # custom provider voice IDs remain valid strings, and omission stays None.
    elif kind == "text":
        if capabilities.get("supportsReasoningEffort") is False:
            fields["reasoning_effort"] = {"choices": []}
        else:
            values = _choices(
                capabilities.get(
                    "reasoningEffortOptions", constraints.get("reasoningEffortOptions")
                )
            )
            if values is not None:
                fields["reasoning_effort"] = {"choices": values}
    return policy


def _restrict_to(choices: list[Any]) -> Callable[[Any], Any]:
    allowed = [*choices, None]

    def check(value: Any) -> Any:
        if value not in allowed:
            raise ValueError(f"Input should be one of: {', '.join(map(str, allowed))}")
        return value

    return check


@lru_cache(maxsize=128)
def options_class(base: type[BaseModel], policy_json: str) -> type[BaseModel]:
    policy: dict[str, Any] = json.loads(policy_json)
    definitions: dict[str, Any] = {}
    for name, raw in cast(dict[str, dict[str, Any]], policy["fields"]).items():
        info: FieldInfo = deepcopy(base.model_fields[name])
        annotation: Any = info.annotation
        rules = dict(raw)
        descriptions: list[str] = []
        if "choices" in rules:
            descriptions.append(
                "Allowed values: "
                + (", ".join(map(str, rules["choices"])) or "none (unsupported)")
                + "."
            )
        for key, label in (("ge", "Minimum"), ("le", "Maximum"), ("multiple_of", "Multiple of")):
            if key in rules:
                descriptions.append(f"{label}: {rules[key]:g}.")
        info.description = " ".join([info.description or "", *descriptions]).strip() or None
        if "choices" in rules:
            choices = rules.pop("choices")
            if choices and all(isinstance(choice, bool) for choice in choices):
                # Keep the base bool type so CLI strings like "false" still coerce;
                # Literal[False] would reject them before the allowed set is checked.
                info.metadata.append(AfterValidator(_restrict_to(choices)))
            else:
                # Dynamic Literal construction preserves useful llm option/schema output.
                annotation = getattr(Literal, "__getitem__")(tuple([*choices, None]))
        if "default" in rules:
            info.default = rules.pop("default")
        info.metadata.extend(Field(**rules).metadata)
        definitions[name] = (annotation, info)
    name = base.__qualname__.replace(".", "")
    return create_model(f"{name}ForModel", __base__=base, **definitions)


def validate_request(policy: dict[str, Any], text: str, options: BaseModel) -> None:
    """Validate positional input lengths with the same normalized model policy."""
    limit: Any = policy.get("text_limit")
    if limit is not None:
        validator = TypeAdapter(Annotated[str, Field(max_length=limit)])
        validator.validate_python(text)
        if policy["kind"] == "image":
            negative: Any = getattr(options, "negative_prompt", None)
            if negative is not None:
                validator.validate_python(negative)
