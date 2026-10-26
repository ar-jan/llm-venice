"""Shared adapter construction and use-time catalog notices."""

import sys
from typing import Any

_deprecated_warnings: set[str] = set()


def normalize_model_id(value: str) -> str:
    return value.removeprefix("venice/")


def initialize_model(
    model: Any, model_id: str, model_name: str | None, model_spec: dict | None
) -> None:
    model.model_id = f"venice/{normalize_model_id(model_id)}"
    model.model_name = normalize_model_id(model_name or model_id)
    model.model_spec = model_spec or {}


def warn_deprecated(model: Any) -> None:
    spec = getattr(model, "model_spec", {})
    deprecation = spec.get("deprecation")
    if (
        not isinstance(deprecation, dict)
        or not deprecation
        or model.model_id in _deprecated_warnings
    ):
        return
    message = f"{model.model_id} is deprecated"
    replacement = deprecation.get("replacementModelId")
    if isinstance(replacement, str) and replacement:
        message += f"; use venice/{normalize_model_id(replacement)} instead"
    _deprecated_warnings.add(model.model_id)
    print(f"Warning: {message}", file=sys.stderr)
