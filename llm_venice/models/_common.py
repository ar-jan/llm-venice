"""Shared adapter construction and use-time catalog notices."""

from contextlib import contextmanager
from collections.abc import Generator

import llm
from pydantic import ValidationError
from llm_venice.errors import VeniceError

import sys
from typing import Any, cast

_deprecated_warnings: set[str] = set()


def normalize_model_id(value: str) -> str:
    return value.removeprefix("venice/")


def initialize_model(
    model: Any, model_id: str, model_name: str | None, model_spec: dict[str, Any] | None
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
    replacement: Any = cast(dict[str, Any], deprecation).get("replacementModelId")
    if isinstance(replacement, str) and replacement:
        message += f"; use venice/{normalize_model_id(replacement)} instead"
    _deprecated_warnings.add(model.model_id)
    print(f"Warning: {message}", file=sys.stderr)


@contextmanager
def model_error_boundary() -> Generator[None, None, None]:
    """Map expected service/file/input failures while preserving llm key handling."""
    try:
        yield
    except llm.NeedsKeyException:
        raise
    except (VeniceError, ValidationError, ValueError, OSError) as exc:
        raise llm.ModelError(str(exc)) from exc
