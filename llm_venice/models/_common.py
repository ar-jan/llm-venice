"""Shared adapter construction and use-time catalog notices."""

from contextlib import contextmanager
from collections.abc import Generator

import llm
from llm.parts import TextPart
from pydantic import ValidationError
from llm_venice.errors import ContentViolationError, VeniceError
from llm_venice.results import ModelDeprecation
from llm_venice._notices import VeniceNotice, render_notice

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


def single_turn_text(model: Any, prompt: Any) -> str:
    """Return the text of the single user turn that a media endpoint receives.

    ``prompt.messages`` is authoritative: with ``messages=`` the user text is
    only there, and positional text outside it is not sent. llm rejects
    assistant and tool messages for single-turn models before execute() runs;
    system messages are ignored, as ``-s`` is. Without a user message the text
    is empty, which request validation rejects.
    """
    user_messages = [message for message in prompt.messages if message.role == "user"]
    if not user_messages:
        return ""
    if len(user_messages) > 1:
        raise ValueError(f"{model.model_id} takes a single prompt, not several user messages")
    parts = user_messages[0].parts
    if not all(isinstance(part, TextPart) for part in parts):
        raise ValueError(f"{model.model_id} accepts only text in its prompt")
    return "\n".join(part.text for part in parts)


def _warn_deprecation_once(model: Any, message: str, replacement: Any) -> None:
    # One warning per model, whether the catalog or a response reported it first.
    if model.model_id in _deprecated_warnings:
        return
    if isinstance(replacement, str) and replacement:
        message += f"; use venice/{normalize_model_id(replacement)} instead"
    _deprecated_warnings.add(model.model_id)
    print(render_notice(VeniceNotice("warning", message)), file=sys.stderr)


def warn_deprecated(model: Any) -> None:
    """Warn before a request when the cached catalog marks the model deprecated."""
    spec = getattr(model, "model_spec", {})
    deprecation = spec.get("deprecation")
    if not isinstance(deprecation, dict) or not deprecation:
        return
    replacement: Any = cast(dict[str, Any], deprecation).get("replacementModelId")
    _warn_deprecation_once(model, f"{model.model_id} is deprecated", replacement)


def warn_response_deprecation(model: Any, deprecation: ModelDeprecation | None) -> None:
    """Warn when response headers report a deprecation the catalog may not know yet."""
    if deprecation is None:
        return
    if deprecation.deprecated:
        message = f"{model.model_id} is deprecated"
    else:
        message = f"{model.model_id} is scheduled for deprecation"
        if deprecation.date:
            message += f" on {deprecation.date}"
    if deprecation.warning:
        message += f": {deprecation.warning}"
    _warn_deprecation_once(model, message, deprecation.replacement)


@contextmanager
def model_error_boundary() -> Generator[None, None, None]:
    """Map expected service/file/input failures while preserving llm key handling."""
    try:
        yield
    except llm.NeedsKeyException:
        raise
    except ContentViolationError as exc:
        # Name the recommended model as llm users select it.
        raise llm.ModelError(exc.describe(model_prefix="venice/")) from exc
    except (VeniceError, ValidationError, ValueError, OSError) as exc:
        raise llm.ModelError(str(exc)) from exc
