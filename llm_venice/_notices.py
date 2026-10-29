"""Shared notice types and rendering helpers."""

from dataclasses import dataclass
from typing import Literal, Sequence


NoticeLevel = Literal["info", "warning"]


@dataclass(frozen=True)
class VeniceNotice:
    """Structured non-fatal notice returned to callers."""

    level: NoticeLevel
    message: str


def render_notice(notice: VeniceNotice) -> str:
    """Render a notice as user-facing text."""
    return f"{notice.level.capitalize()}: {notice.message}"


def render_notices(notices: Sequence[VeniceNotice]) -> list[str]:
    """Render multiple notices as user-facing text."""
    return [render_notice(notice) for notice in notices]
