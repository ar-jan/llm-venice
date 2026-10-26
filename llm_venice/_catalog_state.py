"""Process-local registration policy state, separate from catalog persistence."""

from .types import ModelInfo

failed_bootstraps: set[tuple[str, str]] = set()
fetched_catalogs: dict[tuple[str, str], list[ModelInfo]] = {}
