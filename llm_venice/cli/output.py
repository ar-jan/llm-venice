"""Terminal-safe rendering of text that Venice returned from the web or a document."""

import unicodedata


def clean_text(value: object, *, single_line: bool = False) -> str:
    """Drop terminal control characters, keeping newlines and tabs, and trim the ends."""
    if not isinstance(value, str):
        return ""
    if single_line:
        value = " ".join(value.split())
    return "".join(
        char for char in value if char in "\n\t" or unicodedata.category(char) != "Cc"
    ).strip()
