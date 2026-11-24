"""Terminal-safe rendering of text that Venice returned from the web or a document."""

import unicodedata


def clean_text(value: object, *, single_line: bool = False, strip: bool = True) -> str:
    """Drop terminal control characters, keeping newlines and tabs, and optionally trim the ends."""
    if not isinstance(value, str):
        return ""
    if single_line:
        value = " ".join(value.split())
    cleaned = "".join(
        char for char in value if char in "\n\t" or unicodedata.category(char) != "Cc"
    )
    return cleaned.strip() if strip else cleaned
