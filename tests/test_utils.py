import click
import llm
import pytest

from llm_venice.utils import get_venice_key


def test_get_venice_key_defaults_to_needs_key(monkeypatch):
    """Default behavior should raise NeedsKeyException for library usage."""
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)

    with pytest.raises(llm.NeedsKeyException):
        get_venice_key()


def test_get_venice_key_click_mode(monkeypatch):
    """CLI paths should opt-in to click.ClickException."""
    monkeypatch.setattr(llm, "get_key", lambda *_, **__: None)

    with pytest.raises(click.ClickException):
        get_venice_key(click_exceptions=True)
