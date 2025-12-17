import click
import llm
import pytest

import llm_venice.utils as utils
from llm_venice.utils import get_venice_key, validate_output_directory


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


def test_validate_output_directory_treats_empty_as_none():
    assert validate_output_directory(None) is None
    assert validate_output_directory("") is None
    assert validate_output_directory("   ") is None


def test_validate_output_directory_accepts_existing_directory(tmp_path):
    assert validate_output_directory(tmp_path) == tmp_path


def test_validate_output_directory_allows_creatable_missing_directory(tmp_path):
    target = tmp_path / "new" / "nested"
    assert validate_output_directory(target) == target
    assert not target.exists()


def test_validate_output_directory_create_if_missing_prompts_and_creates(monkeypatch, tmp_path):
    target = tmp_path / "created"

    monkeypatch.setattr(utils.click, "get_current_context", lambda *_, **__: object())
    monkeypatch.setattr(utils.click, "confirm", lambda *_, **__: True)

    class DummyStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(utils.sys, "stdin", DummyStdin())

    assert validate_output_directory(target, create_if_missing=True) == target
    assert target.exists()


def test_validate_output_directory_create_if_missing_decline_raises(monkeypatch, tmp_path):
    target = tmp_path / "declined"

    monkeypatch.setattr(utils.click, "get_current_context", lambda *_, **__: object())
    monkeypatch.setattr(utils.click, "confirm", lambda *_, **__: False)

    class DummyStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(utils.sys, "stdin", DummyStdin())

    with pytest.raises(ValueError, match="is not a writable directory"):
        validate_output_directory(target, create_if_missing=True)
    assert not target.exists()


def test_validate_output_directory_missing_under_file_parent_raises(tmp_path):
    parent_file = tmp_path / "not_a_dir"
    parent_file.write_text("not a directory")

    with pytest.raises(ValueError, match="is not a writable directory"):
        validate_output_directory(parent_file / "child")
