import click
import llm
import pytest

from llm_venice.utils import require_model_key, validate_output_directory


@pytest.mark.parametrize("value", [None, ""])
def test_require_model_key_rejects_missing_key(value):
    """require_model_key narrows get_key()'s Optional[str] result to str."""

    class FakeModel:
        def get_key(self, explicit_key=None):
            return value

    with pytest.raises(llm.NeedsKeyException, match="No key found for Venice"):
        require_model_key(FakeModel(), None)  # type: ignore[arg-type]


def test_require_model_key_passes_explicit_key():
    class FakeModel:
        def get_key(self, explicit_key=None):
            return explicit_key

    assert require_model_key(FakeModel(), "sk-test") == "sk-test"  # type: ignore[arg-type]


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


def test_validate_output_directory_missing_under_file_parent_raises(tmp_path):
    parent_file = tmp_path / "not_a_dir"
    parent_file.write_text("not a directory")

    with pytest.raises(ValueError, match="is not a writable directory"):
        validate_output_directory(parent_file / "child")


def test_directory_validation_never_prompts(tmp_path, monkeypatch):
    def no_prompt(*args, **kwargs):
        pytest.fail("output directories must not prompt")

    monkeypatch.setattr(click, "confirm", no_prompt)
    assert validate_output_directory(tmp_path / "new") == tmp_path / "new"


def test_library_modules_do_not_import_click():
    import ast
    from pathlib import Path

    root = Path(__file__).parents[1] / "llm_venice"
    for file in root.rglob("*.py"):
        if "cli" in file.relative_to(root).parts:
            continue
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Import):
                assert all(name.name.split(".")[0] != "click" for name in node.names), file
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "click", file
