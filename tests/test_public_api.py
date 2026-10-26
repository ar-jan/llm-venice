"""Validate the shipped package and a downstream consumer outside the checkout."""

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

import build
from installer import install
from installer.destinations import SchemeDictionaryDestination
from installer.sources import WheelFile
import pytest

import llm_venice

LEGACY = [
    "fetch_models",
    "persist_models",
    "list_characters",
    "persist_characters",
    "list_api_keys",
    "get_rate_limits",
    "get_rate_limits_log",
    "create_api_key",
    "delete_api_key",
    "perform_image_upscale",
    "write_upscaled_image",
    "UpscaleResult",
    "generate_image_result",
    "save_image_result",
    "ImageGenerationResult",
    "generate_speech_result",
    "save_speech_result",
    "stream_speech_result",
    "SpeechGenerationResult",
    "SpeechStreamResult",
    "VeniceNotice",
    "VeniceChatOptions",
    "VeniceImageOptions",
    "VeniceSpeechOptions",
]


def test_public_exports_and_removed_modules():
    for name in llm_venice.__all__:
        assert getattr(llm_venice, name) is not None
    for name in LEGACY:
        assert not hasattr(llm_venice, name), name
    for module in ["llm_venice.api", "llm_venice.api.refresh", "llm_venice.notices"]:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
    assert "register_models" not in llm_venice.__all__
    assert callable(llm_venice.register_models)
    assert callable(llm_venice.register_commands)


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory):
    root = Path(__file__).parents[1]
    work = tmp_path_factory.mktemp("wheel-consumer")
    source = work / "source"
    source.mkdir()
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(root / filename, source / filename)
    shutil.copytree(
        root / "llm_venice",
        source / "llm_venice",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    wheel = Path(build.ProjectBuilder(str(source)).build("wheel", str(work / "dist")))
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "llm_venice/py.typed" in names
        assert not any(
            name.startswith("llm_venice/api/") or "sync-conflict" in name for name in names
        )
        assert "llm_venice/notices.py" not in names
    target = work / "installed"
    scheme = {key: str(target) for key in ("purelib", "platlib", "headers", "scripts", "data")}
    destination = SchemeDictionaryDestination(scheme, sys.executable, "posix")
    with WheelFile.open(wheel) as wheel_file:
        install(wheel_file, destination, additional_metadata={})
    return work, target


def test_installed_wheel_runtime_and_plugin_imports(installed_wheel):
    work, target = installed_wheel
    code = f"""
import sys
sys.path.insert(0, {str(target)!r})
import llm_venice as v
from pathlib import Path
assert Path(v.__file__).is_relative_to({str(target)!r})
for name in v.__all__:
    assert getattr(v, name) is not None
for name in {LEGACY!r}:
    assert not hasattr(v, name), name
from llm_venice.types import ModelInfo
from llm_venice.catalog import save_catalog, catalog_path
assert v.ImageParams(image_format='webp').image_format == 'webp'
assert v.VeniceWebSearch and v.VeniceXSearch
assert v.VeniceSpeech.Options().response_format == 'mp3'
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=work, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_installed_wheel_consumer_types(installed_wheel):
    pyright = shutil.which("pyright")
    if not pyright:
        pytest.skip("consumer type check requires the dev extra")
    work, target = installed_wheel
    config = {
        "include": ["consumer.py"],
        "extraPaths": [str(target)],
        "typeCheckingMode": "strict",
        "pythonVersion": "3.10",
    }
    (work / "pyrightconfig.json").write_text(json.dumps(config))
    (work / "consumer.py").write_text("""from pathlib import Path
from typing_extensions import assert_type
from llm_venice import (
    VeniceClient, AsyncVeniceClient, ImageParams, SpeechParams,
    ImageResult, GeneratedImage, SpeechResult, SpeechStream, AsyncSpeechStream,
    load_catalog, refresh_catalog, VeniceError, VeniceAPIError, MissingKeyError,
    VeniceWebSearch, VeniceXSearch,
)
from llm_venice.types import ModelInfo, Character, ApiKey, CreatedApiKey, RateLimits, RateLimitLogEntry
from llm_venice.catalog import save_catalog, catalog_path

def use(client: VeniceClient) -> None:
    assert_type(client.list_models(), list[ModelInfo])
    assert_type(client.list_models()[0]["id"], str)
    assert_type(client.list_models()[0].get("created"), int | None)
    assert_type(client.list_characters(adult=False), list[Character])
    assert_type(client.list_api_keys(), list[ApiKey])
    assert_type(client.create_api_key(key_type="ADMIN"), CreatedApiKey)
    assert_type(client.get_rate_limits(), RateLimits)
    assert_type(client.get_rate_limits_log(), list[RateLimitLogEntry])
    assert_type(client.generate_image("model", "hello", ImageParams(image_format="webp")), ImageResult)
    assert_type(client.generate_speech("model", "hello", SpeechParams(voice="custom")), SpeechResult)
    assert_type(client.upscale_image(b"image"), GeneratedImage)
    with client.stream_speech("model", "hello") as stream:
        assert_type(stream, SpeechStream)
        assert_type(stream.save("audio.wav"), Path)
    assert_type(load_catalog(), list[ModelInfo] | None)
    assert_type(refresh_catalog(client), list[ModelInfo])
    assert_type(save_catalog(client.list_models()), Path)
    assert_type(catalog_path(), Path)

async def use_async(client: AsyncVeniceClient) -> None:
    assert_type(await client.list_models(), list[ModelInfo])
    assert_type(await client.generate_image("model", "hello"), ImageResult)
    assert_type(await client.generate_speech("model", "hello"), SpeechResult)
    async with client.stream_speech("model", "hello") as stream:
        assert_type(stream, AsyncSpeechStream)
        assert_type(await stream.save("out.wav"), Path)
    async with client.stream_speech("model", "hello") as stream:
        async for chunk in stream.aiter_bytes():
            assert_type(chunk, bytes)
    async with AsyncVeniceClient(key="key") as owned:
        assert_type(owned, AsyncVeniceClient)

def exported_symbols() -> tuple[type[VeniceError], type[VeniceAPIError], type[MissingKeyError], VeniceWebSearch, VeniceXSearch]:
    return VeniceError, VeniceAPIError, MissingKeyError, VeniceWebSearch(), VeniceXSearch()
""")
    result = subprocess.run(
        [pyright, "--pythonpath", sys.executable, "--project", str(work / "pyrightconfig.json")],
        cwd=work,
        env={**os.environ, "PYTHONPATH": str(target)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
