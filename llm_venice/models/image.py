"""Venice image generation model implementation."""

import asyncio
from contextlib import suppress
import pathlib
import sys
from typing import Literal, cast

import llm
from llm_venice._options import CatalogOptionsMixin, normalize_policy, validate_request
from ._common import initialize_model, warn_deprecated, model_error_boundary
from llm_venice.client import VeniceClient, AsyncVeniceClient
from llm_venice.params import ImageParams, copy_params
from llm_venice.results import ImageResult
from llm_venice._files import async_file_call, publish_media_file, write_chunks
from pydantic import Field

from llm_venice.utils import (
    generate_timestamp_filename,
    get_unique_filepath,
    require_model_key,
    validate_output_directory,
)
from llm_venice._notices import VeniceNotice, render_notices


class ImageOutputOptions(llm.Options):
    output_dir: pathlib.Path | str | None = None
    output_filename: str | None = None
    overwrite_files: bool = False


class _ImageOptions(ImageParams, ImageOutputOptions):
    """Image endpoint parameters with plugin defaults and output options."""

    safe_mode: bool | None = False
    hide_watermark: bool | None = True
    image_format: Literal["png", "jpeg", "webp"] | None = Field(default="png", alias="format")
    embed_exif_metadata: bool | None = False


def append_blurred_notice(notices: list[VeniceNotice], *, is_blurred: bool) -> None:
    """Record a warning when Venice returns a blurred image."""
    if not is_blurred:
        return
    notices.append(
        VeniceNotice(
            level="warning",
            message="generated image was blurred because Safe Venice filtered adult material",
        )
    )


def _filename_with_index(filename: str, index: int) -> str:
    """Append a 1-based index before the filename extension."""
    path = pathlib.Path(filename)
    return f"{path.stem}_{index}{path.suffix}"


def render_notices_for_output(notices: list[VeniceNotice]) -> str:
    """Render notices for llm model output, separated from following text."""
    rendered = render_notices(notices)
    if not rendered:
        return ""
    return "\n".join(rendered) + "\n"


def render_saved_paths_for_output(output_paths: list[pathlib.Path]) -> str:
    """Render one or more saved image paths for llm model output."""
    return "\n".join(f"Image saved to {saved_path}" for saved_path in output_paths)


def _image_options(model, prompt):
    warn_deprecated(model)
    opts = cast(_ImageOptions, model.Options.model_validate(prompt.options.model_dump()))
    validate_output_directory(opts.output_dir)
    validate_request(model.catalog_policy, prompt.prompt, opts)
    params = copy_params(ImageParams, opts)
    if params.enable_web_search and not model.supports_web_search:
        raise ValueError(f"Model {model.model_id} does not support web search")
    return opts, params


def _save_adapter_images(model, opts, result, paths=None):
    """Save each image, appending to ``paths`` as soon as the file is published."""
    paths = [] if paths is None else paths
    for index, image in enumerate(result.images, 1):
        directory = (
            pathlib.Path(opts.output_dir).expanduser()
            if opts.output_dir
            else llm.user_dir() / "images"
        )
        filename = opts.output_filename or generate_timestamp_filename(
            "venice", model.model_name, image.format
        )
        if len(result.images) > 1:
            filename = _filename_with_index(filename, index)
        path = get_unique_filepath(directory, filename, opts.overwrite_files)
        paths.append(
            write_chunks(
                [image.data], path, overwrite=opts.overwrite_files, publish=publish_media_file
            ).path
        )
    return paths


async def _save_adapter_images_async(model, opts, result):
    paths: list[pathlib.Path] = []
    try:
        return await async_file_call(_save_adapter_images, model, opts, result, paths)
    except asyncio.CancelledError:
        # Cancellation waits for the running save, then drops its return value.
        for path in paths:
            with suppress(OSError, ValueError):
                print(f"Warning: image saved to {path} before cancellation", file=sys.stderr)
        raise


def _image_response(response, result: ImageResult, paths):
    response.response_json = {
        **result.raw,
        "is_blurred": result.blurred,
        "content_violation": result.content_violation,
        "content_policy_violation_count": result.content_policy_violation_count,
        "output_paths": [str(path) for path in paths],
    }


def _image_notices(result):
    notices: list[VeniceNotice] = []
    append_blurred_notice(notices, is_blurred=result.blurred)
    if result.content_policy_violation_count:
        notices.append(
            VeniceNotice(
                "warning",
                f"{result.content_policy_violation_count} image variants omitted by content policy",
            )
        )
    return render_notices_for_output(notices)


class VeniceImage(CatalogOptionsMixin, llm.KeyModel):
    """Venice image adapter, with file naming and moderation output policy."""

    can_stream = False
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    model_name: str
    model_spec: dict
    supports_web_search = False

    def __init__(
        self,
        model_id,
        model_name=None,
        image_constraints=None,
        supports_web_search=False,
        model_spec=None,
    ):
        initialize_model(self, model_id, model_name, model_spec)
        self.catalog_policy = normalize_policy("image", model_spec, image_constraints)
        self.image_constraints = image_constraints
        self.supports_web_search = supports_web_search

    def __str__(self):
        return f"Venice Image: {self.model_id}"

    class Options(_ImageOptions):  # type: ignore[reportIncompatibleVariableOverride]
        pass

    def execute(self, prompt, stream, response, conversation=None, key=None):
        with model_error_boundary():
            api_key = require_model_key(self, key)
            opts, params = _image_options(self, prompt)
            with VeniceClient(key=api_key) as client:
                result = client.generate_image(self.model_name, prompt.prompt, params)
            paths = [] if result.content_violation else _save_adapter_images(self, opts, result)
            _image_response(response, result, paths)
            rendered = _image_notices(result)
            if rendered:
                yield rendered
            if result.content_violation:
                yield "Response marked as content violation; no image was returned."
            else:
                yield render_saved_paths_for_output(paths)


class AsyncVeniceImage(CatalogOptionsMixin, llm.AsyncKeyModel):
    """Venice image adapter using native async network I/O."""

    can_stream = False
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    model_name: str
    model_spec: dict
    supports_web_search = False

    def __init__(
        self,
        model_id,
        model_name=None,
        image_constraints=None,
        supports_web_search=False,
        model_spec=None,
    ):
        initialize_model(self, model_id, model_name, model_spec)
        self.catalog_policy = normalize_policy("image", model_spec, image_constraints)
        self.image_constraints = image_constraints
        self.supports_web_search = supports_web_search

    def __str__(self):
        return f"Venice Image: {self.model_id}"

    Options: type[VeniceImage.Options] = VeniceImage.Options  # type: ignore[reportIncompatibleVariableOverride]

    async def execute(self, prompt, stream, response, conversation=None, key=None):
        with model_error_boundary():
            api_key = require_model_key(self, key)
            opts, params = _image_options(self, prompt)
            async with AsyncVeniceClient(key=api_key) as client:
                result = await client.generate_image(self.model_name, prompt.prompt, params)
            paths = (
                []
                if result.content_violation
                else await _save_adapter_images_async(self, opts, result)
            )
            _image_response(response, result, paths)
            rendered = _image_notices(result)
            if rendered:
                yield rendered
            if result.content_violation:
                yield "Response marked as content violation; no image was returned."
            else:
                yield render_saved_paths_for_output(paths)
