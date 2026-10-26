"""Venice image generation model implementation."""

import base64
import os
import pathlib
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal, Optional, cast

import httpx2
import llm
from llm_venice._options import CatalogOptionsMixin, normalize_policy, validate_request
from ._common import initialize_model, warn_deprecated, model_error_boundary
from llm_venice.client import VeniceClient, AsyncVeniceClient
from llm_venice.params import ImageParams, copy_params
from llm_venice.results import ImageResult
from llm_venice._files import async_file_call, write_chunks
from .audio import _publish_media_file
from llm.utils import logging_client
from pydantic import Field

from llm_venice.constants import (
    ENDPOINT_IMAGE_GENERATE,
    DEFAULT_IMAGE_FORMAT,
)
from llm_venice.utils import (
    generate_timestamp_filename,
    get_unique_filepath,
    require_model_key,
    validate_output_directory,
)
from llm_venice.api.client import get_auth_headers_with_content_type
from llm_venice.api.errors import raise_api_error
from llm_venice.notices import VeniceNotice, render_notices


class ImageOutputOptions(llm.Options):
    output_dir: pathlib.Path | str | None = None
    output_filename: str | None = None
    overwrite_files: bool = False


class VeniceImageOptions(ImageParams, ImageOutputOptions):
    """Image endpoint parameters with plugin defaults and output options."""

    safe_mode: bool | None = False
    hide_watermark: bool | None = True
    image_format: Literal["png", "jpeg", "webp"] | None = Field(default="png", alias="format")
    embed_exif_metadata: bool | None = False


@dataclass
class ImageGenerationResult:
    image_bytes_list: list[bytes] = dataclass_field(default_factory=list)
    output_paths: list[pathlib.Path] = dataclass_field(default_factory=list)
    response_json: Optional[dict] = None
    content_violation: bool = False
    is_blurred: bool = False
    notices: list[VeniceNotice] = dataclass_field(default_factory=list)


def _is_true_response_header(headers: httpx2.Headers, header_name: str) -> bool:
    """Return True when a Venice boolean response header is explicitly enabled."""
    return headers.get(header_name, "").lower() == "true"


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


def _decode_base64_images(data: dict[str, Any]) -> list[bytes]:
    """Decode the Venice image array from a JSON response."""
    images = data.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Response did not include any images")

    decoded_images = []
    for image_data in images:
        try:
            decoded_images.append(base64.b64decode(image_data))
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image data: {e}") from e
    return decoded_images


def _filename_with_index(filename: str, index: int) -> str:
    """Append a 1-based index before the filename extension."""
    path = pathlib.Path(filename)
    return f"{path.stem}_{index}{path.suffix}"


def _resolve_output_paths(
    *,
    directory: pathlib.Path,
    output_filename: str,
    overwrite_files: bool,
    image_count: int,
) -> list[pathlib.Path]:
    """Resolve one or more image output paths while preserving current overwrite behavior."""
    if image_count < 1:
        raise ValueError("No output paths available to save image")

    if image_count == 1:
        return [get_unique_filepath(directory, output_filename, overwrite_files)]

    return [
        get_unique_filepath(
            directory,
            _filename_with_index(output_filename, index + 1),
            overwrite_files,
        )
        for index in range(image_count)
    ]


def normalize_image_options_for_model(
    *,
    model_name: str,
    options_dict: dict[str, Any],
    image_constraints: Optional[dict[str, Any]] = None,
) -> list[VeniceNotice]:
    """Drop unsupported options and validate supported values when constraints are available."""
    if not image_constraints:
        return []

    dropped_options = []

    aspect_ratio = options_dict.get("aspect_ratio")
    supported_aspect_ratios = image_constraints.get("aspectRatios")
    if aspect_ratio is not None:
        if not supported_aspect_ratios:
            options_dict.pop("aspect_ratio", None)
            dropped_options.append("aspect_ratio")
        elif supported_aspect_ratios is not None:
            if aspect_ratio not in supported_aspect_ratios:
                allowed = ", ".join(supported_aspect_ratios)
                raise ValueError(
                    f"Invalid aspect_ratio '{aspect_ratio}' for model '{model_name}'. "
                    f"Supported values: {allowed}"
                )

    resolution = options_dict.get("resolution")
    supported_resolutions = image_constraints.get("resolutions")
    if resolution is not None:
        if not supported_resolutions:
            options_dict.pop("resolution", None)
            dropped_options.append("resolution")
        elif supported_resolutions is not None:
            if resolution not in supported_resolutions:
                allowed = ", ".join(supported_resolutions)
                raise ValueError(
                    f"Invalid resolution '{resolution}' for model '{model_name}'. "
                    f"Supported values: {allowed}"
                )

    if not dropped_options:
        return []

    dropped = ", ".join(dropped_options)
    return [
        VeniceNotice(
            level="info",
            message=f"dropped unsupported options for model '{model_name}': {dropped}",
        )
    ]


def generate_image_result(
    *,
    prompt: str,
    options: llm.Options,
    model_id: str,
    model_name: str,
    api_key: str,
    supports_web_search: bool = False,
    image_constraints: Optional[dict[str, Any]] = None,
) -> ImageGenerationResult:
    """
    Generate an image via the Venice API without writing to disk.

    Returns the generated image bytes, resolved output paths, and any response metadata.
    """
    options_dict = options.model_dump(by_alias=True)
    output_dir = options_dict.pop("output_dir", None)
    output_filename = options_dict.pop("output_filename", None)
    overwrite_files = options_dict.pop("overwrite_files", False)
    return_binary = options_dict.get("return_binary", False)
    image_format = options_dict.get("format")
    variants = options_dict.get("variants")
    web_search_requested = options_dict.get("enable_web_search")

    if web_search_requested is True and not supports_web_search:
        raise llm.ModelError(f"Model {model_id} does not support web search")
    if return_binary and variants is not None:
        raise ValueError("variants is only supported when return_binary is false")

    resolved_output_dir = validate_output_directory(output_dir)
    notices = normalize_image_options_for_model(
        model_name=model_name,
        options_dict=options_dict,
        image_constraints=image_constraints,
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        **{k: v for k, v in options_dict.items() if v is not None},
    }

    headers = get_auth_headers_with_content_type(api_key)

    # Logging client option like LLM_OPENAI_SHOW_RESPONSES
    if os.environ.get("LLM_VENICE_SHOW_RESPONSES"):
        with logging_client() as client:
            r = client.post(ENDPOINT_IMAGE_GENERATE, headers=headers, json=payload, timeout=120)
            try:
                r.raise_for_status()
            except httpx2.HTTPStatusError as exc:
                raise_api_error("Generating image", exc)

            content_violation = _is_true_response_header(r.headers, "x-venice-is-content-violation")
            is_blurred = _is_true_response_header(r.headers, "x-venice-is-blurred")
            append_blurred_notice(notices, is_blurred=is_blurred)

            if content_violation:
                return ImageGenerationResult(
                    content_violation=True,
                    is_blurred=is_blurred,
                    notices=notices,
                )

            response_json = None
            if return_binary:
                image_bytes_list = [r.content]
                if is_blurred:
                    response_json = {"is_blurred": True}
            else:
                data = r.json()
                response_json = {
                    "request": data["request"],
                    "timing": data["timing"],
                    "is_blurred": is_blurred,
                }
                image_bytes_list = _decode_base64_images(data)
    else:
        r = httpx2.post(ENDPOINT_IMAGE_GENERATE, headers=headers, json=payload, timeout=120)

        try:
            r.raise_for_status()
        except httpx2.HTTPStatusError as exc:
            raise_api_error("Generating image", exc)

        content_violation = _is_true_response_header(r.headers, "x-venice-is-content-violation")
        is_blurred = _is_true_response_header(r.headers, "x-venice-is-blurred")
        append_blurred_notice(notices, is_blurred=is_blurred)

        if content_violation:
            return ImageGenerationResult(
                content_violation=True,
                is_blurred=is_blurred,
                notices=notices,
            )

        response_json = None
        if return_binary:
            image_bytes_list = [r.content]
            if is_blurred:
                response_json = {"is_blurred": True}
        else:
            data = r.json()
            response_json = {
                "request": data["request"],
                "timing": data["timing"],
                "is_blurred": is_blurred,
            }
            image_bytes_list = _decode_base64_images(data)

    target_dir = resolved_output_dir or (llm.user_dir() / "images")

    if not output_filename:
        extension = image_format or DEFAULT_IMAGE_FORMAT
        output_filename = generate_timestamp_filename("venice", model_name, extension)

    output_filepaths = _resolve_output_paths(
        directory=target_dir,
        output_filename=output_filename,
        overwrite_files=overwrite_files,
        image_count=len(image_bytes_list),
    )

    return ImageGenerationResult(
        image_bytes_list=image_bytes_list,
        output_paths=output_filepaths,
        response_json=response_json,
        content_violation=False,
        is_blurred=is_blurred,
        notices=notices,
    )


def save_image_result(result: ImageGenerationResult) -> list[pathlib.Path]:
    """Persist an ImageGenerationResult to disk."""
    if not result.output_paths:
        raise ValueError("No output path available to save image")
    if not result.image_bytes_list:
        raise ValueError("No image bytes available to save")
    if len(result.output_paths) != len(result.image_bytes_list):
        raise ValueError("Image result paths do not match the number of generated images")

    for output_path, image_bytes in zip(result.output_paths, result.image_bytes_list):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
    return result.output_paths


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
    opts = cast(VeniceImageOptions, model.Options.model_validate(prompt.options.model_dump()))
    validate_output_directory(opts.output_dir)
    validate_request(model.catalog_policy, prompt.prompt, opts)
    params = copy_params(ImageParams, opts)
    if params.enable_web_search and not model.supports_web_search:
        raise ValueError(f"Model {model.model_id} does not support web search")
    return opts, params, []


def _save_adapter_images(model, opts, result):
    paths = []
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
                [image.data], path, overwrite=opts.overwrite_files, publish=_publish_media_file
            ).path
        )
    return paths


def _image_response(response, result: ImageResult, paths):
    response.response_json = {
        **result.raw,
        "is_blurred": result.blurred,
        "content_violation": result.content_violation,
        "content_policy_violation_count": result.content_policy_violation_count,
        "output_paths": [str(path) for path in paths],
    }


def _image_notices(result, notices):
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

    class Options(VeniceImageOptions):  # type: ignore[reportIncompatibleVariableOverride]
        pass

    def execute(self, prompt, stream, response, conversation=None, key=None):
        with model_error_boundary():
            api_key = require_model_key(self, key)
            opts, params, notices = _image_options(self, prompt)
            with VeniceClient(key=api_key) as client:
                result = client.generate_image(self.model_name, prompt.prompt, params)
            paths = [] if result.content_violation else _save_adapter_images(self, opts, result)
            _image_response(response, result, paths)
            rendered = _image_notices(result, notices)
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
            opts, params, notices = _image_options(self, prompt)
            async with AsyncVeniceClient(key=api_key) as client:
                result = await client.generate_image(self.model_name, prompt.prompt, params)
            paths = (
                []
                if result.content_violation
                else await async_file_call(_save_adapter_images, self, opts, result)
            )
            _image_response(response, result, paths)
            rendered = _image_notices(result, notices)
            if rendered:
                yield rendered
            if result.content_violation:
                yield "Response marked as content violation; no image was returned."
            else:
                yield render_saved_paths_for_output(paths)
