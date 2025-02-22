import base64
import datetime
import json
from pathlib import Path
from typing import Optional, Union

import click
import httpx
import llm
from llm.default_plugins.openai_models import Chat

try:
    # Pydantic 2
    from pydantic import field_validator, Field  # type: ignore

except ImportError:
    # Pydantic 1
    from pydantic.fields import Field
    from pydantic.class_validators import validator as field_validator  # type: ignore [no-redef]


class VeniceChatOptions(Chat.Options):
    extra_body: Optional[Union[dict, str]] = Field(
        description=(
            "Additional JSON properties to include in the request body. "
            "When provided via CLI, must be a valid JSON string."
        ),
        default=None,
    )

    @field_validator("extra_body")
    def validate_extra_body(cls, extra_body):
        if extra_body is None:
            return None

        if isinstance(extra_body, str):
            try:
                return json.loads(extra_body)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in extra_body string")

        if not isinstance(extra_body, dict):
            raise ValueError("extra_body must be a dictionary")

        return extra_body


class VeniceChat(Chat):
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"

    def __str__(self):
        return f"Venice Chat: {self.model_id}"

    class Options(VeniceChatOptions):
        pass


class VeniceImageOptions(llm.Options):
    negative_prompt: Optional[str] = Field(
        description="Negative prompt to guide image generation away from certain features",
        default=None,
    )
    style_preset: Optional[str] = Field(
        description="Style preset to use for generation", default=None
    )
    height: Optional[int] = Field(
        description="Height of generated image", default=1024, ge=64, le=1280
    )
    width: Optional[int] = Field(
        description="Width of generated image", default=1024, ge=64, le=1280
    )
    steps: Optional[int] = Field(
        description="Number of inference steps", default=None, ge=7, le=50
    )
    cfg_scale: Optional[float] = Field(
        description="CFG scale for generation", default=None, gt=0, le=20.0
    )
    seed: Optional[int] = Field(
        description="Random seed for reproducible generation",
        default=None,
        ge=-999999999,
        le=999999999,
    )
    lora_strength: Optional[int] = Field(
        description="LoRA adapter strength percentage", default=None, ge=0, le=100
    )
    safe_mode: Optional[bool] = Field(
        description="Enable safety filters", default=False
    )
    hide_watermark: Optional[bool] = Field(
        description="Hide watermark in generated image", default=True
    )
    return_binary: Optional[bool] = Field(
        description="Return raw binary instead of base64", default=False
    )
    output_filename: Optional[str] = Field(
        description="Custom filename for saved image", default=None
    )


class VeniceImage(llm.Model):
    can_stream = False
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"

    def __init__(self, model_id, model_name=None):
        self.model_id = f"venice/{model_id}"
        self.model_name = model_id

    class Options(VeniceImageOptions):
        pass

    def execute(self, prompt, stream, response, conversation=None):
        key = self.get_key()

        options_dict = prompt.options.dict()
        output_filename = options_dict.pop("output_filename", None)
        return_binary = options_dict.get("return_binary", False)

        payload = {
            "model": self.model_name,
            "prompt": prompt.prompt,
            **{k: v for k, v in options_dict.items() if v is not None},
        }

        url = "https://api.venice.ai/api/v1/image/generate"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        r = httpx.post(url, headers=headers, json=payload, timeout=120)

        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ValueError(f"API request failed: {e.response.text}")

        if return_binary:
            image_bytes = r.content
        else:
            data = r.json()
            image_data = data["images"][0]
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image data: {e}")

        image_dir = llm.user_dir() / "images"
        image_dir.mkdir(exist_ok=True)

        if not output_filename:
            datestring = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            output_filename = f"{datestring}_venice_{self.model_name}.png"
        output_filepath = image_dir / output_filename
        output_filepath.write_bytes(image_bytes)
        yield f"Image saved to {output_filepath}"


def refresh_models():
    "Refresh the list of models from the Venice API"
    key = llm.get_key("", "venice", "LLM_VENICE_KEY")
    if not key:
        raise click.ClickException("No key found for Venice")
    headers = {"Authorization": f"Bearer {key}"}

    # Text and image models need to be fetched separately
    text_models = httpx.get(
        "https://api.venice.ai/api/v1/models",
        headers=headers,
        params={"type": "text"},
    )
    text_models.raise_for_status()
    text_models = text_models.json()["data"]

    image_models = httpx.get(
        "https://api.venice.ai/api/v1/models",
        headers=headers,
        params={"type": "image"},
    )
    image_models.raise_for_status()
    image_models = image_models.json()["data"]

    models = text_models + image_models
    if not models:
        raise click.ClickException("No models found")
    path = llm.user_dir() / "venice_models.json"
    path.write_text(json.dumps(models, indent=4))
    click.echo(f"{len(models)} models saved to {path}", err=True)
    click.echo(json.dumps(models, indent=4))

    return models


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="venice")
    def venice():
        "llm-venice plugin commands"

    @venice.command(name="refresh")
    def refresh():
        refresh_models()

    @click.group(name="api-keys", invoke_without_command=True)
    @click.pass_context
    def api_keys(ctx):
        """Manage API keys - list, create, delete, rate-limits"""
        # Retrieve the API key once and store it in context
        key = llm.get_key("", "venice", "LLM_VENICE_KEY")
        if not key:
            raise click.ClickException("No key found for Venice")

        ctx.obj = {"headers": {"Authorization": f"Bearer {key}"}}

        # Default to listing API keys if no subcommand is provided
        if not ctx.invoked_subcommand:
            ctx.invoke(list_keys)

    @api_keys.command(name="list")
    @click.pass_context
    def list_keys(ctx):
        """List all API keys."""
        response = httpx.get(
            "https://api.venice.ai/api/v1/api_keys", headers=ctx.obj["headers"]
        )
        response.raise_for_status()
        click.echo(json.dumps(response.json(), indent=2))

    @api_keys.command(name="rate-limits")
    @click.pass_context
    def rate_limits(ctx):
        "Show current rate limits for your API key"
        response = httpx.get(
            "https://api.venice.ai/api/v1/api_keys/rate_limits",
            headers=ctx.obj["headers"],
        )
        response.raise_for_status()
        click.echo(json.dumps(response.json(), indent=2))

    @api_keys.command(name="create")
    @click.option(
        "--type",
        "key_type",
        type=click.Choice(["ADMIN", "INFERENCE"]),
        required=True,
        help="Type of API key",
    )
    @click.option("--description", default="", help="Description for the new API key")
    @click.option(
        "--expiration-date",
        type=click.DateTime(
            formats=[
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M",
                "%Y-%m-%dT%H:%M:%S",
            ]
        ),
        default=None,
        help="The API Key expiration date",
    )
    @click.pass_context
    def create_key(ctx, description, key_type, expiration_date):
        """Create a new API key."""
        payload = {
            "description": description,
            "apiKeyType": key_type,
            "expiresAt": expiration_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            if expiration_date
            else "",
        }
        response = httpx.post(
            "https://api.venice.ai/api/v1/api_keys",
            headers=ctx.obj["headers"],
            json=payload,
        )
        response.raise_for_status()
        click.echo(json.dumps(response.json(), indent=2))

    @api_keys.command(name="delete")
    @click.argument("api_key_id")
    @click.pass_context
    def delete_key(ctx, api_key_id):
        """Delete an API key by ID."""
        params = {"id": api_key_id}
        response = httpx.delete(
            "https://api.venice.ai/api/v1/api_keys",
            headers=ctx.obj["headers"],
            params=params,
        )
        response.raise_for_status()
        click.echo(json.dumps(response.json(), indent=2))

    # Register api-keys command group under "venice"
    venice.add_command(api_keys)

    # Remove and store the original prompt and chat commands
    # in order to add them back with custom cli options
    original_prompt = cli.commands.pop("prompt")
    original_chat = cli.commands.pop("chat")

    def process_venice_options(kwargs):
        """Helper to process venice-specific options"""
        no_venice_system_prompt = kwargs.pop("no_venice_system_prompt", False)
        character = kwargs.pop("character", None)
        options = list(kwargs.get("options", []))
        model = kwargs.get("model_id")

        if model and model.startswith("venice/"):
            venice_params = {}

            if no_venice_system_prompt:
                venice_params["include_venice_system_prompt"] = False

            if character:
                venice_params["character_slug"] = character

            if venice_params:
                # If a Venice option is used, any `-o extra_body value` is overridden here.
                # TODO: Would prefer to remove the extra_body CLI option, but
                # the implementation is required for venice_parameters.
                options.append(("extra_body", {"venice_parameters": venice_params}))
                kwargs["options"] = options

        return kwargs

    # Create new prompt command
    @cli.command(name="prompt")
    @click.option(
        "--no-venice-system-prompt",
        is_flag=True,
        help="Disable Venice AI's default system prompt",
    )
    @click.option(
        "--character",
        help="Use a Venice AI public character (e.g. 'alan-watts')",
    )
    @click.pass_context
    def new_prompt(ctx, no_venice_system_prompt, character, **kwargs):
        """Execute a prompt"""
        kwargs = process_venice_options(
            {
                **kwargs,
                "no_venice_system_prompt": no_venice_system_prompt,
                "character": character,
            }
        )
        return ctx.invoke(original_prompt, **kwargs)

    # Create new chat command
    @cli.command(name="chat")
    @click.option(
        "--no-venice-system-prompt",
        is_flag=True,
        help="Disable Venice AI's default system prompt",
    )
    @click.option(
        "--character",
        help="Use a Venice AI character (e.g. 'alan-watts')",
    )
    @click.pass_context
    def new_chat(ctx, no_venice_system_prompt, character, **kwargs):
        """Hold an ongoing chat with a model"""
        kwargs = process_venice_options(
            {
                **kwargs,
                "no_venice_system_prompt": no_venice_system_prompt,
                "character": character,
            }
        )
        return ctx.invoke(original_chat, **kwargs)

    # Copy over all params from original commands
    for param in original_prompt.params:
        if param.name not in ("no_venice_system_prompt", "character"):
            new_prompt.params.append(param)

    for param in original_chat.params:
        if param.name not in ("no_venice_system_prompt", "character"):
            new_chat.params.append(param)


@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "venice", "LLM_VENICE_KEY")
    if not key:
        return

    venice_models = llm.user_dir() / "venice_models.json"
    if venice_models.exists():
        models = json.loads(venice_models.read_text())
    else:
        models = refresh_models()

    model_configs = {
        # TODO: get vision config from traits once available
        "qwen-2.5-vl": {"vision": True},
    }

    for model in models:
        model_id = model["id"]
        if model.get("type") == "text":
            register(
                VeniceChat(
                    model_id=f"venice/{model_id}",
                    model_name=model_id,
                    api_base="https://api.venice.ai/api/v1",
                    can_stream=True,
                    **model_configs.get(model_id, {}),
                )
            )
        elif model.get("type") == "image":
            register(VeniceImage(model_id=model_id, model_name=model_id))
