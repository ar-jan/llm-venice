import json
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


MODELS = (
    "deepseek-r1-671b",
    "deepseek-r1-llama-70b",
    "dolphin-2.9.2-qwen2-72b",
    "llama-3.1-405b",
    "llama-3.2-3b",
    "llama-3.3-70b",
    "qwen32b",
)


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


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="venice")
    def venice():
        "llm-venice plugin commands"

    @venice.command(name="refresh")
    def refresh():
        "Refresh the list of models from the Venice API"
        key = llm.get_key("", "venice", "LLM_VENICE_KEY")
        if not key:
            raise click.ClickException("No key found for Venice")
        headers = {"Authorization": f"Bearer {key}"}
        response = httpx.get(
            "https://api.venice.ai/api/v1/models", headers=headers
        )
        response.raise_for_status()
        models = response.json()["data"]
        text_models = [model["id"] for model in models if model.get("type") == "text"]
        if not text_models:
            raise click.ClickException("No text generation models found")
        path = llm.user_dir() / "llm-venice.json"
        path.write_text(json.dumps(text_models, indent=4))
        click.echo(f"{len(text_models)} models saved to {path}", err=True)
        click.echo(json.dumps(text_models, indent=4))

    # Remove and store the original prompt and chat commands
    original_prompt = cli.commands.pop('prompt')
    original_chat = cli.commands.pop('chat')

    def process_venice_options(kwargs):
        """Helper to process venice-specific options"""
        no_venice_sysprompt = kwargs.pop('no_venice_sysprompt', False)
        options = list(kwargs.get('options', []))

        if no_venice_sysprompt:
            model = kwargs.get('model_id')
            if model and model.startswith('venice/'):
                options.append(
                    ('extra_body', '{"venice_parameters": {"include_venice_system_prompt": false}}')
                )
                kwargs['options'] = options
        return kwargs

    # Create new prompt command
    @cli.command(name='prompt')
    @click.option('--no-venice-sysprompt', is_flag=True, help="Disable Venice AI's default system prompt")
    @click.pass_context
    def new_prompt(ctx, no_venice_sysprompt, **kwargs):
        """Execute a prompt"""
        kwargs = process_venice_options({**kwargs, 'no_venice_sysprompt': no_venice_sysprompt})
        return ctx.invoke(original_prompt, **kwargs)

    # Create new chat command
    @cli.command(name='chat')
    @click.option('--no-venice-sysprompt', is_flag=True, help="Disable Venice AI's default system prompt")
    @click.pass_context
    def new_chat(ctx, no_venice_sysprompt, **kwargs):
        """Hold an ongoing chat with a model"""
        kwargs = process_venice_options({**kwargs, 'no_venice_sysprompt': no_venice_sysprompt})
        return ctx.invoke(original_chat, **kwargs)

    # Copy over all params from original commands
    for param in original_prompt.params:
        if param.name != 'no_venice_sysprompt':
            new_prompt.params.append(param)

    for param in original_chat.params:
        if param.name != 'no_venice_sysprompt':
            new_chat.params.append(param)


@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "venice", "LLM_VENICE_KEY")
    if not key:
        return

    path = llm.user_dir() / "llm-venice.json"
    if path.exists():
        model_ids = json.loads(path.read_text())
    else:
        model_ids = MODELS

    for model_id in model_ids:
        register(
            VeniceChat(
                model_id=f"venice/{model_id}",
                model_name=model_id,
                api_base="https://api.venice.ai/api/v1",
                can_stream=True
            )
        )
