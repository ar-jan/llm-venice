import json

import click
import httpx
import llm
from llm.default_plugins.openai_models import Chat


MODELS = (
    "deepseek-r1-671b",
    "deepseek-r1-llama-70b",
    "dolphin-2.9.2-qwen2-72b",
    "llama-3.1-405b",
    "llama-3.2-3b",
    "llama-3.3-70b",
    "qwen32b",
)


class VeniceChat(Chat):
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"

    def __str__(self):
        return f"Venice Chat: {self.model_id}"


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
