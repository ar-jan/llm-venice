import llm
from llm.default_plugins.openai_models import Chat


MODELS = (
    "dolphin-2.9.2-qwen2-72b",
    "llama-3.1-405b",
    "llama-3.2-3b",
    "nous-theta-8b",
)


class VeniceChat(Chat):
    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"

    def __str__(self):
        return f"Venice Chat: {self.model_id}"


@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "venice", "LLM_VENICE_KEY")
    if not key:
        return

    for model_id in MODELS:
        register(
            VeniceChat(
                model_id=f"venice/{model_id}",
                model_name=model_id,
                api_base="https://api.venice.ai/api/v1",
                can_stream=False
            )
        )
