"""Venice embedding model for llm's ``llm embed``, ``embed-multi`` and ``similar``."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import llm
from llm import Attachment
from llm.models import EmbeddingModel

from ._common import initialize_model, model_error_boundary, normalize_model_id, warn_deprecated
from llm_venice.client import VeniceClient
from llm_venice.errors import VeniceResponseError
from llm_venice._endpoints import EMBEDDINGS_ACTION


def _positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


class VeniceEmbeddingModel(EmbeddingModel):
    """Embed text with a Venice embedding model, 100 strings per request by default.

    With ``dimensions``, vectors are shortened to that size and the model ID gains a
    ``-<dimensions>`` suffix, so llm collections never mix sizes under one ID.
    The cached catalog's ``supportsCustomDimensions`` and native
    ``embeddingDimensions`` are checked when present.
    """

    needs_key = "venice"
    key_env_var = "LLM_VENICE_KEY"
    # A list may hold 2048 strings, but long inputs make such a request very large.
    batch_size = 100
    model_name: str
    model_spec: dict[str, Any]

    def __init__(
        self,
        model_id: str,
        model_name: str | None = None,
        model_spec: Mapping[str, Any] | None = None,
        *,
        dimensions: int | None = None,
    ) -> None:
        initialize_model(self, model_id, model_name, dict(model_spec or {}))
        if dimensions is not None:
            self._check_dimensions(dimensions)
            self.model_id = f"venice/{normalize_model_id(model_id)}-{dimensions}"
        self.dimensions = dimensions

    def _check_dimensions(self, dimensions: Any) -> None:
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        native = _positive_int(self.model_spec.get("embeddingDimensions"))
        if native is None:
            # No catalog entry for this model: leave the check to Venice.
            return
        if self.model_spec.get("supportsCustomDimensions") is not True:
            raise ValueError(f"venice/{self.model_name} does not support custom dimensions")
        if dimensions > native:
            raise ValueError(
                f"dimensions must be at most {native}, the native size of venice/{self.model_name}"
            )

    def __str__(self) -> str:
        return f"Venice Embedding: {self.model_id}"

    def embed_batch(
        self,
        items: Iterable[str | bytes | Attachment],
        *,
        key: str | None = None,
    ) -> Iterator[list[float]]:
        texts: list[str] = []
        for item in items:
            # llm rejects bytes and attachments before this, as supports_binary is False.
            if not isinstance(item, str):
                raise llm.ModelError(f"{self.model_id} embeds only text")
            texts.append(item)
        if not texts:
            return iter(())
        with model_error_boundary():
            warn_deprecated(self)
            with VeniceClient(key=key) as client:
                response = client.create_embeddings(
                    self.model_name, texts, dimensions=self.dimensions
                )
            vectors = [item["embedding"] for item in response["data"]]
            for vector in vectors:
                if not isinstance(vector, list):
                    raise VeniceResponseError(EMBEDDINGS_ACTION, "expected a list of floats")
                # A collection needs one size; Venice returns its native size for an
                # unsupported request rather than failing it.
                if self.dimensions is not None and len(vector) != self.dimensions:
                    raise VeniceResponseError(
                        EMBEDDINGS_ACTION,
                        f"requested {self.dimensions} dimensions but received {len(vector)}",
                    )
        return iter([[float(value) for value in vector] for vector in vectors])
