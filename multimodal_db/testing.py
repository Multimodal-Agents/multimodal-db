"""Deterministic stand-ins so tests and dry runs never touch a model or the network.
Configured as ``kind = "fake"`` in mmdb.toml."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback


class FakeEmbedding(BaseEmbedding):
    """Hashed bag of words. Texts sharing words land near each other, which is enough
    for a search test to mean something."""

    dims: int = 256
    model_name: str = "fake-bow"

    def _embed(self, text: str) -> List[float]:
        v = [0.0] * self.dims
        for word in re.findall(r"[a-z0-9_]+", text.lower()):
            h = int(hashlib.blake2b(word.encode(), digest_size=4).hexdigest(), 16)
            v[h % self.dims] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)


class FakeLLM(CustomLLM):
    """Returns ``responses`` in order, then repeats the last one. Prompts are kept in ``calls``."""

    responses: List[str] = Field(default_factory=lambda: ["(fake answer)"])
    calls: List[str] = Field(default_factory=list)
    model_name: str = "fake-llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=8192, num_output=512, model_name=self.model_name)

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.calls.append(prompt)
        i = min(len(self.calls) - 1, len(self.responses) - 1)
        return CompletionResponse(text=self.responses[i])

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        yield self.complete(prompt, formatted=formatted, **kwargs)
