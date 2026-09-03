"""Endpoint -> LlamaIndex objects. One class pair per kind, imported lazily so an
unused kind costs nothing (the huggingface kind pulls in torch)."""
from __future__ import annotations

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM

from .config import ConfigError, Endpoint

AUTO = "auto"


def _headers(ep: Endpoint) -> dict[str, str]:
    key = ep.resolved_api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def list_models(ep: Endpoint) -> list[str]:
    """What the endpoint says it can serve right now. Empty for in-process kinds."""
    if ep.kind == "openai":
        r = httpx.get(f"{ep.base_url}/models", headers=_headers(ep), timeout=ep.timeout)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    if ep.kind == "ollama":
        r = httpx.get(f"{ep.base_url}/api/tags", timeout=ep.timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    return []


def resolve_model(ep: Endpoint, role: str) -> str:
    """The model name for ``role``; ``"auto"`` asks a served endpoint for its first model."""
    name = ep.embed_model if role == "embed" else ep.chat_model
    if not name:
        raise ConfigError(f"endpoint {ep.name!r} has no {role}_model")
    if name != AUTO:
        return name
    if ep.kind not in ("openai", "ollama"):
        raise ConfigError(f'endpoint {ep.name!r}: {role}_model = "auto" only works for openai and ollama kinds')
    models = list_models(ep)
    if not models:
        raise ConfigError(f"endpoint {ep.name!r} at {ep.base_url} lists no models; load one first")
    return models[0]


def embedding_for(ep: Endpoint) -> BaseEmbedding:
    if ep.kind == "fake":
        from .testing import FakeEmbedding

        return FakeEmbedding()
    model = resolve_model(ep, "embed")
    if ep.kind == "openai":
        from llama_index.embeddings.openai_like import OpenAILikeEmbedding

        return OpenAILikeEmbedding(
            model_name=model,
            api_base=ep.base_url,
            api_key=ep.resolved_api_key() or "none",
            timeout=ep.timeout,
        )
    if ep.kind == "huggingface":
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError:
            raise ConfigError('the huggingface kind needs: pip install "multimodal-db[local]"') from None
        return HuggingFaceEmbedding(model_name=model)
    if ep.kind == "ollama":
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
        except ImportError:
            raise ConfigError('the ollama kind needs: pip install "multimodal-db[ollama]"') from None
        return OllamaEmbedding(model_name=model, base_url=ep.base_url)
    raise ConfigError(f"unknown kind {ep.kind!r}")


def llm_for(ep: Endpoint) -> LLM:
    if ep.kind == "fake":
        from .testing import FakeLLM

        return FakeLLM()
    if ep.kind == "huggingface":
        raise ConfigError(
            f"endpoint {ep.name!r} is huggingface: embeddings only. Point --llm at an openai or ollama endpoint"
        )
    model = resolve_model(ep, "llm")
    if ep.kind == "openai":
        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(
            model=model,
            api_base=ep.base_url,
            api_key=ep.resolved_api_key() or "none",
            is_chat_model=True,
            timeout=ep.timeout,
            context_window=ep.context_window,
        )
    if ep.kind == "ollama":
        try:
            from llama_index.llms.ollama import Ollama
        except ImportError:
            raise ConfigError('the ollama kind needs: pip install "multimodal-db[ollama]"') from None
        return Ollama(model=model, base_url=ep.base_url, request_timeout=ep.timeout, context_window=ep.context_window)
    raise ConfigError(f"unknown kind {ep.kind!r}")


def identity_of(embed: BaseEmbedding, kind: str | None = None) -> str:
    """``kind:model:dims``. A collection records this at first ingest, and a later ingest
    or search under a different identity is refused rather than silently mixing vector
    spaces. The model name alone is not the space: two servers can serve the same name
    with different pooling, and dims catches the coarse version of that."""
    dims = len(embed.get_text_embedding("dimension probe"))
    return f"{kind or type(embed).__name__.lower()}:{embed.model_name}:{dims}"


def embedder_identity(ep: Endpoint, embed: BaseEmbedding) -> str:
    return identity_of(embed, ep.kind)


def probe(ep: Endpoint) -> dict:
    """For ``mmdb endpoints --probe``: reachable, and what it lists."""
    if ep.kind in ("huggingface", "fake"):
        return {"ok": True, "note": "in-process", "models": [m for m in (ep.embed_model,) if m]}
    try:
        return {"ok": True, "models": list_models(ep)}
    except Exception as e:  # a probe reports, it does not raise
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
