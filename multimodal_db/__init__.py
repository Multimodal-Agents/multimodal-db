"""multimodal-db: a small document store for brains. Polars index, embedded Qdrant
vectors, LlamaIndex retrieval, any OpenAI-compatible endpoint for models."""
from .brain import Brain, BrainSpec, Note
from .config import Config, ConfigError, Endpoint, load_config
from .ingest import Ingester, IngestResult, extract_text
from .providers import embedder_identity, embedding_for, llm_for
from .search import Answer, Hit, QueryResult, ask, query, search
from .store import Collection, Document, EmbedderMismatch, Store

__version__ = "0.2.0"
__all__ = [
    "Answer",
    "Brain",
    "BrainSpec",
    "Note",
    "Collection",
    "Config",
    "ConfigError",
    "Document",
    "EmbedderMismatch",
    "Endpoint",
    "Hit",
    "Ingester",
    "IngestResult",
    "QueryResult",
    "Store",
    "ask",
    "embedder_identity",
    "embedding_for",
    "extract_text",
    "llm_for",
    "load_config",
    "query",
    "search",
]
