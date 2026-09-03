"""Retrieval over a collection. ``search`` needs only an embedder; ``ask`` and ``query``
need an LLM too."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore

from .ingest import make_index
from .providers import identity_of
from .store import Collection, EmbedderMismatch, Store


@dataclass
class Hit:
    score: float
    source_id: str
    doc_id: str
    title: str | None
    url: str | None
    text: str
    node_id: str


@dataclass
class Answer:
    text: str
    hits: list[Hit] = field(default_factory=list)


@dataclass
class QueryResult:
    question: str
    code: str
    result: Any


class QueryError(RuntimeError):
    pass


def _hit(n: NodeWithScore) -> Hit:
    md = n.node.metadata or {}
    return Hit(
        score=float(n.score or 0.0),
        source_id=md.get("source_id", ""),
        doc_id=n.node.ref_doc_id or "",
        title=md.get("title") or None,
        url=md.get("url") or None,
        text=n.node.get_content(),
        node_id=n.node.node_id,
    )


def _prepare(store: Store, collection: str, embed: BaseEmbedding, identity: str | None) -> Collection:
    coll = store.collection(collection)
    stored = coll.meta.get("embedder")
    current = identity or identity_of(embed)
    if stored and stored != current:
        raise EmbedderMismatch(collection, stored, current)
    return coll


def _mode(coll: Collection, hybrid: bool | None) -> str:
    use_hybrid = coll.meta.get("hybrid", False) if hybrid is None else hybrid
    return "hybrid" if use_hybrid else "default"


def search(
    store: Store,
    collection: str,
    query: str,
    *,
    embed: BaseEmbedding,
    k: int = 5,
    identity: str | None = None,
    hybrid: bool | None = None,
) -> list[Hit]:
    coll = _prepare(store, collection, embed, identity)
    if store.vector_count(collection) == 0:
        return []
    index = make_index(store, coll, embed)
    retriever = index.as_retriever(similarity_top_k=k, vector_store_query_mode=_mode(coll, hybrid))
    return [_hit(n) for n in retriever.retrieve(query)]


def ask(
    store: Store,
    collection: str,
    question: str,
    *,
    embed: BaseEmbedding,
    llm: LLM,
    k: int = 5,
    identity: str | None = None,
    hybrid: bool | None = None,
) -> Answer:
    """Retrieve, then let the LLM answer from what was retrieved."""
    coll = _prepare(store, collection, embed, identity)
    if store.vector_count(collection) == 0:
        return Answer("(collection is empty)")
    index = make_index(store, coll, embed)
    engine = index.as_query_engine(llm=llm, similarity_top_k=k, vector_store_query_mode=_mode(coll, hybrid))
    response = engine.query(question)
    return Answer(str(response), [_hit(n) for n in response.source_nodes])


QUERY_PROMPT = """You have a Polars DataFrame named `df`, one row per document in a collection.

Columns:
{schema}

First rows:
{head}

Write one Polars expression over `df` (Polars is imported as `pl`) that answers the
question. Return only the expression: no explanation, no code fence, no assignment.

Question: {question}
Expression:"""

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in ("abs", "dict", "float", "int", "len", "list", "max", "min", "range", "round", "set", "sorted", "str", "sum", "tuple")
}


def query(store: Store, collection: str, question: str, *, llm: LLM, with_text: bool = False) -> QueryResult:
    """Natural language over the documents table: the LLM writes one Polars expression,
    which is evaluated here with only ``df`` and ``pl`` in scope. It is still model-
    written code running on your machine, so the CLI prints the expression first."""
    df = store.collection(collection).frame(with_text=with_text)
    schema = "\n".join(f"  {name}: {dtype}" for name, dtype in df.schema.items())
    with pl.Config(tbl_rows=5, tbl_cols=20, fmt_str_lengths=40):
        head = str(df.head(5))
    prompt = QUERY_PROMPT.format(schema=schema, head=head, question=question)
    code = _FENCE.sub("", llm.complete(prompt).text).strip().rstrip(";")
    try:
        result = eval(code, {"__builtins__": _SAFE_BUILTINS, "pl": pl, "df": df}, {})  # noqa: S307
    except Exception as e:
        raise QueryError(f"the model wrote {code!r}, which failed: {type(e).__name__}: {e}") from e
    return QueryResult(question, code, result)
