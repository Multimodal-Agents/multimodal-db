"""The index. One folder per collection: a Polars parquet of documents, a
content-addressed blob dir and a meta.json. Vectors live in one embedded Qdrant under
the store root, one Qdrant collection per mmdb collection.

The parquet is held in memory and rewritten whole on save, which is fine into the tens
of thousands of documents. Past that, partition it."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .provenance import Ledger, Record

SCHEMA: dict[str, Any] = {
    "doc_id": pl.String,
    "source_id": pl.String,
    "url": pl.String,
    "kind": pl.String,
    "media_type": pl.String,
    "mime": pl.String,
    "title": pl.String,
    "fetched_at": pl.Datetime("us", "UTC"),
    "ingested_at": pl.Datetime("us", "UTC"),
    "content_hash": pl.String,
    "blob_path": pl.String,
    "size": pl.Int64,
    "text": pl.String,
    "tags": pl.List(pl.String),
    "metadata": pl.String,
    "embedder": pl.String,
    "n_chunks": pl.Int64,
}
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EmbedderMismatch(ValueError):
    def __init__(self, collection: str, stored: str, current: str):
        super().__init__(
            f"collection {collection!r} was embedded with {stored}, not {current}. "
            f"Use that endpoint, or move the collection with: mmdb reindex {collection} --embed <name>"
        )
        self.collection, self.stored, self.current = collection, stored, current


@dataclass
class Document:
    """One row of the index. ``source_id`` is the key: a stable id from the crawler
    (``hip-error-codes``, ``arxiv:2401.12345``, a relative path). ``doc_id`` is what the
    vectors are filed under and survives re-ingests."""

    source_id: str
    doc_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    url: str | None = None
    kind: str = "file"
    media_type: str = "text"
    mime: str = "text/plain"
    title: str | None = None
    fetched_at: datetime | None = None
    ingested_at: datetime = field(default_factory=utcnow)
    content_hash: str = ""
    blob_path: str = ""
    size: int = 0
    text: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedder: str | None = None
    n_chunks: int = 0

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["metadata"] = json.dumps(self.metadata, sort_keys=True, default=str)
        row["tags"] = list(self.tags)
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Document":
        row = dict(row)
        row["metadata"] = json.loads(row.get("metadata") or "{}")
        row["tags"] = list(row.get("tags") or [])
        return cls(**row)


class Collection:
    def __init__(self, store: "Store", name: str):
        self.store = store
        self.name = name
        self.path = store.root / "collections" / name
        self.parquet = self.path / "documents.parquet"
        self.blobs = self.path / "blobs"
        self.meta_path = self.path / "meta.json"
        self._meta: dict[str, Any] | None = (
            json.loads(self.meta_path.read_text("utf-8")) if self.meta_path.exists() else None
        )
        self._df: pl.DataFrame | None = None
        self.dirty = False
        self._ledger: Ledger | None = None

    @property
    def exists(self) -> bool:
        return self._meta is not None

    @property
    def meta(self) -> dict[str, Any]:
        if self._meta is None:
            raise KeyError(f"no collection {self.name!r}")
        return self._meta

    def create(self, *, hybrid: bool = False, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.blobs.mkdir(exist_ok=True)
        self._meta = {
            "name": self.name,
            "created": utcnow().isoformat(),
            "embedder": None,
            "generation": 0,
            "vector_collection": None,
            "hybrid": bool(hybrid),
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
        }
        self._write_meta()
        self._df = pl.DataFrame(schema=SCHEMA)
        self.dirty = True
        self.save()

    @property
    def vector_collection(self) -> str:
        """The Qdrant collection holding this generation's vectors."""
        return self.meta.get("vector_collection") or self.name

    def set_embedder(self, identity: str) -> None:
        """Lock the collection to ``identity`` and start a fresh vector generation. A
        new Qdrant collection name every time, never a same-name recreate: embedded
        Qdrant on Windows keeps a deleted collection's vector file mapped, and a
        recreate under the old name reads the old vectors back with the new config."""
        gen = int(self.meta.get("generation", 0)) + 1
        self.meta["embedder"] = identity
        self.meta["generation"] = gen
        self.meta["vector_collection"] = f"{self.name}__g{gen}"
        self._write_meta()

    def _write_meta(self) -> None:
        self.meta_path.write_text(json.dumps(self.meta, indent=2), "utf-8")

    def _load(self) -> pl.DataFrame:
        if self._df is None:
            self._df = pl.read_parquet(self.parquet) if self.parquet.exists() else pl.DataFrame(schema=SCHEMA)
        return self._df

    def frame(self, with_text: bool = False) -> pl.DataFrame:
        """The documents table. ``text`` is dropped unless asked for, it is the bulk of the bytes."""
        df = self._load()
        return df if with_text else df.drop("text")

    def count(self) -> int:
        if self._df is not None:
            return self._df.height
        if self.parquet.exists():
            return int(pl.scan_parquet(self.parquet).select(pl.len()).collect().item())
        return 0

    def __len__(self) -> int:
        return self.count()

    def get(self, source_id: str) -> Document | None:
        rows = self._load().filter(pl.col("source_id") == source_id).to_dicts()
        return Document.from_row(rows[0]) if rows else None

    def upsert(self, doc: Document) -> None:
        df = self._load()
        new = pl.DataFrame([doc.to_row()], schema=SCHEMA)
        self._df = pl.concat([df.filter(pl.col("source_id") != doc.source_id), new], how="vertical")
        self.dirty = True

    def delete(self, source_id: str) -> Document | None:
        existing = self.get(source_id)
        if existing is None:
            return None
        self._df = self._load().filter(pl.col("source_id") != source_id)
        self.dirty = True
        return existing

    @property
    def ledger(self) -> Ledger:
        """What changed in this collection and when, hash-chained per ``source_id``."""
        if self._ledger is None:
            self._ledger = Ledger(self.path / "ledger.jsonl")
        return self._ledger

    def history(self, source_id: str) -> list[Record]:
        return self.ledger.history(source_id)

    def save(self) -> None:
        if not self.dirty or self._df is None:
            return
        tmp = self.parquet.with_suffix(".parquet.tmp")
        self._df.write_parquet(tmp)
        os.replace(tmp, self.parquet)
        self.dirty = False

    def write_blob(self, data: bytes, ext: str = "") -> str:
        """Content-addressed: the same bytes land in the same place however many
        documents point at them. Returns the path relative to the blob dir."""
        digest = hashlib.sha256(data).hexdigest()
        rel = f"{digest[:2]}/{digest}{ext}"
        target = self.blobs / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return rel

    def read_blob(self, rel: str) -> bytes:
        return (self.blobs / rel).read_bytes()

    def stats(self) -> dict[str, Any]:
        df = self.frame()
        by = lambda col: dict(df.group_by(col).len().iter_rows()) if df.height else {}  # noqa: E731
        return {
            "name": self.name,
            "documents": df.height,
            "chunks": int(df["n_chunks"].sum()) if df.height else 0,
            "bytes": int(df["size"].sum()) if df.height else 0,
            "by_kind": by("kind"),
            "by_media_type": by("media_type"),
            "embedder": self.meta.get("embedder"),
            "hybrid": self.meta.get("hybrid", False),
            "chunk_size": self.meta.get("chunk_size"),
            "chunk_overlap": self.meta.get("chunk_overlap"),
            "created": self.meta.get("created"),
        }


class Store:
    """A root directory. Open it, read or write collections, close it (the embedded
    Qdrant holds a lock on its directory while open)."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._qdrant = None
        self._open: dict[str, Collection] = {}

    @property
    def qdrant(self):
        if self._qdrant is None:
            from qdrant_client import QdrantClient

            self._qdrant = QdrantClient(path=str(self.root / "qdrant"))
        return self._qdrant

    def close(self) -> None:
        for coll in self._open.values():
            coll.save()
        if self._qdrant is not None:
            self._qdrant.close()
            self._qdrant = None

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def collections(self) -> list[str]:
        base = self.root / "collections"
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir() if (p / "meta.json").is_file())

    def collection(self, name: str, create: bool = False, **create_kwargs: Any) -> Collection:
        if name in self._open:
            return self._open[name]
        if not NAME_RE.match(name):
            raise ValueError(f"collection name {name!r}: lowercase letters, digits, . _ - only, 64 max")
        coll = Collection(self, name)
        if not coll.exists:
            if not create:
                raise KeyError(f"no collection {name!r} in {self.root}")
            coll.create(**create_kwargs)
        self._open[name] = coll
        return coll

    def drop(self, name: str) -> None:
        """Rows, blobs and every vector generation."""
        for qname in [c.name for c in self.qdrant.get_collections().collections]:
            if qname == name or qname.startswith(f"{name}__g"):
                self.delete_qdrant_collection(qname)
        self._open.pop(name, None)
        path = self.root / "collections" / name
        if path.exists():
            shutil.rmtree(path)

    def delete_qdrant_collection(self, qname: str) -> None:
        if self.qdrant.collection_exists(qname):
            self.qdrant.delete_collection(qname)

    def drop_vectors(self, name: str) -> None:
        self.delete_qdrant_collection(self.collection(name).vector_collection)

    def vector_count(self, name: str) -> int:
        qname = self.collection(name).vector_collection
        if not self.qdrant.collection_exists(qname):
            return 0
        return int(self.qdrant.count(qname, exact=True).count)

    def vector_store(self, name: str):
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        coll = self.collection(name)
        hybrid = bool(coll.meta.get("hybrid"))
        try:
            return QdrantVectorStore(client=self.qdrant, collection_name=coll.vector_collection, enable_hybrid=hybrid)
        except (ImportError, ValueError) as e:
            if hybrid:
                raise RuntimeError('hybrid collections need: pip install "multimodal-db[hybrid]"') from e
            raise
