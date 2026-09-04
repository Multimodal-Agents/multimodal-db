"""Bytes in, a document row plus embedded chunks out. Crawlers live elsewhere: this
takes what they fetched and a little provenance."""
from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator

from llama_index.core import Document as LIDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter

from .providers import identity_of
from .store import Collection, Document, EmbedderMismatch, Store, utcnow

TEXT_MIMES = {
    "application/json",
    "application/x-yaml",
    "application/yaml",
    "application/toml",
    "application/xml",
    "application/javascript",
}
EXTRA_EXT = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".rst": "text/x-rst",
    ".txt": "text/plain",
    ".toml": "application/toml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".jsonl": "application/json",
    ".py": "text/x-python",
}


def guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in EXTRA_EXT:
        return EXTRA_EXT[ext]
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def media_type_for(mime: str) -> str:
    if mime == "application/pdf":
        return "document"
    if mime.startswith("text/") or mime in TEXT_MIMES:
        return "text"
    for prefix in ("image/", "audio/", "video/"):
        if mime.startswith(prefix):
            return prefix[:-1]
    return "document"


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "noscript", "template"}
    BLOCK = {"p", "div", "br", "li", "tr", "section", "article", "pre", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self._in_title = False
        self.title = ""
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str | None]:
    p = _HTMLText()
    p.feed(html)
    p.close()
    text = re.sub(r"[ \t]+", " ", "".join(p.parts))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text, p.title.strip() or None


def extract_text(data: bytes, mime: str) -> tuple[str, str | None]:
    """(text, title). Empty text for media we cannot read; the blob is still stored."""
    if mime == "text/html":
        return html_to_text(data.decode("utf-8", "replace"))
    if mime.startswith("text/") or mime in TEXT_MIMES:
        text = data.decode("utf-8", "replace")
        title = None
        if mime == "text/markdown":
            m = re.search(r"^#\s+(.+?)\s*$", text, re.M)
            title = m.group(1) if m else None
        return text, title
    if mime == "application/pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return "", None
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        title = getattr(reader.metadata, "title", None) if reader.metadata else None
        return text, (title or None)
    return "", None


def make_splitter(coll: Collection) -> SentenceSplitter:
    return SentenceSplitter(chunk_size=coll.meta["chunk_size"], chunk_overlap=coll.meta["chunk_overlap"])


def make_index(store: Store, coll: Collection, embed: BaseEmbedding) -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(
        store.vector_store(coll.name), embed_model=embed, transformations=[make_splitter(coll)]
    )


@dataclass
class IngestResult:
    status: str  # added | updated | touched | unchanged | reindexed | removed | missing
    document: Document


class Ingester:
    """Holds the embedder, splitter and index for one collection so a batch shares
    them. Use as a context manager, or call ``close()``: saves happen every
    ``save_every`` documents and at close."""

    def __init__(
        self,
        store: Store,
        collection: str,
        embed: BaseEmbedding,
        *,
        identity: str | None = None,
        create: bool = True,
        hybrid: bool = False,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        save_every: int = 1,
        allow_mismatch: bool = False,
    ):
        self.store = store
        self.embed = embed
        self.coll = store.collection(
            collection, create=create, hybrid=hybrid, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.identity = identity or identity_of(embed)
        stored = self.coll.meta.get("embedder")
        if stored and stored != self.identity and not allow_mismatch:
            raise EmbedderMismatch(collection, stored, self.identity)
        if not stored:
            self.coll.set_embedder(self.identity)
        self.splitter = make_splitter(self.coll)
        self.vector_store = store.vector_store(self.coll.name)
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store, embed_model=embed, transformations=[self.splitter]
        )
        self.save_every = max(1, int(save_every))
        self._since_save = 0

    def __enter__(self) -> "Ingester":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.coll.save()
        self._since_save = 0

    def _maybe_save(self) -> None:
        self._since_save += 1
        if self._since_save >= self.save_every:
            self.close()

    def _embed(self, doc: Document) -> int:
        if not doc.text.strip():
            return 0
        li = LIDocument(
            text=doc.text,
            doc_id=doc.doc_id,
            metadata={
                "source_id": doc.source_id,
                "title": doc.title or "",
                "url": doc.url or "",
                "kind": doc.kind,
                "collection": self.coll.name,
            },
            excluded_embed_metadata_keys=["source_id", "url", "kind", "collection"],
            excluded_llm_metadata_keys=["collection"],
        )
        nodes = self.splitter.get_nodes_from_documents([li])
        if nodes:
            self.index.insert_nodes(nodes)
        return len(nodes)

    def _drop_vectors_for(self, doc_id: str) -> None:
        if self.store.qdrant.collection_exists(self.coll.vector_collection):
            self.vector_store.delete(doc_id)

    def ingest_bytes(
        self,
        data: bytes,
        *,
        source_id: str,
        filename: str | None = None,
        url: str | None = None,
        kind: str | None = None,
        title: str | None = None,
        fetched_at: datetime | None = None,
        tags: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
        text: str | None = None,
        force: bool = False,
    ) -> IngestResult:
        """Store ``data`` under ``source_id``. Same bytes under the same id keeps the vectors
        and refreshes only the provenance you passed (``touched``), or is a no-op
        (``unchanged``); changed bytes replace the row and its vectors. ``text`` overrides
        extraction, which is how an image or audio file gets a caption or transcript to
        search on. ``kind`` defaults to ``file`` on first ingest."""
        digest = hashlib.sha256(data).hexdigest()
        existing = self.coll.get(source_id)
        if existing and existing.content_hash == digest and existing.embedder == self.identity and not force:
            return self._touch(existing, url=url, kind=kind, title=title, tags=tags, metadata=metadata)

        name = filename or source_id
        mime = guess_mime(name)
        extracted, found_title = (text, None) if text is not None else extract_text(data, mime)
        ext = Path(name).suffix.lower() or (mimetypes.guess_extension(mime) or "")
        doc = Document(
            source_id=source_id,
            doc_id=existing.doc_id if existing else Document(source_id=source_id).doc_id,
            url=url,
            kind=kind or "file",
            media_type=media_type_for(mime),
            mime=mime,
            title=title or found_title,
            fetched_at=fetched_at,
            ingested_at=utcnow(),
            content_hash=digest,
            blob_path=self.coll.write_blob(data, ext),
            size=len(data),
            text=extracted or "",
            tags=list(tags),
            metadata=dict(metadata or {}),
            embedder=self.identity,
        )
        if existing:
            self._drop_vectors_for(existing.doc_id)
        doc.n_chunks = self._embed(doc)
        self.coll.upsert(doc)
        self._maybe_save()
        return IngestResult("updated" if existing else "added", doc)

    def _touch(self, existing: Document, *, url: str | None, kind: str | None, title: str | None,
               tags: Iterable[str], metadata: dict[str, Any] | None) -> IngestResult:
        """Same bytes: refresh the provenance the caller passed, keep the rest, leave the
        vectors alone. A note that gained a tag or moved cluster is `touched`; a bare
        re-ingest with nothing passed is `unchanged`. Clearing a field takes ``force``."""
        tags = list(tags)
        new = {"url": existing.url if url is None else url, "kind": existing.kind if kind is None else kind,
               "title": existing.title if title is None else title, "tags": tags or list(existing.tags),
               "metadata": existing.metadata if metadata is None else dict(metadata)}
        old = {"url": existing.url, "kind": existing.kind, "title": existing.title,
               "tags": list(existing.tags), "metadata": existing.metadata}
        if new == old:
            return IngestResult("unchanged", existing)
        for key, value in new.items():
            setattr(existing, key, value)
        self.coll.upsert(existing)
        self._maybe_save()
        return IngestResult("touched", existing)

    def remove(self, source_id: str) -> IngestResult:
        """Drop one document's row and vectors. The blob stays, it is content-addressed."""
        doc = self.coll.delete(source_id)
        if doc is None:
            raise KeyError(f"no document {source_id!r} in {self.coll.name!r}")
        self._drop_vectors_for(doc.doc_id)
        self._maybe_save()
        return IngestResult("removed", doc)

    def ingest_file(
        self, path: str | Path, *, source_id: str | None = None, base: str | Path | None = None, **kw: Any
    ) -> IngestResult:
        p = Path(path)
        sid = source_id or (p.relative_to(base).as_posix() if base else p.name)
        if kw.get("fetched_at") is None:
            kw["fetched_at"] = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
        return self.ingest_bytes(p.read_bytes(), source_id=sid, filename=p.name, **kw)

    def ingest_dir(self, directory: str | Path, *, glob: str = "**/*", **kw: Any) -> Iterator[IngestResult]:
        """Every file under ``directory``; ``source_id`` is the relative posix path."""
        base = Path(directory)
        for p in sorted(base.glob(glob)):
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(base).parts):
                yield self.ingest_file(p, base=base, **kw)

    def ingest_manifest(
        self,
        manifest_path: str | Path,
        *,
        base_dir: str | Path | None = None,
        aliases: dict[str, str] | None = None,
        force: bool = False,
    ) -> Iterator[IngestResult]:
        """The manifest ``fetch_reference.py`` writes in a brain's reference cache:
        ``{"sources": [{"id", "kind", "url", "sha256", "at", "ok"}]}``. Each id resolves
        to ``<id>.md`` somewhere under ``base_dir`` (default: next to the manifest), or
        to a directory of files, which ingest as ``<id>/<relative path>``. ``aliases``
        maps an id to a path under ``base_dir`` when neither holds, as with the
        ``gh-prs`` entry that feeds ``upstream-prs/``."""
        mp = Path(manifest_path)
        base = Path(base_dir) if base_dir else mp.parent
        manifest = json.loads(mp.read_text("utf-8"))
        for src in manifest.get("sources", []):
            if not src.get("ok", True):
                continue
            sid, url, kind = src["id"], src.get("url"), src.get("kind", "file")
            fetched = _parse_iso(src.get("at"))
            target = _resolve_manifest_target(base, sid, aliases or {})
            if target is None:
                yield IngestResult("missing", Document(source_id=sid, url=url, kind=kind))
            elif target.is_dir():
                for f in sorted(target.rglob("*")):
                    if f.is_file():
                        yield self.ingest_file(
                            f,
                            source_id=f"{sid}/{f.relative_to(target).as_posix()}",
                            url=url,
                            kind=kind,
                            fetched_at=fetched,
                            metadata={"manifest_id": sid},
                            force=force,
                        )
            else:
                yield self.ingest_file(
                    target,
                    source_id=sid,
                    url=url,
                    kind=kind,
                    fetched_at=fetched,
                    metadata={"manifest_sha256": src.get("sha256")},
                    force=force,
                )

    def reindex(self) -> Iterator[IngestResult]:
        """Re-embed every document from its stored text under this ingester's embedder.
        Needs no source files, so this is how a collection moves between endpoints. The
        new vectors go to a fresh generation; the old one is dropped once they are in."""
        old = self.coll.vector_collection
        self.coll.set_embedder(self.identity)
        self.vector_store = self.store.vector_store(self.coll.name)
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store, embed_model=self.embed, transformations=[self.splitter]
        )
        for row in self.coll.frame(with_text=True).to_dicts():
            doc = Document.from_row(row)
            doc.embedder = self.identity
            doc.n_chunks = self._embed(doc)
            self.coll.upsert(doc)
            self._maybe_save()
            yield IngestResult("reindexed", doc)
        self.close()
        self.store.delete_qdrant_collection(old)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _resolve_manifest_target(base: Path, source_id: str, aliases: dict[str, str]) -> Path | None:
    if source_id in aliases:
        candidate = base / aliases[source_id]
        return candidate if candidate.exists() else None
    for candidate in (base / source_id, base / f"{source_id}.md"):
        if candidate.exists():
            return candidate
    files = sorted(base.rglob(f"{source_id}.md"))
    if files:
        return files[0]
    dirs = sorted(p for p in base.rglob(source_id) if p.is_dir())
    return dirs[0] if dirs else None
