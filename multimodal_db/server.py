"""One process that owns the store and every registered brain. It speaks MCP to Claude
(``/mcp``) and plain JSON over HTTP to the Obsidian plugin (``/api/...``), so many sessions
and a plugin share one embedded Qdrant instead of fighting over its lock.

    mmdb serve                       # http://127.0.0.1:8765, MCP at /mcp, JSON at /api
    mmdb serve --stdio               # MCP over stdio for a single client

Register with Claude Code:  claude mcp add --transport http mmdb http://127.0.0.1:8765/mcp
Nothing here writes into a vault. ``load`` writes to the store only.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable

import anyio

from . import __version__
from .brain import Brain
from .config import BrainEntry, Config, ConfigError
from .ingest import IngestResult
from .providers import embedder_identity, embedding_for, llm_for
from .search import ask as _ask, search as _search, similar as _similar, similarity_edges
from .store import EmbedderMismatch, Store


def _vault_signature(root: Path) -> tuple[int, float]:
    """Cheap change detector: how many notes, newest mtime. Enough to drop a cached parse."""
    count, newest = 0, 0.0
    for p in root.rglob("*.md"):
        rel = p.relative_to(root).parts
        if any(part.startswith(".") or part in ("data", "tools") for part in rel[:-1]):
            continue
        count += 1
        newest = max(newest, p.stat().st_mtime)
    return count, newest


class Service:
    """What the tools and routes share: one store, one lock, models resolved lazily,
    parsed brains and computed graphs cached until the vault changes."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = Store(cfg.root)
        self.lock = threading.Lock()
        self._embed: Any = None
        self._identity: str | None = None
        self._llm: Any = None
        self._brains: dict[str, tuple[tuple[int, float], Brain]] = {}
        self._graphs: dict[tuple, dict] = {}

    async def run(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        def locked() -> Any:
            with self.lock:
                return fn(*args, **kwargs)

        return await anyio.to_thread.run_sync(locked)

    def close(self) -> None:
        self.store.close()

    # ---- models ---------------------------------------------------------------------------

    def embed(self) -> tuple[Any, str]:
        if self._embed is None:
            ep = self.cfg.endpoint(None, "embed")
            self._embed = embedding_for(ep)
            self._identity = embedder_identity(ep, self._embed)
        return self._embed, self._identity  # type: ignore[return-value]

    def llm(self) -> Any:
        if self._llm is None:
            self._llm = llm_for(self.cfg.endpoint(None, "llm"))
        return self._llm

    # ---- brains ---------------------------------------------------------------------------

    def entry(self, ref: str) -> BrainEntry:
        return self.cfg.resolve_brain(ref)

    def brain(self, ref: str) -> tuple[BrainEntry, Brain]:
        entry = self.entry(ref)
        if not entry.path.is_dir():
            raise ConfigError(f"brain {entry.name!r}: {entry.path} is not a directory")
        sig = _vault_signature(entry.path)
        cached = self._brains.get(entry.name)
        if cached and cached[0] == sig:
            return entry, cached[1]
        b = Brain.open(entry.path)
        b.notes  # parse now, under the lock
        self._brains[entry.name] = (sig, b)
        self._graphs = {k: v for k, v in self._graphs.items() if k[0] != entry.name}
        return entry, b

    def _counts(self, collection: str) -> tuple[int, int]:
        if collection not in self.store.collections():
            return 0, 0
        return self.store.collection(collection).count(), self.store.vector_count(collection)

    def brains(self) -> list[dict[str, Any]]:
        out = []
        for entry in self.cfg.brains.values():
            docs, vectors = self._counts(entry.collection)
            out.append({"name": entry.name, "path": str(entry.path), "exists": entry.path.is_dir(),
                        "collection": entry.collection, "docs": docs, "vectors": vectors})
        return out

    def brain_info(self, ref: str) -> dict[str, Any]:
        entry, b = self.brain(ref)
        info = b.summary()
        docs, vectors = self._counts(entry.collection)
        info.update({"registered_as": entry.name, "collection": entry.collection, "docs": docs, "vectors": vectors})
        return info

    def load(self, ref: str, layers: set[str] | None = None) -> dict[str, Any]:
        entry, b = self.brain(ref)
        embed, identity = self.embed()
        counts: dict[str, int] = {}
        for r in b.index(self.store, embed, identity=identity, collection=entry.collection, layers=layers,
                         chunk_size=self.cfg.chunk_size, chunk_overlap=self.cfg.chunk_overlap):
            counts[r.status] = counts.get(r.status, 0) + 1
        self._graphs = {k: v for k, v in self._graphs.items() if k[0] != entry.name}
        docs, vectors = self._counts(entry.collection)
        return {"brain": entry.name, "collection": entry.collection, "counts": counts, "docs": docs, "vectors": vectors}

    def note(self, ref: str, note: str) -> dict[str, Any]:
        entry, b = self.brain(ref)
        n = b.note(note)
        if n is None:
            raise KeyError(f"no note {note!r} in {entry.name}")
        return {"path": n.path, "id": n.source_id, "name": n.name, "layer": n.layer, "role": n.role,
                "kind": n.kind, "state": n.state, "title": n.title, "tags": n.tags, "links": n.links,
                "generated": n.is_generated, "has_thinking": n.has_thinking,
                "frontmatter": n.frontmatter, "body": n.body, "thinking": n.hand if n.is_generated else n.body}

    # ---- retrieval ------------------------------------------------------------------------

    def _hit(self, h: Any, layer_of: dict[str, str] | None = None) -> dict[str, Any]:
        d = {"score": h.score, "id": h.source_id, "title": h.title, "url": h.url, "text": h.text}
        if layer_of is not None:
            d["layer"] = layer_of.get(h.source_id)
        return d

    def search_collection(self, collection: str, query: str, k: int = 8) -> list[dict[str, Any]]:
        embed, identity = self.embed()
        return [self._hit(h) for h in _search(self.store, collection, query, embed=embed, k=k, identity=identity)]

    def search_brain(self, ref: str, query: str, k: int = 8) -> list[dict[str, Any]]:
        entry, b = self.brain(ref)
        if entry.collection not in self.store.collections():
            raise KeyError(f"brain {entry.name!r} is not indexed yet; load it first")
        embed, identity = self.embed()
        layer_of = {n.source_id: n.layer for n in b.notes}
        hits = _search(self.store, entry.collection, query, embed=embed, k=k, identity=identity)
        return [self._hit(h, layer_of) for h in hits]

    def related(self, ref: str, note: str, k: int = 8) -> list[dict[str, Any]]:
        entry, b = self.brain(ref)
        n = b.note(note)
        if n is None:
            raise KeyError(f"no note {note!r} in {entry.name}")
        layer_of = {x.source_id: x.layer for x in b.notes}
        return [{"id": r.source_id, "title": r.title, "score": r.score, "kind": r.kind, "url": r.url,
                 "layer": layer_of.get(r.source_id)} for r in _similar(self.store, entry.collection, n.source_id, k=k)]

    def ask(self, ref: str, question: str, k: int = 5) -> dict[str, Any]:
        try:
            collection = self.entry(ref).collection
        except ConfigError:
            collection = ref
        embed, identity = self.embed()
        a = _ask(self.store, collection, question, embed=embed, llm=self.llm(), k=k, identity=identity)
        return {"answer": a.text, "sources": [self._hit(h) for h in a.hits]}

    def collections(self) -> list[dict[str, Any]]:
        out = []
        for name in self.store.collections():
            coll = self.store.collection(name)
            out.append({"name": name, "docs": coll.count(), "vectors": self.store.vector_count(name),
                        "embedder": coll.meta.get("embedder")})
        return out

    # ---- graph ----------------------------------------------------------------------------

    def graph(self, ref: str, *, k: int = 5, min_score: float = 0.75, similar: bool = True) -> dict[str, Any]:
        entry, b = self.brain(ref)
        indexed = entry.collection in self.store.collections()
        vectors = self.store.vector_count(entry.collection) if indexed else 0
        key = (entry.name, self._brains[entry.name][0], vectors, k, min_score, similar)
        if key in self._graphs:
            return self._graphs[key]
        nodes = b.nodes()
        strip = lambda p: p[:-3] if p.endswith(".md") else p  # noqa: E731
        graph: dict[str, Any] = {
            "brain": entry.name,
            "indexed": indexed,
            "nodes": [{"id": strip(r["path"]), "name": r["name"], "layer": r["layer"], "kind": r["kind"],
                       "state": r["state"], "title": r["title"], "tags": r["tags"], "in": r["n_in"], "out": r["n_out"],
                       "has_thinking": r["has_thinking"], "generated": r["generated"]} for r in nodes.to_dicts()],
            "links": [{"source": strip(r["src"]), "target": strip(r["dst"]), "kind": "wikilink"}
                      for r in b.edges().to_dicts() if r["resolved"]],
        }
        if similar and indexed and vectors:
            graph["links"] += [{"source": a, "target": c, "kind": "similar", "score": round(s, 4)}
                               for a, c, s in similarity_edges(self.store, entry.collection, k=k, min_score=min_score)]
        self._graphs[key] = graph
        return graph


def build_server(svc: Service):
    """The MCP server plus the JSON routes, on one Starlette app."""
    from mcp.server.mcpserver import MCPServer
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    server = MCPServer(
        "multimodal-db",
        instructions="Search and read the registered brains (Obsidian vaults of working memory) and any other "
                     "collection in the store. list_brains first; search_brain and related_notes return note ids "
                     "you can pass to read_note. Nothing here writes to a vault.",
        version=__version__,
    )

    @server.tool()
    async def list_brains() -> list[dict[str, Any]]:
        """Registered brains with their vault path, collection and index counts."""
        return await svc.run(svc.brains)

    @server.tool()
    async def brain_info(brain: str) -> dict[str, Any]:
        """Layers, ownership split, link graph and tags of one brain (name or vault path)."""
        return await svc.run(svc.brain_info, brain)

    @server.tool()
    async def search_brain(brain: str, query: str, k: int = 8) -> list[dict[str, Any]]:
        """Semantic search over a brain's notes. Returns note ids, titles, layers and the matching chunk."""
        return await svc.run(svc.search_brain, brain, query, k)

    @server.tool()
    async def related_notes(brain: str, note: str, k: int = 8) -> list[dict[str, Any]]:
        """Notes nearest to one note by vector similarity, no query needed. note is a name or path."""
        return await svc.run(svc.related, brain, note, k)

    @server.tool()
    async def read_note(brain: str, note: str) -> dict[str, Any]:
        """One note in full: frontmatter, tags, links, the generated block and the hand-written half."""
        return await svc.run(svc.note, brain, note)

    @server.tool()
    async def search_collection(collection: str, query: str, k: int = 8) -> list[dict[str, Any]]:
        """Semantic search over any collection in the store (a reference cache, arXiv papers...)."""
        return await svc.run(svc.search_collection, collection, query, k)

    @server.tool()
    async def ask(brain_or_collection: str, question: str, k: int = 5) -> dict[str, Any]:
        """Retrieve then answer with the configured LLM endpoint. Sources come back with the answer."""
        return await svc.run(svc.ask, brain_or_collection, question, k)

    @server.tool()
    async def list_collections() -> list[dict[str, Any]]:
        """Every collection in the store with counts and the embedder it is locked to."""
        return await svc.run(svc.collections)

    @server.tool()
    async def load_brain(brain: str) -> dict[str, Any]:
        """Index or refresh a brain's notes into the store. Reads the vault, writes the store only."""
        return await svc.run(svc.load, brain)

    def _json(fn: Callable) -> Callable:
        async def handler(request: Request) -> JSONResponse:
            try:
                return JSONResponse(await fn(request))
            except (KeyError, ConfigError, EmbedderMismatch, ValueError, FileNotFoundError) as e:
                msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
                return JSONResponse({"error": msg}, status_code=404 if isinstance(e, KeyError) else 400)

        return handler

    def _int(request: Request, name: str, default: int) -> int:
        return int(request.query_params.get(name, default))

    @server.custom_route("/api/health", methods=["GET"])
    @_json
    async def health(request: Request) -> dict[str, Any]:
        return {"ok": True, "version": __version__, "root": str(svc.cfg.root), "brains": await svc.run(svc.brains)}

    @server.custom_route("/api/brains", methods=["GET"])
    @_json
    async def brains(request: Request) -> list[dict[str, Any]]:
        return await svc.run(svc.brains)

    @server.custom_route("/api/brains/{name}", methods=["GET"])
    @_json
    async def brain_detail(request: Request) -> dict[str, Any]:
        return await svc.run(svc.brain_info, request.path_params["name"])

    @server.custom_route("/api/brains/{name}/search", methods=["GET"])
    @_json
    async def brain_search(request: Request) -> list[dict[str, Any]]:
        return await svc.run(svc.search_brain, request.path_params["name"], request.query_params.get("q", ""),
                             _int(request, "k", 8))

    @server.custom_route("/api/brains/{name}/related", methods=["GET"])
    @_json
    async def brain_related(request: Request) -> list[dict[str, Any]]:
        return await svc.run(svc.related, request.path_params["name"], request.query_params.get("note", ""),
                             _int(request, "k", 8))

    @server.custom_route("/api/brains/{name}/note", methods=["GET"])
    @_json
    async def brain_note(request: Request) -> dict[str, Any]:
        return await svc.run(svc.note, request.path_params["name"], request.query_params.get("path", ""))

    @server.custom_route("/api/brains/{name}/graph", methods=["GET"])
    @_json
    async def brain_graph(request: Request) -> dict[str, Any]:
        q = request.query_params
        return await svc.run(svc.graph, request.path_params["name"], k=_int(request, "k", 5),
                             min_score=float(q.get("min", 0.75)), similar=q.get("similar", "1") != "0")

    @server.custom_route("/api/brains/{name}/load", methods=["POST"])
    @_json
    async def brain_load(request: Request) -> dict[str, Any]:
        return await svc.run(svc.load, request.path_params["name"])

    @server.custom_route("/api/collections", methods=["GET"])
    @_json
    async def collections(request: Request) -> list[dict[str, Any]]:
        return await svc.run(svc.collections)

    @server.custom_route("/api/search", methods=["GET"])
    @_json
    async def search_any(request: Request) -> list[dict[str, Any]]:
        q = request.query_params
        return await svc.run(svc.search_collection, q.get("collection", ""), q.get("q", ""), _int(request, "k", 8))

    return server


def build_app(cfg: Config, *, host: str = "127.0.0.1"):
    """The Starlette app: MCP at /mcp, JSON at /api/*, CORS open (it only ever binds loopback)."""
    from starlette.middleware.cors import CORSMiddleware

    svc = Service(cfg)
    server = build_server(svc)
    app = server.streamable_http_app(streamable_http_path="/mcp", json_response=True, stateless_http=True, host=host)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.service = svc
    app.state.mcp = server
    return app


def serve(cfg: Config, *, host: str = "127.0.0.1", port: int = 8765, stdio: bool = False) -> None:
    if stdio:
        svc = Service(cfg)
        try:
            build_server(svc).run(transport="stdio")
        finally:
            svc.close()
        return
    import uvicorn

    app = build_app(cfg, host=host)
    try:
        _, identity = app.state.service.embed()  # warm the embedder so the first search is not a model load
        print(f"embedder ready: {identity}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"embedder not ready ({e}); will retry on first request", flush=True)
    try:
        uvicorn.run(app, host=host, port=port, log_level=os.environ.get("MMDB_LOG", "info"))
    finally:
        app.state.service.close()
