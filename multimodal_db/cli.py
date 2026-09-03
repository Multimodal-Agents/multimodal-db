"""mmdb, the command line. Endpoints come from mmdb.toml; see mmdb.example.toml."""
from __future__ import annotations

import functools
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import click

from . import __version__
from .config import Config, ConfigError, load_config
from .ingest import Ingester, IngestResult
from .providers import embedder_identity, embedding_for, llm_for, probe
from .search import QueryError, ask as _ask, query as _query, search as _search
from .store import EmbedderMismatch, Store


class State:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._store: Store | None = None

    @property
    def store(self) -> Store:
        if self._store is None:
            self._store = Store(self.cfg.root)
        return self._store

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


pass_state = click.make_pass_decorator(State)


def guarded(fn: Callable) -> Callable:
    """Library errors become one-line CLI errors."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (ConfigError, EmbedderMismatch, KeyError, ValueError, RuntimeError) as e:
            msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
            raise click.ClickException(msg) from e

    return wrapper


def _embedder(state: State, name: str | None):
    ep = state.cfg.endpoint(name, "embed")
    embed = embedding_for(ep)
    return embed, embedder_identity(ep, embed)


def _llm(state: State, name: str | None):
    return llm_for(state.cfg.endpoint(name, "llm"))


def _table(rows: list[list[Any]], headers: list[str]) -> str:
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()  # noqa: E731
    return "\n".join([line(headers), line(["-" * w for w in widths]), *(line(r) for r in cells)])


def _dt(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d") if value else "-"


def _report(r: IngestResult) -> None:
    doc = r.document
    click.echo(f"{r.status:<10} {doc.source_id}  ({doc.n_chunks} chunks, {doc.media_type})")


def _summary(results: list[IngestResult]) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    click.echo(", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing to do")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="mmdb")
@click.option("--config", "config_path", type=click.Path(dir_okay=False), default=None,
              help="mmdb.toml to use (default: $MMDB_CONFIG, then ./mmdb.toml).")
@click.option("--root", type=click.Path(file_okay=False), default=None,
              help="Store root (default: $MMDB_ROOT, then [store].root, then ./data).")
@click.pass_context
def main(ctx: click.Context, config_path: str | None, root: str | None) -> None:
    """A small multimodal document store for brains."""
    try:
        cfg = load_config(config_path, root)
    except ConfigError as e:
        raise click.ClickException(str(e)) from e
    ctx.obj = State(cfg)
    ctx.call_on_close(ctx.obj.close)


@main.command()
@pass_state
@guarded
def init(state: State) -> None:
    """Create the store root and say where things are."""
    click.echo(f"store    {state.store.root.resolve()}")
    if state.cfg.path:
        click.echo(f"config   {state.cfg.path.resolve()}")
    else:
        click.echo("config   none (built-in 'local' endpoint; copy mmdb.example.toml to mmdb.toml to add more)")
    click.echo(f"embed    {state.cfg.default_embed or '-'}")
    click.echo(f"llm      {state.cfg.default_llm or '-'}")


@main.command()
@click.option("--probe", "do_probe", is_flag=True, help="Ask each served endpoint what it lists.")
@pass_state
@guarded
def endpoints(state: State, do_probe: bool) -> None:
    """List configured endpoints."""
    headers = ["name", "kind", "base_url", "embed_model", "chat_model"] + (["status"] if do_probe else [])
    rows = []
    for ep in state.cfg.endpoints.values():
        row = [ep.name, ep.kind, ep.base_url or "-", ep.embed_model or "-", ep.chat_model or "-"]
        if do_probe:
            p = probe(ep)
            row.append("ok " + ", ".join(p.get("models", [])[:5]) if p["ok"] else p["error"])
        rows.append(row)
    click.echo(_table(rows, headers))


@main.command()
@pass_state
@guarded
def collections(state: State) -> None:
    """List collections with document and vector counts."""
    rows = []
    for name in state.store.collections():
        coll = state.store.collection(name)
        rows.append([name, coll.count(), state.store.vector_count(name), coll.meta.get("embedder") or "-"])
    click.echo(_table(rows, ["collection", "docs", "vectors", "embedder"]) if rows else "no collections")


@main.command()
@click.argument("collection")
@click.option("--kind", help="Only this kind.")
@click.option("--tag", help="Only documents carrying this tag.")
@click.option("--json", "as_json", is_flag=True)
@pass_state
@guarded
def ls(state: State, collection: str, kind: str | None, tag: str | None, as_json: bool) -> None:
    """List documents in a collection."""
    import polars as pl

    df = state.store.collection(collection).frame()
    if kind:
        df = df.filter(pl.col("kind") == kind)
    if tag:
        df = df.filter(pl.col("tags").list.contains(tag))
    df = df.sort("source_id")
    if as_json:
        click.echo(json.dumps(df.to_dicts(), default=str, indent=1))
        return
    rows = [
        [r["source_id"], r["kind"], r["media_type"], r["size"], r["n_chunks"], _dt(r["fetched_at"]), (r["title"] or "")[:50]]
        for r in df.to_dicts()
    ]
    click.echo(_table(rows, ["source_id", "kind", "media", "bytes", "chunks", "fetched", "title"]) if rows else "empty")


@main.command()
@click.argument("collection")
@pass_state
@guarded
def info(state: State, collection: str) -> None:
    """Counts, sizes and the embedder a collection is locked to."""
    stats = state.store.collection(collection).stats()
    stats["vectors"] = state.store.vector_count(collection)
    for k, v in stats.items():
        click.echo(f"{k:<14} {json.dumps(v) if isinstance(v, dict) else v}")


@main.command()
@click.argument("collection")
@click.argument("source_id")
@click.option("--text", "with_text", is_flag=True, help="Print the extracted text too.")
@pass_state
@guarded
def show(state: State, collection: str, source_id: str, with_text: bool) -> None:
    """One document's row."""
    doc = state.store.collection(collection).get(source_id)
    if doc is None:
        raise click.ClickException(f"no document {source_id!r} in {collection!r}")
    for k, v in doc.to_row().items():
        if k == "text":
            continue
        click.echo(f"{k:<14} {v if v not in (None, '') else '-'}")
    if with_text:
        click.echo("")
        click.echo(doc.text)


@main.command()
@click.argument("collection")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--embed", "embed_name", help="Endpoint to embed with.")
@click.option("--source-id", help="Id for a single file (default: file name).")
@click.option("--url", help="Where a single file came from.")
@click.option("--kind", default="file", show_default=True, help="Provenance kind: file, web, arxiv, conversation...")
@click.option("--title")
@click.option("--tag", "tags", multiple=True, help="Repeatable.")
@click.option("--glob", default="**/*", show_default=True, help="Which files under a directory.")
@click.option("--force", is_flag=True, help="Re-embed even if the bytes are unchanged.")
@click.option("--hybrid", is_flag=True, help="When creating: dense+sparse search (needs the hybrid extra).")
@pass_state
@guarded
def ingest(state: State, collection: str, paths: tuple[str, ...], embed_name: str | None, source_id: str | None,
           url: str | None, kind: str, title: str | None, tags: tuple[str, ...], glob: str, force: bool,
           hybrid: bool) -> None:
    """Ingest files or directories into a collection, creating it if needed."""
    embed, identity = _embedder(state, embed_name)
    results: list[IngestResult] = []
    with Ingester(state.store, collection, embed, identity=identity, hybrid=hybrid,
                  chunk_size=state.cfg.chunk_size, chunk_overlap=state.cfg.chunk_overlap, save_every=50) as ing:
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                for r in ing.ingest_dir(p, glob=glob, kind=kind, tags=tags, force=force):
                    _report(r)
                    results.append(r)
            else:
                r = ing.ingest_file(p, source_id=source_id if len(paths) == 1 else None, url=url, kind=kind,
                                    title=title, tags=tags, force=force)
                _report(r)
                results.append(r)
    _summary(results)


@main.command("ingest-manifest")
@click.argument("collection")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False))
@click.option("--base-dir", type=click.Path(exists=True, file_okay=False), help="Where the ids resolve (default: next to the manifest).")
@click.option("--alias", "aliases", multiple=True, metavar="ID=PATH",
              help="Where an id lives when it is not <id>.md or <id>/, e.g. unsloth-amd-prs=upstream-prs. Repeatable.")
@click.option("--embed", "embed_name")
@click.option("--force", is_flag=True)
@click.option("--hybrid", is_flag=True)
@pass_state
@guarded
def ingest_manifest(state: State, collection: str, manifest: str, base_dir: str | None, aliases: tuple[str, ...],
                    embed_name: str | None, force: bool, hybrid: bool) -> None:
    """Ingest a brain's reference cache from its fetch_reference.py manifest."""
    if any("=" not in a for a in aliases):
        raise click.BadParameter("expected ID=PATH", param_hint="--alias")
    alias_map = dict(a.split("=", 1) for a in aliases)
    embed, identity = _embedder(state, embed_name)
    results: list[IngestResult] = []
    with Ingester(state.store, collection, embed, identity=identity, hybrid=hybrid,
                  chunk_size=state.cfg.chunk_size, chunk_overlap=state.cfg.chunk_overlap, save_every=50) as ing:
        for r in ing.ingest_manifest(manifest, base_dir=base_dir, aliases=alias_map, force=force):
            _report(r)
            results.append(r)
    _summary(results)


@main.command()
@click.argument("collection")
@click.argument("question")
@click.option("-k", "top_k", default=5, show_default=True)
@click.option("--embed", "embed_name")
@click.option("--json", "as_json", is_flag=True)
@pass_state
@guarded
def search(state: State, collection: str, question: str, top_k: int, embed_name: str | None, as_json: bool) -> None:
    """Semantic search; prints the best chunks with their source."""
    embed, identity = _embedder(state, embed_name)
    hits = _search(state.store, collection, question, embed=embed, k=top_k, identity=identity)
    if as_json:
        click.echo(json.dumps([h.__dict__ for h in hits], indent=1))
        return
    if not hits:
        click.echo("no hits")
    for h in hits:
        head = f"{h.score:.3f}  {h.source_id}" + (f"  {h.title}" if h.title else "")
        click.echo(head)
        if h.url:
            click.echo(f"       {h.url}")
        snippet = " ".join(h.text.split())
        click.echo(f"       {snippet[:300]}{'...' if len(snippet) > 300 else ''}")
        click.echo("")


@main.command()
@click.argument("collection")
@click.argument("question")
@click.option("-k", "top_k", default=5, show_default=True)
@click.option("--embed", "embed_name")
@click.option("--llm", "llm_name")
@pass_state
@guarded
def ask(state: State, collection: str, question: str, top_k: int, embed_name: str | None, llm_name: str | None) -> None:
    """Retrieve, then answer with the LLM endpoint."""
    embed, identity = _embedder(state, embed_name)
    llm = _llm(state, llm_name)
    answer = _ask(state.store, collection, question, embed=embed, llm=llm, k=top_k, identity=identity)
    click.echo(answer.text.strip())
    if answer.hits:
        click.echo("")
        click.echo("sources:")
        for h in answer.hits:
            click.echo(f"  {h.score:.3f}  {h.source_id}" + (f"  {h.url}" if h.url else ""))


@main.command()
@click.argument("collection")
@click.argument("question")
@click.option("--llm", "llm_name")
@pass_state
@guarded
def query(state: State, collection: str, question: str, llm_name: str | None) -> None:
    """Natural language over the documents table. Prints the Polars expression the
    model wrote, then its result."""
    try:
        res = _query(state.store, collection, question, llm=_llm(state, llm_name))
    except QueryError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"expression: {res.code}")
    click.echo(str(res.result))


@main.command()
@click.argument("collection")
@click.option("--embed", "embed_name", help="Endpoint to move the collection to.")
@pass_state
@guarded
def reindex(state: State, collection: str, embed_name: str | None) -> None:
    """Re-embed every document from stored text, e.g. to switch embedding endpoints."""
    embed, identity = _embedder(state, embed_name)
    ing = Ingester(state.store, collection, embed, identity=identity, create=False, save_every=50, allow_mismatch=True)
    results = list(ing.reindex())
    _summary(results)
    click.echo(f"embedder now {identity}")


@main.command()
@click.argument("collection")
@click.argument("source_id")
@pass_state
@guarded
def rm(state: State, collection: str, source_id: str) -> None:
    """Remove one document and its vectors. The blob stays (content-addressed)."""
    store = state.store
    coll = store.collection(collection)
    doc = coll.delete(source_id)
    if doc is None:
        raise click.ClickException(f"no document {source_id!r} in {collection!r}")
    if store.qdrant.collection_exists(coll.vector_collection):
        store.vector_store(collection).delete(doc.doc_id)
    coll.save()
    click.echo(f"removed {source_id}")


@main.command()
@click.argument("collection")
@click.option("--yes", is_flag=True, help="Skip the prompt.")
@pass_state
@guarded
def drop(state: State, collection: str, yes: bool) -> None:
    """Delete a whole collection: rows, blobs and vectors."""
    state.store.collection(collection)
    if not yes:
        click.confirm(f"drop collection {collection!r} and everything in it?", abort=True)
    state.store.drop(collection)
    click.echo(f"dropped {collection}")
