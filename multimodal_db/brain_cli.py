"""``mmdb brain``: open a vault read-only, describe it, export its graph, index it."""
from __future__ import annotations

import json
from pathlib import Path

import click

from .brain import Brain
from .ingest import IngestResult


def _table(rows: list[list], headers: list[str]) -> str:
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()  # noqa: E731
    return "\n".join([line(headers), line(["-" * w for w in widths]), *(line(r) for r in cells)])


@click.group()
def brain() -> None:
    """Open a brain (an Obsidian vault in the abstract-brain shape). Read-only."""


@brain.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--json", "as_json", is_flag=True)
def info(path: str, as_json: bool) -> None:
    """Layers, the ownership split, the link graph, the tags in use."""
    s = Brain.open(path).summary()
    if as_json:
        click.echo(json.dumps(s, indent=1))
        return
    for key in ("name", "prefix", "root", "contract"):
        click.echo(f"{key:<12} {s[key]}")
    click.echo(f"{'notes':<12} {s['notes']}  ({s['generated']} generated, {s['hand_written']} hand-written, "
               f"{s['with_thinking']} with thinking)")
    click.echo(f"{'links':<12} {s['links']}  ({s['unresolved_links']} unresolved, {s['orphans']} orphan notes)")
    click.echo("")
    click.echo(_table([[k, v["notes"], v["generated"], v["with_thinking"]] for k, v in s["by_layer"].items()],
                      ["layer", "notes", "generated", "thinking"]))
    click.echo("")
    click.echo("top tags  " + ", ".join(f"{t} {c}" for t, c in s["top_tags"].items()))


@brain.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--layer", "layers", multiple=True, help="Only these layers. Repeatable.")
@click.option("--thinking", is_flag=True, help="Only notes with a hand-written half.")
@click.option("--json", "as_json", is_flag=True)
def notes(path: str, layers: tuple[str, ...], thinking: bool, as_json: bool) -> None:
    """List notes with their layer, kind, state and link counts."""
    import polars as pl

    df = Brain.open(path).nodes()
    if layers:
        df = df.filter(pl.col("layer").is_in(list(layers)))
    if thinking:
        df = df.filter(pl.col("has_thinking"))
    df = df.sort(["layer", "name"])
    if as_json:
        click.echo(json.dumps(df.to_dicts(), default=str, indent=1))
        return
    rows = [[r["layer"], r["name"], r["kind"], r["state"] or "-", r["n_out"], r["n_in"],
             "yes" if r["has_thinking"] else "-"] for r in df.to_dicts()]
    click.echo(_table(rows, ["layer", "note", "kind", "state", "out", "in", "thinking"]) if rows else "no notes")


@brain.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.argument("note")
def links(path: str, note: str) -> None:
    """What a note links to and what links to it."""
    import polars as pl

    b = Brain.open(path)
    n = b.note(note)
    if n is None:
        raise click.ClickException(f"no note {note!r} in {b.name}")
    edges = b.edges()
    click.echo(f"{n.path}  [{n.layer}] {n.title}")
    click.echo(f"tags: {', '.join(n.tags) or '-'}")
    out = edges.filter(pl.col("src") == n.path)
    click.echo(f"\nout ({out.height}):")
    for r in out.to_dicts():
        click.echo(f"  {r['dst_name']}" + ("" if r["resolved"] else "  (unresolved)"))
    inc = edges.filter(pl.col("dst") == n.path)
    click.echo(f"\nin ({inc.height}):")
    for r in inc.to_dicts():
        click.echo(f"  {r['src_name']}")


@brain.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--out", type=click.Path(file_okay=False), help="Directory to write into (default: <store root>/graphs/<brain>).")
@click.pass_context
def graph(ctx: click.Context, path: str, out: str | None) -> None:
    """Write nodes.parquet, edges.parquet and a d3-shaped graph.json."""
    b = Brain.open(path)
    state = ctx.find_root().obj
    target = Path(out) if out else Path(state.cfg.root) / "graphs" / b.slug
    written = b.export_graph(target)
    for k, p in written.items():
        click.echo(f"{k:<6} {p}")


@brain.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--collection", help="Default: brain-<slug>.")
@click.option("--embed", "embed_name")
@click.option("--layer", "layers", multiple=True, help="Only these layers. Repeatable.")
@click.option("--no-prune", is_flag=True, help="Keep rows for notes that no longer exist.")
@click.pass_context
def load(ctx: click.Context, path: str, collection: str | None, embed_name: str | None,
         layers: tuple[str, ...], no_prune: bool) -> None:
    """Index every note into the store; then `mmdb search <collection> "..."`."""
    from .cli import _embedder

    state = ctx.find_root().obj
    b = Brain.open(path)
    embed, identity = _embedder(state, embed_name)
    counts: dict[str, int] = {}
    changed: list[IngestResult] = []
    for r in b.index(state.store, embed, identity=identity, collection=collection,
                     layers=set(layers) or None, prune=not no_prune,
                     chunk_size=state.cfg.chunk_size, chunk_overlap=state.cfg.chunk_overlap):
        counts[r.status] = counts.get(r.status, 0) + 1
        if r.status != "unchanged":
            changed.append(r)
    for r in changed[:40]:
        click.echo(f"{r.status:<10} {r.document.source_id}  ({r.document.n_chunks} chunks)")
    if len(changed) > 40:
        click.echo(f"... and {len(changed) - 40} more")
    click.echo(", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing to do")
    click.echo(f"collection {collection or 'brain-' + b.slug}")
