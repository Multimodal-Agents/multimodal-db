"""Open a brain read-only: an Obsidian vault built to the abstract-brain shape. Its notes, the
two ownership regions inside each generated one, the wikilink graph, and an index of it in
the store. The contract comes from the vault's brain.toml, or is inferred from the same
defaults for a vault that predates it. Nothing here writes into a vault."""
from __future__ import annotations

import json
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import polars as pl
import yaml

from .ingest import Ingester, IngestResult
from .store import Store

DEFAULT_BEGIN = "<!-- sync:begin -->"
DEFAULT_END = "<!-- sync:end -->"
DEFAULT_TAIL = "## Thinking"
SKIP_DIRS = {".obsidian", ".git", ".trash", "data", "tools", "__pycache__", "node_modules"}
# non-note files a vault may carry that have text worth indexing; images and the like are not
# copied into the store
FILE_EXTS = {".pdf", ".tex", ".bib", ".txt", ".rst", ".html", ".htm"}
KNOWN_LAYERS = {
    "Open": "items", "Archive": "items", "Clusters": "clusters", "Views": "views",
    "Concepts": "concepts", "Threads": "threads", "Reference": "reference", "Hardware": "hardware",
    "DAILY TODOS": "daily", "Discord": "field", "Field reports": "field", "People": "people",
    "Runners": "runners", "Sweeps": "sweeps", "Templates": "templates", "Assets": "assets",
}
GENERATED_LAYERS = {"Open", "Archive", "Clusters", "Views", "Hardware"}
# tag namespaces that are shared vocabulary, not the brain's own prefix
SHARED_NAMESPACES = {"state", "cluster", "ring", "repo", "label", "gfx", "arch", "gpu", "sm", "chip",
                     "person", "hw", "runner", "cap"}

_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_WIKILINK = re.compile(r"!?\[\[([^\]\|#]+)(?:#[^\]\|]*)?(?:\|[^\]]*)?\]\]")
_TAG = re.compile(r"(?<![\w/#])#((?=[\w/\-]*[A-Za-z])[\w/\-]+)")
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.M)
_FRONTMATTER = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.S)


@dataclass
class BrainSpec:
    name: str
    root: Path
    prefix: str | None = None
    begin: str = DEFAULT_BEGIN
    end: str = DEFAULT_END
    tail: str = DEFAULT_TAIL
    layers: dict[str, str] = field(default_factory=lambda: dict(KNOWN_LAYERS))
    generated: set[str] = field(default_factory=lambda: set(GENERATED_LAYERS))
    source: str = "inferred"


@dataclass
class Note:
    path: str
    name: str
    layer: str
    role: str
    title: str
    kind: str
    state: str | None
    frontmatter: dict[str, Any]
    tags: list[str]
    body: str
    generated: str | None
    hand: str
    has_thinking: bool
    links: list[str]
    mtime: datetime

    @property
    def is_generated(self) -> bool:
        return self.generated is not None

    @property
    def source_id(self) -> str:
        return self.path[:-3] if self.path.endswith(".md") else self.path


@dataclass
class VaultFile:
    """A PDF, a .tex source, or another text-bearing file living in a vault folder. Its
    source id keeps the extension, so it can never collide with a note."""
    path: str
    layer: str
    role: str
    file: Path
    mtime: datetime


def load_spec(root: Path) -> BrainSpec:
    toml_path = root / "brain.toml"
    if not toml_path.exists():
        return BrainSpec(name=_infer_name(root), root=root)
    raw = tomllib.loads(toml_path.read_text("utf-8"))
    brain, sync, layers = raw.get("brain", {}), raw.get("sync", {}), raw.get("layers", {})
    roles = {k: (v.get("role", "other") if isinstance(v, dict) else str(v)) for k, v in layers.items()}
    generated = {k for k, v in layers.items() if isinstance(v, dict) and v.get("generated") in (True, "half")}
    return BrainSpec(
        name=brain.get("name") or _infer_name(root),
        root=root,
        prefix=brain.get("prefix") or None,
        begin=sync.get("begin", DEFAULT_BEGIN),
        end=sync.get("end", DEFAULT_END),
        tail=sync.get("tail", DEFAULT_TAIL),
        layers=roles or dict(KNOWN_LAYERS),
        generated=generated or set(GENERATED_LAYERS),
        source="brain.toml",
    )


def _infer_name(root: Path) -> str:
    for candidate in ("CLAUDE.md", "README.md", "Home.md"):
        p = root / candidate
        if p.exists():
            m = _H1.search(p.read_text("utf-8", errors="replace"))
            if m:
                return m.group(1).strip()
    return root.name


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    return value


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    if not isinstance(fm, dict):
        fm = {}
    return {str(k): _normalize_value(v) for k, v in fm.items()}, text[m.end():]


def _fm_tags(fm: dict[str, Any]) -> list[str]:
    raw = fm.get("tags")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [t for t in re.split(r"[,\s]+", raw) if t]
    return [str(t).lstrip("#") for t in raw if t is not None]


def _link_target(raw: str) -> str:
    target = raw.strip()
    if "/" in target:
        target = target.rsplit("/", 1)[-1]
    return target[:-3] if target.lower().endswith(".md") else target


def parse_note(spec: BrainSpec, path: Path, rel: Path) -> Note:
    raw = path.read_text("utf-8", errors="replace")
    fm, body = split_frontmatter(raw)
    generated: str | None = None
    hand = body
    if spec.begin in body and spec.end in body:
        _, _, rest = body.partition(spec.begin)
        block, _, after = rest.partition(spec.end)
        generated, hand = block.strip("\n"), after
    stripped = _INLINE_CODE.sub("", _FENCE.sub("", body))
    links = list(dict.fromkeys(_link_target(m) for m in _WIKILINK.findall(stripped)))
    tags = list(dict.fromkeys(_fm_tags(fm) + _TAG.findall(stripped)))
    layer = rel.parts[0] if len(rel.parts) > 1 else "(root)"
    role = "root" if layer == "(root)" else spec.layers.get(layer, "other")
    hand_text = hand.strip()
    has_thinking = bool(hand_text) and hand_text != spec.tail and hand_text != spec.tail.rstrip("#").strip()
    if generated is None:
        has_thinking = bool(body.strip())
    m = _H1.search(body)
    kind = str(fm.get("kind") or role)
    state = fm.get("state")
    return Note(
        path=rel.as_posix(),
        name=path.stem,
        layer=layer,
        role=role,
        title=(m.group(1).strip() if m else path.stem),
        kind=kind,
        state=str(state) if state not in (None, "") else None,
        frontmatter=fm,
        tags=tags,
        body=body,
        generated=generated,
        hand=hand,
        has_thinking=has_thinking,
        links=links,
        mtime=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
    )


class Brain:
    def __init__(self, spec: BrainSpec):
        self.spec = spec
        self._notes: list[Note] | None = None
        self._by_name: dict[str, Note] | None = None
        self._by_path: dict[str, Note] | None = None

    @classmethod
    def open(cls, path: str | Path) -> "Brain":
        root = Path(path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"no such directory: {root}")
        return cls(load_spec(root))

    @property
    def root(self) -> Path:
        return self.spec.root

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")[:48] or "brain"

    @property
    def notes(self) -> list[Note]:
        if self._notes is None:
            self._notes = list(self._scan())
            self._by_path = {n.path: n for n in self._notes}
            self._by_name = {}
            for n in sorted(self._notes, key=lambda n: (len(Path(n.path).parts), n.path)):
                self._by_name.setdefault(n.name.lower(), n)
        return self._notes

    def _scan(self) -> Iterator[Note]:
        for path in sorted(self.root.rglob("*.md")):
            rel = path.relative_to(self.root)
            if self._skipped(rel):
                continue
            yield parse_note(self.spec, path, rel)

    def _skipped(self, rel: Path) -> bool:
        return any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]) or rel.name.startswith(".")

    @property
    def files(self) -> list[VaultFile]:
        """Non-note files in the vault's folders whose suffix is in ``FILE_EXTS``."""
        out: list[VaultFile] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in FILE_EXTS:
                continue
            rel = path.relative_to(self.root)
            if self._skipped(rel):
                continue
            layer = rel.parts[0] if len(rel.parts) > 1 else "(root)"
            role = self.spec.layers.get(layer, "other") if layer != "(root)" else "root"
            out.append(VaultFile(path=rel.as_posix(), layer=layer, role=role, file=path,
                                 mtime=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)))
        return out

    @property
    def prefix(self) -> str | None:
        """The brain's own tag namespace, from brain.toml or the tags in use."""
        if self.spec.prefix:
            return self.spec.prefix
        counts: Counter[str] = Counter()
        for n in self.notes:
            for t in n.tags:
                head = t.split("/", 1)[0]
                if "/" in t and head not in SHARED_NAMESPACES:
                    counts[head] += 1
        return counts.most_common(1)[0][0] if counts else None

    def note(self, name_or_path: str) -> Note | None:
        self.notes
        assert self._by_name is not None and self._by_path is not None
        key = name_or_path.replace("\\", "/")
        if key in self._by_path:
            return self._by_path[key]
        if key + ".md" in self._by_path:
            return self._by_path[key + ".md"]
        return self._by_name.get(_link_target(key).lower())

    def resolve(self, target: str) -> Note | None:
        self.notes
        assert self._by_name is not None
        return self._by_name.get(_link_target(target).lower())

    # ---- graph ------------------------------------------------------------------------------

    def edges(self) -> pl.DataFrame:
        rows = []
        for n in self.notes:
            for target in n.links:
                dst = self.resolve(target)
                rows.append({"src": n.path, "src_name": n.name, "dst_name": target,
                             "dst": dst.path if dst else None, "resolved": dst is not None})
        schema = {"src": pl.String, "src_name": pl.String, "dst_name": pl.String, "dst": pl.String, "resolved": pl.Boolean}
        return pl.DataFrame(rows, schema=schema)

    def nodes(self) -> pl.DataFrame:
        edges = self.edges()
        out_deg = dict(edges.group_by("src").len().iter_rows()) if edges.height else {}
        in_deg = dict(edges.filter(pl.col("resolved")).group_by("dst").len().iter_rows()) if edges.height else {}
        rows = [{
            "path": n.path, "name": n.name, "layer": n.layer, "role": n.role, "kind": n.kind, "state": n.state,
            "title": n.title, "generated": n.is_generated, "has_thinking": n.has_thinking,
            "n_out": int(out_deg.get(n.path, 0)), "n_in": int(in_deg.get(n.path, 0)),
            "tags": n.tags, "mtime": n.mtime,
        } for n in self.notes]
        schema = {"path": pl.String, "name": pl.String, "layer": pl.String, "role": pl.String, "kind": pl.String,
                  "state": pl.String, "title": pl.String, "generated": pl.Boolean, "has_thinking": pl.Boolean,
                  "n_out": pl.Int64, "n_in": pl.Int64, "tags": pl.List(pl.String), "mtime": pl.Datetime("us", "UTC")}
        return pl.DataFrame(rows, schema=schema)

    def tags(self) -> pl.DataFrame:
        counts = Counter(t for n in self.notes for t in n.tags)
        return pl.DataFrame({"tag": list(counts), "count": list(counts.values())}).sort("count", descending=True)

    def summary(self) -> dict[str, Any]:
        nodes, edges = self.nodes(), self.edges()
        by_layer = {}
        if nodes.height:
            for layer, total, generated, thinking in nodes.group_by("layer").agg(
                pl.len().alias("total"), pl.col("generated").sum(), pl.col("has_thinking").sum()
            ).sort("total", descending=True).iter_rows():
                by_layer[layer] = {"notes": int(total), "generated": int(generated), "with_thinking": int(thinking)}
        orphans = nodes.filter((pl.col("n_in") == 0) & (pl.col("n_out") == 0)).height if nodes.height else 0
        return {
            "name": self.name,
            "prefix": self.prefix,
            "root": str(self.root),
            "contract": self.spec.source,
            "notes": nodes.height,
            "generated": int(nodes["generated"].sum()) if nodes.height else 0,
            "hand_written": int((~nodes["generated"]).sum()) if nodes.height else 0,
            "with_thinking": int(nodes["has_thinking"].sum()) if nodes.height else 0,
            "links": edges.height,
            "unresolved_links": int((~edges["resolved"]).sum()) if edges.height else 0,
            "orphans": orphans,
            "by_layer": by_layer,
            "top_tags": dict(self.tags().head(12).iter_rows()) if self.notes else {},
        }

    def export_graph(self, out_dir: str | Path) -> dict[str, Path]:
        """nodes.parquet and edges.parquet for Polars, graph.json in the {nodes, links}
        shape d3-force and most graph viewers take."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        nodes, edges = self.nodes(), self.edges()
        nodes.write_parquet(out / "nodes.parquet")
        edges.write_parquet(out / "edges.parquet")
        graph = {
            "brain": self.name,
            "nodes": [{"id": r["path"], "name": r["name"], "layer": r["layer"], "kind": r["kind"], "state": r["state"],
                       "generated": r["generated"], "has_thinking": r["has_thinking"], "tags": r["tags"],
                       "in": r["n_in"], "out": r["n_out"]} for r in nodes.to_dicts()],
            "links": [{"source": r["src"], "target": r["dst"]} for r in edges.filter(pl.col("resolved")).to_dicts()],
            "unresolved": [{"source": r["src"], "target": r["dst_name"]}
                           for r in edges.filter(~pl.col("resolved")).to_dicts()],
        }
        (out / "graph.json").write_text(json.dumps(graph, indent=1, default=str), "utf-8")
        return {"nodes": out / "nodes.parquet", "edges": out / "edges.parquet", "graph": out / "graph.json"}

    # ---- indexing ---------------------------------------------------------------------------

    def index(
        self,
        store: Store,
        embed: Any,
        *,
        identity: str | None = None,
        collection: str | None = None,
        layers: set[str] | None = None,
        prune: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        save_every: int = 50,
    ) -> Iterator[IngestResult]:
        """Every note into a collection (default ``brain-<slug>``), keyed by its path. The
        hashed bytes are the body without frontmatter, so a sync that only bumps ``synced:``
        does not re-embed anything; a move between layers is a new key plus a prune. The
        vault's PDFs and other ``FILE_EXTS`` files ride along under their full relative
        path, so a brain's papers are searchable next to its notes."""
        name = collection or f"brain-{self.slug}"
        seen: set[str] = set()
        with Ingester(store, name, embed, identity=identity, chunk_size=chunk_size,
                      chunk_overlap=chunk_overlap, save_every=save_every) as ing:
            for n in self.notes:
                if layers and n.layer not in layers:
                    continue
                seen.add(n.source_id)
                meta = {
                    "brain": self.name, "layer": n.layer, "role": n.role, "state": n.state,
                    "links": n.links, "generated": n.is_generated, "has_thinking": n.has_thinking,
                    # synced: is the sync's own clock, bumped on every run; keeping it would
                    # mark every generated note touched daily for nothing
                    "frontmatter": {k: v for k, v in n.frontmatter.items() if k not in ("tags", "synced")},
                }
                yield ing.ingest_bytes(n.body.encode("utf-8"), source_id=n.source_id, filename=n.path,
                                       kind=n.kind, title=n.title, fetched_at=n.mtime, tags=n.tags,
                                       metadata=meta, text=n.body)
            for f in self.files:
                if layers and f.layer not in layers:
                    continue
                seen.add(f.path)
                yield ing.ingest_bytes(f.file.read_bytes(), source_id=f.path, filename=f.path, kind="file",
                                       fetched_at=f.mtime, tags=[f"{self.prefix}/{f.role}"] if self.prefix else (),
                                       metadata={"brain": self.name, "layer": f.layer, "role": f.role, "file": True})
            if prune:
                stale = set(ing.coll.frame()["source_id"].to_list()) - seen
                if layers:
                    stale = {s for s in stale if s.split("/", 1)[0] in layers or "/" not in s}
                for sid in sorted(stale):
                    yield ing.remove(sid)
