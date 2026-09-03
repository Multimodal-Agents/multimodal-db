"""Ingest a brain's reference cache from its manifest.json, then search it.

    python examples/ingest_reference.py path/to/unsloth-amd/reference --embed local
    python examples/ingest_reference.py path/to/reference --query "hipErrorNoBinaryForGpu"

Read-only against the reference directory. Uses mmdb.toml when there is one, else the
built-in ``local`` endpoint. This is what ``mmdb ingest-manifest`` and ``mmdb search`` do,
written against the library.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_db import Ingester, Store, embedder_identity, embedding_for, load_config, search


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference_dir", help="directory holding manifest.json and the cached pages")
    ap.add_argument("--collection", default="amd-reference")
    ap.add_argument("--embed", help="endpoint name from mmdb.toml")
    ap.add_argument("--root", help="store root (default: mmdb.toml [store].root or ./data)")
    ap.add_argument("--query", default="which cards does ROCm support on Windows")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--alias", action="append", default=["unsloth-amd-prs=upstream-prs"], metavar="ID=PATH",
                    help="where a manifest id lives when it is not <id>.md or <id>/ (the gh-prs entry feeds upstream-prs/)")
    args = ap.parse_args()
    aliases = dict(a.split("=", 1) for a in args.alias)

    cfg = load_config(root=args.root)
    ep = cfg.endpoint(args.embed, "embed")
    embed = embedding_for(ep)
    identity = embedder_identity(ep, embed)
    manifest = Path(args.reference_dir) / "manifest.json"

    with Store(cfg.root) as store:
        counts: dict[str, int] = {}
        with Ingester(store, args.collection, embed, identity=identity, chunk_size=cfg.chunk_size,
                      chunk_overlap=cfg.chunk_overlap, save_every=50) as ing:
            for r in ing.ingest_manifest(manifest, aliases=aliases):
                counts[r.status] = counts.get(r.status, 0) + 1
                if r.status != "unchanged":
                    print(f"{r.status:<10} {r.document.source_id}  ({r.document.n_chunks} chunks)")
        print(", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        print(f"embedder {identity}")
        print()
        print(f"search: {args.query}")
        for hit in search(store, args.collection, args.query, embed=embed, k=args.k, identity=identity):
            print(f"  {hit.score:.3f}  {hit.source_id}  {hit.url or ''}")
            print(f"         {' '.join(hit.text.split())[:160]}")


if __name__ == "__main__":
    main()
