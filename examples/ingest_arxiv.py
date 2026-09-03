"""Fetch arXiv papers by id and ingest the PDFs.

    python examples/ingest_arxiv.py 2305.14314 2106.09685 --collection ml-papers --embed local

Needs the ``pdf`` extra for text out of the PDF and network for the fetch. Title, authors,
date and abstract come from the arXiv API and go into the row; the PDF is the blob. When
a PDF yields no text (scanned, or pypdf missing) the abstract is indexed instead so the
paper is still findable.
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from multimodal_db import Ingester, Store, embedder_identity, embedding_for, load_config

API = "https://export.arxiv.org/api/query?id_list={ids}&max_results={n}"
NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch_meta(ids: list[str]) -> dict[str, dict]:
    r = httpx.get(API.format(ids=",".join(ids), n=len(ids)), timeout=60, follow_redirects=True)
    r.raise_for_status()
    out: dict[str, dict] = {}
    for e in ET.fromstring(r.text).findall("a:entry", NS):
        raw_id = e.find("a:id", NS).text.rsplit("/", 1)[-1]
        aid = re.sub(r"v\d+$", "", raw_id)
        pdf = next((l.get("href") for l in e.findall("a:link", NS) if l.get("title") == "pdf"), None)
        out[aid] = {
            "title": " ".join(e.find("a:title", NS).text.split()),
            "abstract": " ".join(e.find("a:summary", NS).text.split()),
            "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
            "published": e.find("a:published", NS).text,
            "pdf": pdf or f"https://arxiv.org/pdf/{aid}",
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+", help="arXiv ids like 2305.14314")
    ap.add_argument("--collection", default="arxiv")
    ap.add_argument("--embed")
    ap.add_argument("--root")
    args = ap.parse_args()

    cfg = load_config(root=args.root)
    ep = cfg.endpoint(args.embed, "embed")
    embed = embedding_for(ep)
    identity = embedder_identity(ep, embed)
    meta = fetch_meta(args.ids)

    with Store(cfg.root) as store, Ingester(store, args.collection, embed, identity=identity,
                                            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap) as ing:
        for aid in args.ids:
            m = meta.get(aid)
            if m is None:
                print(f"skip       {aid}: not on arXiv")
                continue
            pdf = httpx.get(m["pdf"], timeout=120, follow_redirects=True).content
            common = dict(
                source_id=f"arxiv:{aid}",
                filename=f"{aid}.pdf",
                url=f"https://arxiv.org/abs/{aid}",
                kind="arxiv",
                title=m["title"],
                fetched_at=datetime.now(timezone.utc),
                tags=["arxiv"],
                metadata={"authors": m["authors"], "published": m["published"], "abstract": m["abstract"]},
            )
            r = ing.ingest_bytes(pdf, **common)
            if r.status != "unchanged" and r.document.n_chunks == 0:
                r = ing.ingest_bytes(pdf, text=m["abstract"], force=True, **common)
            print(f"{r.status:<10} {r.document.source_id}  {m['title'][:70]}  ({r.document.n_chunks} chunks)")


if __name__ == "__main__":
    main()
