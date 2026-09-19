import json

import polars as pl
import pytest

from multimodal_db.ingest import Ingester, extract_text, guess_mime, html_to_text, media_type_for
from multimodal_db.store import EmbedderMismatch
from multimodal_db.testing import FakeEmbedding


def test_mime_and_media():
    assert guess_mime("a.md") == "text/markdown" and media_type_for("text/markdown") == "text"
    assert media_type_for("application/pdf") == "document"
    assert guess_mime("paper.tex") == "text/x-tex" and media_type_for(guess_mime("refs.bib")) == "text"
    assert media_type_for("image/png") == "image" and media_type_for("audio/wav") == "audio"
    assert media_type_for("application/octet-stream") == "document"


def test_html_and_markdown_extraction():
    text, title = html_to_text(
        "<html><head><title> T </title><style>a{}</style></head>"
        "<body><h1>H</h1><p>one</p><script>x()</script><p>two</p></body></html>"
    )
    assert title == "T" and "x()" not in text and "a{}" not in text
    assert text.split() == ["H", "one", "two"]
    text, title = extract_text(b"# Heading\n\nbody\n", "text/markdown")
    assert title == "Heading" and text.startswith("# Heading")
    assert extract_text(b"\x89PNG", "image/png") == ("", None)


def test_ingest_bytes_lifecycle(store, embed):
    body = b"gfx1151 is Strix Halo. " * 5
    with Ingester(store, "docs", embed, chunk_size=64, chunk_overlap=8) as ing:
        r1 = ing.ingest_bytes(body, source_id="strix", filename="strix.md", url="https://a", kind="web", tags=["amd"])
        assert r1.status == "added" and r1.document.n_chunks >= 1 and r1.document.media_type == "text"
        assert store.vector_count("docs") == r1.document.n_chunks

        r2 = ing.ingest_bytes(body, source_id="strix", filename="strix.md")
        assert r2.status == "unchanged" and r2.document.doc_id == r1.document.doc_id

        changed = b"gfx1151 changed. " * 50
        r3 = ing.ingest_bytes(changed, source_id="strix", filename="strix.md", url="https://a", kind="web", tags=["amd"])
        assert r3.status == "updated" and r3.document.doc_id == r1.document.doc_id
        assert store.vector_count("docs") == r3.document.n_chunks  # old vectors gone
        assert ing.coll.count() == 1

    coll = store.collection("docs")
    assert coll.meta["embedder"] == "fakeembedding:fake-bow:256"
    assert coll.read_blob(r3.document.blob_path) == changed
    assert coll.get("strix").url == "https://a" and coll.get("strix").tags == ["amd"]


def test_text_override_for_media(store, embed):
    png = b"\x89PNG\r\n" + b"\0" * 16
    with Ingester(store, "docs", embed) as ing:
        r = ing.ingest_bytes(png, source_id="logo.png", filename="logo.png", text="the unsloth sloth logo")
        assert r.document.media_type == "image" and r.document.n_chunks == 1
        r2 = ing.ingest_bytes(png, source_id="raw.png", filename="raw.png")
        assert r2.document.n_chunks == 0 and r2.document.text == ""
        assert r.document.blob_path == r2.document.blob_path  # same bytes, one blob


def test_ingest_dir_ids_and_extraction(store, embed, docs_dir):
    with Ingester(store, "docs", embed, save_every=100) as ing:
        results = list(ing.ingest_dir(docs_dir))
    ids = sorted(r.document.source_id for r in results)
    assert ids == ["gfx1151.md", "hip-errors.md", "logo.png", "notes/triton.txt", "page.html"]
    by = {r.document.source_id: r.document for r in results}
    assert by["page.html"].title == "Compat matrix" and "var a" not in by["page.html"].text
    assert by["gfx1151.md"].title == "Strix Halo"
    assert by["logo.png"].media_type == "image" and by["logo.png"].n_chunks == 0
    assert by["notes/triton.txt"].fetched_at is not None
    assert store.collection("docs").count() == 5


def test_save_every_batches_writes(store, embed, docs_dir):
    ing = Ingester(store, "docs", embed, save_every=100)
    list(ing.ingest_dir(docs_dir))
    assert pl.read_parquet(ing.coll.parquet).height == 0
    ing.close()
    assert pl.read_parquet(ing.coll.parquet).height == 5


def test_manifest(store, embed, tmp_path):
    base = tmp_path / "reference"
    (base / "rocm-docs").mkdir(parents=True)
    (base / "upstream-prs").mkdir()
    (base / "rocm-docs" / "hip-install.md").write_text("# HIP install\n\ninstall steps", "utf-8")
    (base / "upstream-prs" / "pr-1.md").write_text("pr one", "utf-8")
    (base / "upstream-prs" / "pr-2.md").write_text("pr two", "utf-8")
    (base / "notes").mkdir()
    (base / "notes" / "a.md").write_text("note a", "utf-8")
    manifest = {
        "sources": [
            {"id": "hip-install", "kind": "web", "url": "https://rocm/install", "ok": True, "sha256": "x",
             "at": "2026-08-26T05:38:49.750757+00:00"},
            {"id": "unsloth-amd-prs", "kind": "gh-prs", "url": "https://github.com/unslothai/unsloth", "ok": True},
            {"id": "notes", "kind": "gh-file", "url": "https://notes", "ok": True},
            {"id": "gone", "kind": "web", "url": "https://gone", "ok": True},
            {"id": "failed", "kind": "web", "url": "https://failed", "ok": False},
        ]
    }
    (base / "manifest.json").write_text(json.dumps(manifest), "utf-8")

    with Ingester(store, "ref", embed) as ing:
        results = list(ing.ingest_manifest(base / "manifest.json", aliases={"unsloth-amd-prs": "upstream-prs"}))
    statuses = {r.document.source_id: r.status for r in results}
    assert statuses == {
        "hip-install": "added",
        "unsloth-amd-prs/pr-1.md": "added",
        "unsloth-amd-prs/pr-2.md": "added",
        "notes/a.md": "added",
        "gone": "missing",
    }
    coll = store.collection("ref")
    doc = coll.get("hip-install")
    assert doc.url == "https://rocm/install" and doc.kind == "web" and doc.fetched_at.year == 2026
    assert doc.metadata["manifest_sha256"] == "x" and doc.title == "HIP install"
    assert coll.get("unsloth-amd-prs/pr-2.md").metadata["manifest_id"] == "unsloth-amd-prs"
    assert coll.get("unsloth-amd-prs/pr-2.md").kind == "gh-prs"

    # without the alias the gh-prs id has nowhere to go
    with Ingester(store, "ref2", embed) as ing:
        statuses = {r.document.source_id: r.status for r in ing.ingest_manifest(base / "manifest.json")}
    assert statuses["unsloth-amd-prs"] == "missing"


def test_embedder_mismatch_and_reindex(store, docs_dir):
    with Ingester(store, "docs", FakeEmbedding(dims=32)) as ing:
        list(ing.ingest_dir(docs_dir))
    before = store.vector_count("docs")
    assert before > 0

    with pytest.raises(EmbedderMismatch, match="mmdb reindex docs"):
        Ingester(store, "docs", FakeEmbedding(dims=8))

    coll = store.collection("docs")
    assert coll.vector_collection == "docs__g1"
    ing = Ingester(store, "docs", FakeEmbedding(dims=8), allow_mismatch=True)
    results = list(ing.reindex())
    assert {r.status for r in results} == {"reindexed"} and len(results) == 5
    assert coll.meta["embedder"] == "fakeembedding:fake-bow:8" and coll.vector_collection == "docs__g2"
    assert store.vector_count("docs") == before
    assert not store.qdrant.collection_exists("docs__g1")
    assert store.qdrant.get_collection("docs__g2").config.params.vectors.size == 8
    assert {row["embedder"] for row in coll.frame().to_dicts()} == {"fakeembedding:fake-bow:8"}

    # a second reindex under the same embedder still gets a fresh generation
    results = list(Ingester(store, "docs", FakeEmbedding(dims=8), allow_mismatch=True).reindex())
    assert len(results) == 5 and coll.vector_collection == "docs__g3" and store.vector_count("docs") == before
