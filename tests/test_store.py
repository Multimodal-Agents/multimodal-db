from datetime import datetime, timezone

import polars as pl
import pytest

from multimodal_db.store import SCHEMA, Document, Store


def test_create_and_reopen(tmp_path):
    with Store(tmp_path / "s") as store:
        coll = store.collection("docs", create=True, chunk_size=128, chunk_overlap=16)
        assert coll.meta["chunk_size"] == 128 and coll.meta["embedder"] is None
        assert store.collections() == ["docs"]
        assert coll.frame().columns == [c for c in SCHEMA if c != "text"]
    with Store(tmp_path / "s") as store:
        assert store.collections() == ["docs"]
        assert store.collection("docs").count() == 0


def test_missing_collection_and_bad_name(store):
    with pytest.raises(KeyError):
        store.collection("nope")
    with pytest.raises(ValueError):
        store.collection("Bad Name", create=True)


def test_upsert_get_delete_roundtrip(store):
    coll = store.collection("docs", create=True)
    fetched = datetime(2026, 8, 26, 5, 38, tzinfo=timezone.utc)
    doc = Document(
        source_id="hip-error-codes",
        url="https://x/y",
        kind="web",
        title="HIP errors",
        fetched_at=fetched,
        content_hash="abc",
        size=3,
        text="hello",
        tags=["rocm", "hip"],
        metadata={"sha": "abc", "n": 1},
        n_chunks=2,
    )
    coll.upsert(doc)
    assert coll.count() == 1
    assert coll.get("hip-error-codes") == doc
    assert coll.get("missing") is None

    coll.upsert(Document(source_id="hip-error-codes", text="changed", content_hash="def"))
    assert coll.count() == 1 and coll.get("hip-error-codes").text == "changed"

    assert coll.delete("hip-error-codes").content_hash == "def"
    assert coll.delete("hip-error-codes") is None
    assert coll.count() == 0


def test_save_is_explicit_and_close_saves(tmp_path):
    root = tmp_path / "s"
    with Store(root) as store:
        coll = store.collection("docs", create=True)
        coll.upsert(Document(source_id="a", text="t"))
        assert coll.dirty and pl.read_parquet(coll.parquet).height == 0
        coll.save()
        assert not coll.dirty and pl.read_parquet(coll.parquet).height == 1
        coll.upsert(Document(source_id="b"))
    with Store(root) as store:
        assert store.collection("docs").count() == 2


def test_blobs_are_content_addressed(store):
    coll = store.collection("docs", create=True)
    a = coll.write_blob(b"same bytes", ".md")
    b = coll.write_blob(b"same bytes", ".md")
    c = coll.write_blob(b"other", ".md")
    assert a == b != c
    assert a.split("/")[0] == a.split("/")[1][:2] and a.endswith(".md")
    assert coll.read_blob(a) == b"same bytes"


def test_stats_and_drop(store):
    coll = store.collection("docs", create=True)
    coll.upsert(Document(source_id="a", kind="web", media_type="text", size=10, n_chunks=2))
    coll.upsert(Document(source_id="b", kind="file", media_type="image", size=5))
    s = coll.stats()
    assert (s["documents"], s["chunks"], s["bytes"]) == (2, 2, 15)
    assert s["by_kind"] == {"web": 1, "file": 1} and s["by_media_type"] == {"text": 1, "image": 1}
    store.drop("docs")
    assert store.collections() == [] and not coll.path.exists()
