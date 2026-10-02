import json

import pytest
from click.testing import CliRunner

from multimodal_db.cli import main
from multimodal_db.ingest import Ingester
from multimodal_db.provenance import Ledger, LedgerCorrupt, Replay, canonical, changed, digest, fingerprint

H = {"source_id": "a", "url": None}
P = {"event": "added", "content_hash": "0" * 64, "size": 3}


def test_digest_ignores_key_and_parent_order():
    p1, p2 = "1" * 64, "2" * 64
    assert digest("a", H, P) == digest("a", dict(reversed(H.items())), dict(reversed(P.items())))
    assert digest("a", H, P, parents=[p1, p2]) == digest("a", H, P, parents=[p2, p1])
    assert digest("a", H, P) != digest("b", H, P)
    assert digest("a", H, P) != digest("a", H, P, prev=p1)
    assert canonical({"b": 1, "a": "é"}) == '{"a":"é","b":1}'.encode()


@pytest.mark.parametrize("bad", [{"x": 1.5}, {1: "x"}, {"x": object()}])
def test_non_canonical_values_rejected(bad):
    with pytest.raises(ValueError):
        canonical(bad)


def test_parents_must_be_digests():
    with pytest.raises(ValueError):
        digest("a", H, P, parents=["nope"])


def test_append_chain_and_reopen(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    r1 = led.append("a", H, P)
    r2 = led.append("a", H, P | {"event": "updated"})
    other = led.append("b", H, P)
    assert r1.prev is None and r2.prev == r1.digest and other.prev is None
    assert [r.seq for r in led] == [1, 2, 3]

    again = Ledger(path)
    assert len(again) == 3 and again.head("a").digest == r2.digest
    assert [r.digest for r in again.history("a")] == [r1.digest, r2.digest]
    assert again.verify() == 3
    receipt = r2.receipt()
    assert receipt["event"] == "updated" and "payload" not in receipt and "header" not in receipt


def test_edited_record_fails_closed(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    led.append("a", H, P)
    led.append("a", H, P | {"event": "updated"})
    raw = path.read_text("utf-8")
    path.write_text(raw.replace('"size":3', '"size":4', 1), "utf-8")
    with pytest.raises(LedgerCorrupt) as e:
        Ledger(path)
    assert e.value.line == 1 and "edited" in e.value.reason
    assert path.read_text("utf-8") != raw  # left as found, not repaired


def test_removed_line_breaks_the_chain(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    for event in ("added", "updated", "removed"):
        led.append("a", H, P | {"event": event})
    lines = path.read_text("utf-8").splitlines(keepends=True)
    path.write_text(lines[0] + lines[2], "utf-8")
    with pytest.raises(LedgerCorrupt):
        Ledger(path)


def test_torn_tail_is_dropped_but_a_whole_record_is_not(tmp_path):
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append("a", H, P)
    good = path.read_bytes()
    path.write_bytes(good + b'{"seq":2,"cha')
    assert len(Ledger(path)) == 1 and path.read_bytes() == good

    path.write_bytes(good.rstrip(b"\n"))
    with pytest.raises(LedgerCorrupt):
        Ledger(path)


def test_replay_rejected(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    rec = led.append("a", H, P)
    led._heads.pop("a")  # pretend the chain head was lost, so the same record would be rebuilt
    with pytest.raises(Replay):
        led.append("a", H, P)
    assert len(led) == 1 and rec.seq == 1


def test_fingerprint_catches_a_hand_edit(tmp_path):
    note = tmp_path / "note.md"
    note.write_bytes(b"# Note\n\nwritten by sync\n")
    fp = fingerprint(note.read_bytes())
    assert fp == fingerprint("# Note\n\nwritten by sync\n")
    assert not changed(note, fp)
    note.write_bytes(b"# Note\n\nwritten by sync\nand a line by hand\n")
    assert changed(note, fp)
    note.unlink()
    assert changed(note, fp)


def test_ingest_writes_the_ledger(store, embed):
    with Ingester(store, "docs", embed, chunk_size=64, chunk_overlap=8) as ing:
        ing.ingest_bytes(b"one " * 20, source_id="n", filename="n.md")
        ing.ingest_bytes(b"one " * 20, source_id="n", filename="n.md")  # unchanged: no record
        ing.ingest_bytes(b"two " * 20, source_id="n", filename="n.md")
        ing.remove("n")
    hist = store.collection("docs").history("n")
    assert [r.payload["event"] for r in hist] == ["added", "updated", "removed"]
    assert hist[1].prev == hist[0].digest and hist[2].prev == hist[1].digest
    assert hist[0].payload["content_hash"] != hist[1].payload["content_hash"]
    assert "text" not in hist[0].payload and hist[0].header["source_id"] == "n"


def test_baseline_for_documents_indexed_before_the_ledger(store, embed):
    with Ingester(store, "docs", embed, chunk_size=64, chunk_overlap=8) as ing:
        ing.ingest_bytes(b"old " * 20, source_id="n", filename="n.md")
    coll = store.collection("docs")
    coll.ledger.path.unlink()
    coll._ledger = None
    with Ingester(store, "docs", embed, chunk_size=64, chunk_overlap=8) as ing:
        assert ing.ingest_bytes(b"old " * 20, source_id="n", filename="n.md").status == "unchanged"
        ing.ingest_bytes(b"old " * 20, source_id="n", filename="n.md")
    assert [r.payload["event"] for r in coll.history("n")] == ["baseline"]


def test_cli_verify_and_history(tmp_path, embed):
    root = tmp_path / "data"
    from multimodal_db.store import Store

    with Store(root) as s, Ingester(s, "docs", embed, chunk_size=64, chunk_overlap=8) as ing:
        ing.ingest_bytes(b"one " * 20, source_id="n", filename="n.md")
        ing.ingest_bytes(b"two " * 20, source_id="n", filename="n.md")
    runner = CliRunner()
    r = runner.invoke(main, ["--root", str(root), "provenance", "verify", "docs"])
    assert r.exit_code == 0 and "2 records" in r.output
    r = runner.invoke(main, ["--root", str(root), "provenance", "history", "docs", "n", "--json"])
    assert r.exit_code == 0 and [x["event"] for x in json.loads(r.output)] == ["added", "updated"]

    ledger = root / "collections" / "docs" / "ledger.jsonl"
    ledger.write_text(ledger.read_text("utf-8").replace('"added"', '"updated"', 1), "utf-8")
    r = runner.invoke(main, ["--root", str(root), "provenance", "verify", "docs"])
    assert r.exit_code != 0 and "line 1" in r.output
