import json

import pytest
from click.testing import CliRunner

from multimodal_db.brain import Brain, split_frontmatter
from multimodal_db.cli import main
from multimodal_db.ingest import Ingester
from multimodal_db.search import search

GENERATED = """---
number: {n}
kind: {kind}
state: {state}
cluster: {cluster}
url: "https://example.com/{n}"
synced: 2026-09-03
tags:
  - test/{kind}
  - state/{state}
---

<!-- sync:begin -->
# {title}

**Cluster** [[Cluster A|A]] · see [[PR 2]] and [[Concept A#why]]
<!-- sync:end -->

{tail}"""


def make_vault(tmp_path, with_toml=False):
    v = tmp_path / "vault"
    for d in ("Open", "Archive", "Clusters", "Concepts", "Threads", "data", ".obsidian", ".trash", "tools"):
        (v / d).mkdir(parents=True)
    (v / "CLAUDE.md").write_text("# Test Brain\n\nProtocol.\n", "utf-8")
    (v / "Home.md").write_text("---\ntags:\n  - test/moc\n---\n\n# Test Brain\n\nStart at [[Board]] and [[Concept A]].\n", "utf-8")
    (v / "Board.md").write_text("---\nsynced: 2026-09-03\n---\n\n<!-- sync:begin -->\n# Board\n\n- [[Issue 1]]\n- [[PR 2]]\n<!-- sync:end -->\n", "utf-8")
    (v / "Open" / "Issue 1.md").write_text(GENERATED.format(
        n=1, kind="issue", state="open", cluster="A", title="Something breaks on gfx1151",
        tail="## Thinking\n\nreal thinking about strix halo here\n"), "utf-8")
    (v / "Open" / "PR 2.md").write_text(GENERATED.format(
        n=2, kind="pr", state="open", cluster="", title="Fix the thing", tail="## Thinking\n"), "utf-8")
    (v / "Archive" / "Issue 3.md").write_text(GENERATED.format(
        n=3, kind="issue", state="closed", cluster="A", title="Old report",
        tail="## Thinking\n\nduplicate of [[Issue 1]], see `[[Not A Link]]` and #state/closed inline\n"), "utf-8")
    (v / "Clusters" / "Cluster A.md").write_text(
        "---\ncluster: A\ntags:\n  - test/cluster\n---\n\n<!-- sync:begin -->\n# Cluster A\n\n- [[Issue 1]]\n<!-- sync:end -->\n\n## Thinking\n\nroot cause is the kernel\n", "utf-8")
    (v / "Concepts" / "Concept A.md").write_text(
        "---\ntags:\n  - test/concept\ncorrected: 2026-08-24\n---\n\n# Concept A\n\nThe durable idea about #kernel/timing and [[Issue 1]] and [[Missing Note]].\n\n```\n[[Fenced Link]]\n```\n", "utf-8")
    (v / "Threads" / "Thread 1.md").write_text(
        "---\ntags:\n  - test/thread\nstatus: open\n---\n\n# Should we\n\n## Decision\n\n_unset_\n", "utf-8")
    (v / "data" / "notes.md").write_text("# not a note\n", "utf-8")
    (v / ".obsidian" / "x.md").write_text("# not a note\n", "utf-8")
    (v / ".trash" / "old.md").write_text("# not a note\n", "utf-8")
    (v / "tools" / "README.md").write_text("# not a note\n", "utf-8")
    if with_toml:
        (v / "brain.toml").write_text(
            '[brain]\nname = "Toml Brain"\nprefix = "tb"\n\n[layers]\nOpen = { role = "items", generated = true }\n'
            'Archive = { role = "items", generated = true }\nClusters = { role = "clusters", generated = true }\n'
            'Concepts = { role = "concepts", generated = false }\nThreads = { role = "threads", generated = false }\n\n'
            '[sync]\ntail = "## Notes"\n', "utf-8")
    return v


def test_split_frontmatter_handles_dates_and_lists():
    fm, body = split_frontmatter("---\nsynced: 2026-09-03\ntags:\n  - a/b\n  - c\nurl: \"https://x\"\nempty:\n---\n\n# T\n")
    assert fm["synced"] == "2026-09-03" and fm["tags"] == ["a/b", "c"] and fm["url"] == "https://x" and fm["empty"] is None
    assert body.startswith("\n# T")
    assert split_frontmatter("# no frontmatter\n") == ({}, "# no frontmatter\n")


def test_open_inferred(tmp_path):
    b = Brain.open(make_vault(tmp_path))
    assert b.name == "Test Brain" and b.spec.source == "inferred" and b.prefix == "test"
    assert sorted(n.path for n in b.notes) == [
        "Archive/Issue 3.md", "Board.md", "CLAUDE.md", "Clusters/Cluster A.md", "Concepts/Concept A.md",
        "Home.md", "Open/Issue 1.md", "Open/PR 2.md", "Threads/Thread 1.md",
    ]
    issue = b.note("Issue 1")
    assert issue.layer == "Open" and issue.role == "items" and issue.kind == "issue" and issue.state == "open"
    assert issue.is_generated and issue.has_thinking and "strix halo" in issue.hand
    assert issue.title == "Something breaks on gfx1151" and issue.frontmatter["number"] == 1
    assert issue.links == ["Cluster A", "PR 2", "Concept A"]
    assert not b.note("PR 2").has_thinking
    concept = b.note("Concepts/Concept A")
    assert not concept.is_generated and concept.has_thinking and concept.role == "concepts"
    assert concept.links == ["Issue 1", "Missing Note"]  # the fenced one is not a link
    assert "kernel/timing" in concept.tags and "test/concept" in concept.tags
    archived = b.note("issue 3")
    assert archived.state == "closed" and "state/closed" in archived.tags and "Not A Link" not in archived.links


def test_open_with_toml(tmp_path):
    b = Brain.open(make_vault(tmp_path, with_toml=True))
    assert b.name == "Toml Brain" and b.prefix == "tb" and b.spec.source == "brain.toml" and b.spec.tail == "## Notes"
    # "## Thinking" is now just text under the custom tail rule, so a bare heading counts as thinking
    assert b.note("PR 2").has_thinking


def test_graph(tmp_path):
    b = Brain.open(make_vault(tmp_path))
    edges, nodes = b.edges(), b.nodes()
    # Home 2, Board 2, Issue 1 3, PR 2 3, Issue 3 4 (the inline-code one is not a link),
    # Cluster A 1, Concept A 2 (the fenced one is not a link), Thread 1 0, CLAUDE 0
    assert edges.height == 17 and int((~edges["resolved"]).sum()) == 1
    assert edges.filter(~edges["resolved"])["dst_name"].to_list() == ["Missing Note"]
    row = nodes.filter(nodes["name"] == "Issue 1").to_dicts()[0]
    assert row["n_out"] == 3 and row["n_in"] == 4  # Board, Issue 3, Cluster A, Concept A
    s = b.summary()
    assert s["notes"] == 9 and s["links"] == 17 and s["unresolved_links"] == 1
    assert s["by_layer"]["Open"] == {"notes": 2, "generated": 2, "with_thinking": 1}
    assert s["orphans"] == 2 and s["top_tags"]["state/open"] == 2  # Thread 1 and CLAUDE: no links either way
    out = b.export_graph(tmp_path / "g")
    g = json.loads(out["graph"].read_text("utf-8"))
    assert len(g["nodes"]) == 9 and len(g["links"]) == 16 and g["unresolved"] == [{"source": "Concepts/Concept A.md", "target": "Missing Note"}]
    assert out["nodes"].exists() and out["edges"].exists()


def test_index_search_touch_and_prune(store, embed, tmp_path):
    v = make_vault(tmp_path)
    b = Brain.open(v)
    results = list(b.index(store, embed, collection="brain"))
    assert {r.status for r in results} == {"added"} and len(results) == 9
    doc = store.collection("brain").get("Open/Issue 1")
    assert doc.kind == "issue" and doc.metadata["layer"] == "Open" and doc.metadata["links"] == ["Cluster A", "PR 2", "Concept A"]
    assert doc.metadata["frontmatter"]["number"] == 1 and "state/open" in doc.tags and doc.title == "Something breaks on gfx1151"
    assert store.collection("brain").read_blob(doc.blob_path).startswith(b"\n<!-- sync:begin -->")

    hits = search(store, "brain", "real thinking about strix halo", embed=embed, k=2)
    assert hits[0].source_id == "Open/Issue 1"

    # a sync that only bumps synced: re-embeds nothing
    p = v / "Open" / "Issue 1.md"
    p.write_text(p.read_text("utf-8").replace("synced: 2026-09-03", "synced: 2026-09-04"), "utf-8")
    assert {r.status for r in Brain.open(v).index(store, embed, collection="brain")} == {"unchanged"}

    # tags change but the body does not: the row is refreshed without new vectors
    before = store.vector_count("brain")
    p.write_text(p.read_text("utf-8").replace("  - state/open", "  - state/open\n  - label/important"), "utf-8")
    statuses = {r.document.source_id: r.status for r in Brain.open(v).index(store, embed, collection="brain")}
    assert statuses["Open/Issue 1"] == "touched" and store.vector_count("brain") == before
    assert "label/important" in store.collection("brain").get("Open/Issue 1").tags

    # a note moves Open -> Archive: new key, old one pruned
    (v / "Archive" / "Issue 1.md").write_bytes(p.read_bytes())
    p.unlink()
    statuses = {r.document.source_id: r.status for r in Brain.open(v).index(store, embed, collection="brain")}
    assert statuses["Archive/Issue 1"] == "added" and statuses["Open/Issue 1"] == "removed"
    assert store.collection("brain").get("Open/Issue 1") is None and store.collection("brain").count() == 9
    assert store.vector_count("brain") == before


def test_index_layers_only(store, embed, tmp_path):
    b = Brain.open(make_vault(tmp_path))
    results = list(b.index(store, embed, collection="c", layers={"Concepts", "Threads"}))
    assert sorted(r.document.source_id for r in results) == ["Concepts/Concept A", "Threads/Thread 1"]


def test_cli(tmp_path, docs_dir):
    v = make_vault(tmp_path)
    cfg = tmp_path / "mmdb.toml"
    cfg.write_text('[store]\nroot = "data"\n[defaults]\nembed = "fake"\n[endpoints.fake]\nkind = "fake"\n', "utf-8")
    r = CliRunner()
    out = r.invoke(main, ["--config", str(cfg), "brain", "info", str(v)], catch_exceptions=False)
    assert out.exit_code == 0 and "Test Brain" in out.output and "prefix        test" in out.output and "Open" in out.output
    out = r.invoke(main, ["--config", str(cfg), "brain", "notes", str(v), "--layer", "Open"], catch_exceptions=False)
    assert "Issue 1" in out.output and "Concept A" not in out.output
    out = r.invoke(main, ["--config", str(cfg), "brain", "links", str(v), "Issue 1"], catch_exceptions=False)
    assert "out (3)" in out.output and "in (4)" in out.output
    out = r.invoke(main, ["--config", str(cfg), "brain", "graph", str(v)], catch_exceptions=False)
    assert (tmp_path / "data" / "graphs" / "vault" / "graph.json").exists()
    out = r.invoke(main, ["--config", str(cfg), "brain", "load", str(v)], catch_exceptions=False)
    assert out.exit_code == 0 and "9 added" in out.output and "collection brain-vault" in out.output
    out = r.invoke(main, ["--config", str(cfg), "search", "brain-vault", "root cause kernel"], catch_exceptions=False)
    assert "Clusters/Cluster A" in out.output
