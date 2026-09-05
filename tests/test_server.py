import anyio
import pytest

pytest.importorskip("mcp")
from starlette.testclient import TestClient  # noqa: E402

from multimodal_db.config import load_config  # noqa: E402
from multimodal_db.search import similar  # noqa: E402
from multimodal_db.server import Service, build_app, build_server  # noqa: E402
from tests.test_brain import make_vault  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    vault = make_vault(tmp_path)
    p = tmp_path / "mmdb.toml"
    p.write_text(
        '[store]\nroot = "data"\nchunk_size = 64\nchunk_overlap = 8\n\n[defaults]\nembed = "fake"\nllm = "fake"\n\n'
        '[endpoints.fake]\nkind = "fake"\n\n'
        f'[brains.test]\npath = {str(vault)!r}\n\n[brains.ghost]\npath = "nowhere"\n',
        "utf-8",
    )
    return load_config(p)


def test_config_registers_brains(cfg, tmp_path):
    assert set(cfg.brains) == {"test", "ghost"}
    assert cfg.brains["test"].collection == "brain-test" and cfg.brains["test"].path == (tmp_path / "vault").resolve()
    assert cfg.resolve_brain("test").name == "test"
    assert cfg.resolve_brain(str(tmp_path / "vault")).name == "test"  # a registered path resolves to its name
    other = tmp_path / "other vault"
    other.mkdir()
    assert cfg.resolve_brain(str(other)).collection == "brain-other-vault"
    with pytest.raises(Exception, match="no brain"):
        cfg.resolve_brain("nope")


def test_service_end_to_end(cfg):
    svc = Service(cfg)
    try:
        brains = {b["name"]: b for b in svc.brains()}
        assert brains["test"]["exists"] and brains["test"]["docs"] == 0 and not brains["ghost"]["exists"]

        loaded = svc.load("test")
        assert loaded["counts"] == {"added": 9} and loaded["docs"] == 9 and loaded["vectors"] > 0
        assert svc.brain_info("test")["notes"] == 9 and svc.brain_info("test")["docs"] == 9

        hits = svc.search_brain("test", "real thinking about strix halo", k=3)
        assert hits[0]["id"] == "Open/Issue 1" and hits[0]["layer"] == "Open"

        rel = svc.related("test", "Issue 1", k=3)
        assert rel and all(r["id"] != "Open/Issue 1" for r in rel) and "layer" in rel[0]
        assert [r.source_id for r in similar(svc.store, "brain-test", "Open/Issue 1", k=3)] == [r["id"] for r in rel]

        note = svc.note("test", "Concept A")
        assert note["id"] == "Concepts/Concept A" and note["links"] == ["Issue 1", "Missing Note"] and "durable" in note["body"]

        g = svc.graph("test", k=3, min_score=0.0)
        assert g["indexed"] and len(g["nodes"]) == 9
        kinds = {l["kind"] for l in g["links"]}
        assert kinds == {"wikilink", "similar"}
        assert all("/" not in n["id"] or not n["id"].endswith(".md") for n in g["nodes"])
        assert svc.graph("test", k=3, min_score=0.0) is g  # cached until the vault or the index changes

        m = svc.map("test")
        assert m["method"] in ("tsne", "pca") and len(m["points"]) == 9
        assert all(-1 <= p["x"] <= 1 and -1 <= p["y"] <= 1 for p in m["points"])
        assert {p["id"] for p in m["points"]} == {n["id"] for n in g["nodes"]}
        assert svc.map("test") is m
        node = next(n for n in g["nodes"] if n["id"] == "Open/Issue 1")
        assert "created" in node and "updated" in node and node["mtime"] > 1e9

        assert svc.ask("test", "what is gfx1151?", k=2)["answer"] == "(fake answer)"
        with pytest.raises(KeyError):
            svc.note("test", "Nope")
    finally:
        svc.close()


def test_http_routes(cfg):
    app = build_app(cfg)
    with TestClient(app) as c:
        h = c.get("/api/health").json()
        assert h["ok"] and {b["name"] for b in h["brains"]} == {"test", "ghost"}
        assert c.post("/api/brains/test/load").json()["docs"] == 9
        assert c.get("/api/brains/test").json()["notes"] == 9
        hits = c.get("/api/brains/test/search", params={"q": "strix halo thinking", "k": 2}).json()
        assert hits[0]["id"] == "Open/Issue 1"
        rel = c.get("/api/brains/test/related", params={"note": "Issue 1", "k": 2}).json()
        assert len(rel) == 2
        g = c.get("/api/brains/test/graph", params={"k": 2, "min": 0}).json()
        assert len(g["nodes"]) == 9 and any(l["kind"] == "similar" for l in g["links"])
        g2 = c.get("/api/brains/test/graph", params={"similar": 0}).json()
        assert all(l["kind"] == "wikilink" for l in g2["links"])
        assert c.get("/api/brains/test/note", params={"path": "Open/Issue 1"}).json()["layer"] == "Open"
        assert len(c.get("/api/brains/test/map").json()["points"]) == 9
        assert c.get("/api/collections").json()[0]["name"] == "brain-test"
        assert c.get("/api/search", params={"collection": "brain-test", "q": "kernel root cause"}).json()[0]["id"] == "Clusters/Cluster A"
        assert c.get("/api/brains/nope").status_code == 400
        assert c.get("/api/brains/test/note", params={"path": "Nope"}).status_code == 404
        assert c.get("/api/brains/ghost").status_code == 400
    app.state.service.close()


def test_mcp_tools_are_registered(cfg):
    svc = Service(cfg)
    try:
        server = build_server(svc)
        tools = anyio.run(server.list_tools)
        names = {t.name for t in tools}
        assert {"list_brains", "brain_info", "search_brain", "related_notes", "read_note", "search_collection",
                "ask", "list_collections", "load_brain"} <= names
        result = anyio.run(server.call_tool, "list_brains", {})
        text = str(result)
        assert "test" in text and "ghost" in text
    finally:
        svc.close()
