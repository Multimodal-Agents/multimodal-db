import json

from click.testing import CliRunner

from multimodal_db.cli import main

CFG = """
[store]
root = "data"
chunk_size = 64
chunk_overlap = 8

[defaults]
embed = "fake"
llm = "fake"

[endpoints.fake]
kind = "fake"
"""


def run(runner, cfg, *args):
    return runner.invoke(main, ["--config", str(cfg), *args], catch_exceptions=False)


def test_end_to_end(tmp_path, docs_dir):
    cfg = tmp_path / "mmdb.toml"
    cfg.write_text(CFG)
    r = CliRunner()

    out = run(r, cfg, "init")
    assert out.exit_code == 0 and "store" in out.output and "embed    fake" in out.output
    out = run(r, cfg, "endpoints", "--probe")
    assert out.exit_code == 0 and "fake" in out.output and "ok" in out.output
    assert "no collections" in run(r, cfg, "collections").output

    out = run(r, cfg, "ingest", "docs", str(docs_dir), "--tag", "amd")
    assert out.exit_code == 0, out.output
    assert "5 added" in out.output
    assert "5 unchanged" in run(r, cfg, "ingest", "docs", str(docs_dir)).output
    assert "docs" in run(r, cfg, "collections").output

    out = run(r, cfg, "ls", "docs", "--tag", "amd")
    assert "gfx1151.md" in out.output and "notes/triton.txt" in out.output
    rows = json.loads(run(r, cfg, "ls", "docs", "--json").output)
    assert len(rows) == 5 and "text" not in rows[0]
    assert "fake:fake-bow:256" in run(r, cfg, "info", "docs").output
    assert "Strix Halo APU" in run(r, cfg, "show", "docs", "gfx1151.md", "--text").output

    out = run(r, cfg, "search", "docs", "Strix Halo gfx1151")
    assert out.output.splitlines()[0].endswith("gfx1151.md  Strix Halo")
    assert json.loads(run(r, cfg, "search", "docs", "Strix Halo gfx1151", "--json").output)[0]["source_id"] == "gfx1151.md"

    out = run(r, cfg, "ask", "docs", "what is gfx1151?")
    assert "(fake answer)" in out.output and "sources:" in out.output

    assert "5 reindexed" in run(r, cfg, "reindex", "docs").output
    assert "removed" in run(r, cfg, "rm", "docs", "logo.png").output
    assert run(r, cfg, "rm", "docs", "logo.png").exit_code != 0
    assert "dropped" in run(r, cfg, "drop", "docs", "--yes").output
    assert "no collections" in run(r, cfg, "collections").output
    assert (tmp_path / "data").is_dir()


def test_errors_are_one_liners(tmp_path):
    cfg = tmp_path / "mmdb.toml"
    cfg.write_text(CFG)
    r = CliRunner()
    out = r.invoke(main, ["--config", str(cfg), "ls", "nothere"])
    assert out.exit_code == 1 and "no collection 'nothere'" in out.output and "Traceback" not in out.output
    out = r.invoke(main, ["--config", str(cfg), "search", "x", "q", "--embed", "nope"])
    assert out.exit_code == 1 and "unknown endpoint" in out.output
