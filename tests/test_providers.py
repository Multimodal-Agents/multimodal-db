from pathlib import Path

import pytest

from multimodal_db import providers
from multimodal_db.config import ConfigError, Endpoint, load_config, parse_config
from multimodal_db.providers import embedder_identity, embedding_for, llm_for, probe, resolve_model

TOML = """
[store]
root = "mystore"
chunk_size = 256
chunk_overlap = 32

[defaults]
embed = "local"
llm = "studio"

[endpoints.studio]
kind = "openai"
base_url = "http://127.0.0.1:8889/v1/"
api_key_env = "MMDB_TEST_TOKEN"
chat_model = "auto"
embed_model = "auto"

[endpoints.local]
kind = "huggingface"
embed_model = "unsloth/bge-small-en-v1.5"

[endpoints.fake]
kind = "fake"
"""


def test_parse(tmp_path, monkeypatch):
    monkeypatch.delenv("MMDB_EMBED", raising=False)
    p = tmp_path / "mmdb.toml"
    p.write_text(TOML)
    cfg = parse_config(p)
    assert cfg.root == (tmp_path / "mystore").resolve() and cfg.chunk_size == 256 and cfg.chunk_overlap == 32
    assert cfg.endpoints["studio"].base_url == "http://127.0.0.1:8889/v1"
    assert cfg.endpoint(None, "embed").name == "local" and cfg.endpoint(None, "llm").name == "studio"
    assert cfg.endpoint("studio", "embed").name == "studio"
    monkeypatch.setenv("MMDB_EMBED", "fake")
    assert cfg.endpoint(None, "embed").name == "fake"
    with pytest.raises(ConfigError, match="unknown endpoint"):
        cfg.endpoint("nope", "embed")
    assert cfg.endpoints["studio"].resolved_api_key() is None
    monkeypatch.setenv("MMDB_TEST_TOKEN", "tok")
    assert cfg.endpoints["studio"].resolved_api_key() == "tok"


def test_load_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MMDB_CONFIG", raising=False)
    monkeypatch.delenv("MMDB_ROOT", raising=False)
    cfg = load_config()
    assert cfg.path is None and cfg.root == Path("data") and cfg.default_embed == "local"
    assert cfg.endpoints["local"].kind == "huggingface"

    (tmp_path / "mmdb.toml").write_text(TOML)
    assert load_config().path == Path("mmdb.toml")
    monkeypatch.setenv("MMDB_ROOT", str(tmp_path / "env"))
    assert load_config().root == tmp_path / "env"
    assert load_config(root=tmp_path / "arg").root == tmp_path / "arg"
    with pytest.raises(ConfigError, match="not found"):
        load_config(path=tmp_path / "missing.toml")


def test_validation(tmp_path):
    with pytest.raises(ConfigError, match="kind"):
        Endpoint(name="x", kind="magic")
    with pytest.raises(ConfigError, match="base_url"):
        Endpoint(name="x", kind="openai")
    p = tmp_path / "bad.toml"
    p.write_text('[endpoints.a]\nkind = "fake"\nbogus = 1\n')
    with pytest.raises(ConfigError, match="bogus"):
        parse_config(p)
    p.write_text('[defaults]\nembed = "nope"\n[endpoints.a]\nkind = "fake"\n')
    with pytest.raises(ConfigError, match="nope"):
        parse_config(p)


def test_openai_kind_builds_llamaindex_objects(monkeypatch):
    ep = Endpoint(name="studio", kind="openai", base_url="http://127.0.0.1:8889/v1", chat_model="auto", embed_model="bge-m3")
    emb = embedding_for(ep)
    assert type(emb).__name__ == "OpenAILikeEmbedding" and emb.model_name == "bge-m3" and emb.api_base == ep.base_url

    monkeypatch.setattr(providers, "list_models", lambda e: ["qwen3-8b-gguf", "other"])
    llm = llm_for(ep)
    assert type(llm).__name__ == "OpenAILike" and llm.model == "qwen3-8b-gguf" and llm.api_base == ep.base_url

    monkeypatch.setattr(providers, "list_models", lambda e: [])
    with pytest.raises(ConfigError, match="no models"):
        resolve_model(ep, "llm")


def test_huggingface_has_no_llm():
    with pytest.raises(ConfigError, match="embeddings only"):
        llm_for(Endpoint(name="local", kind="huggingface", embed_model="m"))


def test_fake_identity_and_probe():
    ep = Endpoint(name="fake", kind="fake")
    emb = embedding_for(ep)
    assert embedder_identity(ep, emb) == "fake:fake-bow:256"
    assert probe(ep)["ok"] is True
    dead = Endpoint(name="dead", kind="openai", base_url="http://127.0.0.1:9", timeout=2)
    assert probe(dead)["ok"] is False
