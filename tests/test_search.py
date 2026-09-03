import pytest

from multimodal_db.ingest import Ingester
from multimodal_db.search import QueryError, ask, query, search
from multimodal_db.store import EmbedderMismatch
from multimodal_db.testing import FakeEmbedding, FakeLLM


@pytest.fixture
def filled(store, embed, docs_dir):
    with Ingester(store, "docs", embed) as ing:
        list(ing.ingest_dir(docs_dir))
    return store


def test_search_ranks_the_right_source_first(filled, embed):
    hits = search(filled, "docs", "Strix Halo gfx1151 APU", embed=embed, k=3)
    assert hits and hits[0].source_id == "gfx1151.md" and hits[0].title == "Strix Halo" and hits[0].score > 0
    assert search(filled, "docs", "hipErrorNoBinaryForGpu code object", embed=embed, k=3)[0].source_id == "hip-errors.md"
    assert search(filled, "docs", "TRITON_OVERRIDE_ARCH", embed=embed, k=1)[0].source_id == "notes/triton.txt"


def test_search_empty_collection(store, embed):
    store.collection("empty", create=True)
    assert search(store, "empty", "anything", embed=embed) == []


def test_search_refuses_other_embedder(filled):
    with pytest.raises(EmbedderMismatch):
        search(filled, "docs", "x", embed=FakeEmbedding(dims=8))


def test_ask_grounds_the_prompt(filled, embed):
    llm = FakeLLM(responses=["gfx1151 is Strix Halo."])
    a = ask(filled, "docs", "what is gfx1151?", embed=embed, llm=llm, k=2)
    assert a.text == "gfx1151 is Strix Halo."
    assert a.hits and a.hits[0].source_id == "gfx1151.md"
    assert any("Strix Halo APU target" in p for p in llm.calls)


def test_query_runs_the_expression(filled):
    r = query(filled, "docs", "how many documents?", llm=FakeLLM(responses=["```python\ndf.height\n```"]))
    assert r.code == "df.height" and r.result == 5

    llm = FakeLLM(responses=['df.filter(pl.col("media_type") == "image").select("source_id")'])
    r = query(filled, "docs", "which are images?", llm=llm)
    assert r.result.to_series().to_list() == ["logo.png"]
    columns_block = llm.calls[0].split("Columns:")[1].split("First rows:")[0]
    assert "  text:" not in columns_block  # the bulk column stays out of the prompt

    with pytest.raises(QueryError):
        query(filled, "docs", "boom", llm=FakeLLM(responses=["__import__('os').getcwd()"]))
