# multimodal-db

A small document store for brains. You point it at what a crawler fetched (ROCm docs,
arXiv PDFs, release notes, transcripts, images with captions) and it keeps the bytes, a
Polars index of where each thing came from, and embedded chunks you can search. Models
come from wherever you already run them: Unsloth Studio, a bare `llama-server`, vLLM,
LM Studio, Ollama, or in-process.

- **Polars** for the document table, one parquet per collection
- **Qdrant** in embedded mode for vectors, no server
- **LlamaIndex** for chunking and retrieval, with the LLM and embedder swapped in per
  endpoint
- **No LLM in the core.** Search needs an embedder. `ask` and `query` need a chat model
  and say so when there isn't one.

## Install

```bash
pip install -e ".[local]"          # in-process embeddings with unsloth/bge-small-en-v1.5
```

| extra | adds |
|---|---|
| `local` | `HuggingFaceEmbedding`, pulls torch |
| `ollama` | the Ollama LLM and embedding classes |
| `hybrid` | fastembed, for dense+sparse collections |
| `pdf` | pypdf, for text out of PDFs |
| `dev` | pytest |

Core alone is polars, qdrant-client, llama-index-core and the OpenAI-like classes.
Nothing heavy until you ask for it.

## Quick start

```bash
cp mmdb.example.toml mmdb.toml     # edit endpoints; mmdb.toml is gitignored
mmdb init
mmdb ingest rocm ./reference/rocm-docs --kind web --tag rocm
mmdb search rocm "which cards does ROCm support on Windows"
mmdb ask rocm "is gfx1151 supported on Windows?" --llm studio
mmdb query rocm "how many documents per kind?" --llm studio
```

The same from Python:

```python
from multimodal_db import Ingester, Store, embedding_for, embedder_identity, load_config, search

cfg = load_config()
ep = cfg.endpoint("local", "embed")
embed = embedding_for(ep)

with Store(cfg.root) as store:
    with Ingester(store, "rocm", embed, identity=embedder_identity(ep, embed)) as ing:
        for r in ing.ingest_dir("reference/rocm-docs", kind="web"):
            print(r.status, r.document.source_id, r.document.n_chunks)
    for hit in search(store, "rocm", "gfx1151 on Windows", embed=embed, k=5):
        print(f"{hit.score:.3f} {hit.source_id} {hit.url}")
```

## Endpoints

`mmdb.toml` names the places models are served from. Pick one per call with
`--embed` / `--llm`, per shell with `MMDB_EMBED` / `MMDB_LLM`, or once under
`[defaults]`.

| kind | LLM | embeddings | for |
|---|---|---|---|
| `openai` | `OpenAILike` | `OpenAILikeEmbedding` | Unsloth Studio, llama-server, vLLM, LM Studio, anything OpenAI-shaped |
| `huggingface` | none | `HuggingFaceEmbedding` | in-process, no server |
| `ollama` | `Ollama` | `OllamaEmbedding` | opt-in extra |
| `fake` | canned | hashed bag of words | tests and dry runs |

```toml
[endpoints.studio]
kind = "openai"
base_url = "http://127.0.0.1:8889/v1"
api_key_env = "UNSLOTH_STUDIO_TOKEN"
chat_model = "auto"          # GET /models, first entry
embed_model = "auto"
```

Studio serves `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` and
`/v1/models`. Its `/v1/embeddings` only answers while a GGUF model is loaded, so load an
embedding GGUF there (or run `llama-server --embedding` yourself) before pointing
`embed_model` at it. For the same model Studio uses internally without a server, use the
`huggingface` kind with `unsloth/bge-small-en-v1.5`.

**A collection is locked to one embedder.** At first ingest it records
`kind:model:dims`, and any later ingest or search under a different one is refused rather
than mixing vector spaces and quietly returning worse results. Move a collection with
`mmdb reindex <collection> --embed <name>`, which re-embeds from the stored text and
needs no source files.

## Commands

| command | does |
|---|---|
| `init` | create the store root, print what is configured |
| `endpoints [--probe]` | list endpoints; probe asks each served one what it lists |
| `collections` | collections with document and vector counts |
| `ingest C PATH... [--kind --tag --url --source-id --force --hybrid]` | files or directories; source id is the relative path |
| `ingest-manifest C MANIFEST` | a brain's reference cache from its `manifest.json` |
| `ls C [--kind --tag --json]` | the document table |
| `info C` | counts, bytes, the embedder it is locked to |
| `show C SOURCE_ID [--text]` | one row |
| `search C "q" [-k --json]` | best chunks with their source |
| `ask C "q" [--llm]` | retrieve, then answer from what was retrieved |
| `query C "q" [--llm]` | natural language over the document table; prints the Polars expression before running it |
| `reindex C --embed NAME` | move a collection to another embedder |
| `rm C SOURCE_ID` | drop one document and its vectors |
| `drop C [--yes]` | drop a collection |

## For brains

A brain is an Obsidian vault built to the [abstract-brain](https://github.com/LeoBorcherding/abstract-brain)
shape: generated notes with a sync-owned block and a hand-written `## Thinking` half,
hand-written Concepts and Threads, a reference cache with a manifest. `mmdb brain` opens one
read-only, from its `brain.toml` or by inferring the same defaults for the vaults that
predate it:

```bash
mmdb brain info "L:/AMD Brain/AMD Brain"     # layers, ownership split, link graph, tags in use
mmdb brain load "L:/AMD Brain/AMD Brain"     # every note into brain-amd-brain
mmdb search brain-amd-brain "why torch.compile gets disabled on a healthy ROCm host"
mmdb brain links "L:/AMD Brain/AMD Brain" "PR 7704"
mmdb brain graph "L:/AMD Brain/AMD Brain"    # nodes.parquet, edges.parquet, a d3-shaped graph.json
```

Notes are keyed by path and hashed on their body without frontmatter, so a sync that only
bumps `synced:` re-embeds nothing. Layer, tags, wikilinks, state and the rest of the
frontmatter ride along as metadata; a note that gained a tag is `touched` (row refreshed,
vectors kept), one that moved between layers is a new key and the old one is pruned. The
Python side is `Brain.open(path)` with `.notes`, `.nodes()`, `.edges()`, `.summary()` and
`.index(store, embed)`.

The raw reference pages a brain distils from are too big and too full of chrome to live in
the vault, so they sit in a cache next to it with a `manifest.json` of ids, URLs, hashes
and fetch times. This is the store for that cache too:

```bash
mmdb ingest-manifest amd-reference path/to/unsloth-amd/reference/manifest.json \
    --alias unsloth-amd-prs=upstream-prs --embed local
mmdb search amd-reference "hipErrorNoBinaryForGpu"
```

Each manifest id becomes a document keyed by that id, with the URL, kind and fetch time
carried through, so a note can cite `source_id` and the date and a later refresh can be
checked against the content hash. An id that is neither `<id>.md` nor a directory of
that name gets an `--alias` (the `gh-prs` entry above feeds `upstream-prs/`, and its
files ingest as `unsloth-amd-prs/pr-NNNN.md`). Nothing here writes to a vault. See
[examples/ingest_reference.py](examples/ingest_reference.py) for the same from Python and
[examples/ingest_arxiv.py](examples/ingest_arxiv.py) for papers by id.

## On disk

```
data/
  collections/<name>/
    documents.parquet     one row per document, extracted text included
    blobs/<hh>/<sha256>.<ext>   the original bytes, content-addressed
    meta.json             embedder identity, chunking, hybrid flag, vector generation
  qdrant/                 embedded Qdrant; <name>__g<N>, a new generation per reindex
```

The parquet is rewritten whole on save, which is fine into the tens of thousands of
documents. [ARCHITECTURE.md](ARCHITECTURE.md) has the data model and the reasons.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The suite uses the `fake` kind and refuses any non-loopback socket, so it runs with no
model, no server and no network.

## License

Apache 2.0. See [LICENSE](LICENSE).
