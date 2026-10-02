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
| `provenance verify C` | recompute every ledger digest; fails naming the first edited line |
| `provenance history C SOURCE_ID [--json]` | every add, update, reindex and removal of one document, as a hash chain |
| `serve [--port 8765 --stdio]` | one long-lived process: MCP for Claude at `/mcp`, JSON for the Obsidian plugin at `/api` |
| `brain list\|info\|notes\|links\|related\|graph\|load BRAIN` | a registered brain by name, or any vault by path; see below |

## For brains

A brain is an Obsidian vault built to the [abstract-brain](https://github.com/LeoBorcherding/abstract-brain)
shape: generated notes with a sync-owned block and a hand-written `## Thinking` half,
hand-written Concepts and Threads, a reference cache with a manifest. `mmdb brain` opens one
read-only, from its `brain.toml` or by inferring the same defaults for the vaults that
predate it:

```toml
[brains.amd]
path = "L:/AMD Brain/AMD Brain"      # collection brain-amd unless you say otherwise
[brains.nvidia]
path = "L:/nvidia-brain"
```

```bash
mmdb brain list                      # what is registered, what is indexed
mmdb brain info amd                  # layers, ownership split, link graph, tags in use
mmdb brain load amd                  # every note into brain-amd; rerun after a sync, unchanged notes cost nothing
mmdb search brain-amd "why torch.compile gets disabled on a healthy ROCm host"
mmdb brain related amd "PR 5303"     # nearest notes by vector, no query needed
mmdb brain links amd "PR 7704"
mmdb brain graph amd                 # nodes.parquet, edges.parquet, a d3-shaped graph.json
```

Any command also takes a vault path instead of a name. Each brain stays its own vault; the
registry is only names to paths.

Notes are keyed by path and hashed on their body without frontmatter, so a sync that only
bumps `synced:` re-embeds nothing. Layer, tags, wikilinks, state and the rest of the
frontmatter ride along as metadata; a note that gained a tag is `touched` (row refreshed,
vectors kept), one that moved between layers is a new key and the old one is pruned. PDFs,
`.tex`, `.bib` and other text-bearing files sitting in a vault folder are indexed alongside
the notes under their full relative path (`Reference/paper.pdf`), so a brain's papers are
searchable in the same collection. The Python side is `Brain.open(path)` with `.notes`,
`.files`, `.nodes()`, `.edges()`, `.summary()` and `.index(store, embed)`.

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

### Serving brains to Claude and to Obsidian

Embedded Qdrant locks its directory to one process, so the way many sessions and a plugin
share one store is one long-lived server:

```bash
pip install "multimodal-db[server]"
mmdb serve                            # http://127.0.0.1:8765, MCP at /mcp, JSON at /api
claude mcp add --transport http mmdb http://127.0.0.1:8765/mcp
```

Claude then has `list_brains`, `search_brain`, `related_notes`, `read_note`, `brain_info`,
`search_collection`, `ask`, `list_collections`, `load_brain` and `note_history`, and can work any registered
brain by name. The same process answers the [Obsidian plugin](obsidian-plugin/) at
`/api/brains/<name>/{search,related,graph,note}`: semantic search over the open vault, the
notes nearest to the one you are reading, and a graph with wikilinks and similarity edges
side by side. The plugin matches its vault to a brain by path, reads only, and never writes
into the vault. `--stdio` runs MCP over stdio instead for a single client.

## On disk

```
data/
  collections/<name>/
    documents.parquet     one row per document, extracted text included
    blobs/<hh>/<sha256>.<ext>   the original bytes, content-addressed
    meta.json             embedder identity, chunking, hybrid flag, vector generation
    ledger.jsonl          append-only provenance, hash-chained per document
  qdrant/                 embedded Qdrant; <name>__g<N>, a new generation per reindex
```

Every add, update, reindex and removal appends one record to `ledger.jsonl`: where the
document came from, its content hash and the digest of its previous record, SHA-256 over
canonical JSON. Opening the ledger recomputes every digest, so a hand-edited line fails
loudly instead of being trusted. No text is stored in it. `fingerprint()` and `changed()`
are the same idea for files a renderer writes: keep the fingerprint, and a mismatch later
is a hand edit to pull back.

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
