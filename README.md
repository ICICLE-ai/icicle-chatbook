# ICICLE Chatbook

An interactive [marimo](https://marimo.io/) notebook that turns the ICICLE AI Tapis services into a hands-on RAG (retrieval-augmented generation) playground. Paste any text, ingest it into the vector store, and chat against it — the notebook chains the embed, vector, and chat services behind a single Tapis access token.

**Tags:** AI4CI, Software

For guidance on what to include in Tutorials, How-To Guides, Explanation, and Reference, see [Diátaxis](https://diataxis.fr/).

### License

[![License](https://img.shields.io/badge/License-GPL%203.0-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)

## References

- [ICICLE AI Tapis portal](https://icicleai.tapis.io) — where you log in and grab an access token.
- [Embedding service (`icicleaiembedserver`)](https://github.com/ICICLE-ai/icicle-ai-embed-service) — Qwen3-Embedding behind a JWT-gated FastAPI.
- [Vector service (`icicleaivecserver`) API docs](https://github.com/ICICLE-ai/icicle-ai-vector-service) — Qdrant-backed store with cosine + MMR retrieval.
- [marimo documentation](https://docs.marimo.io/) — the reactive Python notebook used to host this playground.
- [uv documentation](https://docs.astral.sh/uv/) — the package/env manager used to bootstrap the project.

## Acknowledgements

<!-- Please include other funding sources above this line. -->

*National Science Foundation (NSF) funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)*

## Issue reporting

File bugs, ideas, or questions on the [GitHub issues page](https://github.com/thevyasamit/icicle-chatbook/issues).

---

# Tutorials

## Run the RAG playground end-to-end

This tutorial walks a first-time user from a clean checkout to chatting with their own document.

### Prerequisites

- macOS or Linux shell (the commands below are zsh/bash).
- [`uv`](https://docs.astral.sh/uv/) installed (`brew install uv` on macOS).
- A free account on the [ICICLE AI Tapis portal](https://icicleai.tapis.io) — TACC login, CILogon (university SSO), or self-signup all work.

### Steps

1. **Clone and enter the repo.**
   ```bash
   git clone https://github.com/thevyasamit/icicle-chatbook.git
   cd icicle-chatbook
   ```
2. **Sync the environment.** `uv` reads `pyproject.toml` + `uv.lock` and creates `.venv/` with `marimo` and `requests` pinned.
   ```bash
   uv sync
   ```
3. **Launch the notebook in app mode** (code hidden, chat-style UI):
   ```bash
   uv run marimo run notebooks/rag_chat_marimo.py
   ```
   Or in editor mode (code visible, hot-reload) while developing:
   ```bash
   uv run marimo edit notebooks/rag_chat_marimo.py
   ```
4. **Grab a Tapis access token** from [icicleai.tapis.io](https://icicleai.tapis.io) (click your username in the bottom-left → *Copy Access Token*), paste it into the token box, and click **🔐 Validate token**.
5. **Ingest a document** — paste any text into the textarea, then click **🚀 Ingest into vector store**.
6. **Ask questions** in the chat panel at the bottom. Each message embeds the question, retrieves the top-K nearest chunks, stitches them into a grounded prompt, and sends it to the chat model.

### End result

A working in-browser RAG demo backed by your own document. The notebook surfaces the retrieved chunks under each answer so you can see exactly what the model was given.

---

# How-To Guides

## Get a Tapis access token

1. Visit [icicleai.tapis.io](https://icicleai.tapis.io) and sign in (TACC account, fresh signup, or CILogon).
2. Click your **username** in the bottom-left corner.
3. Choose **Copy Access Token** and paste the JWT into the notebook.

![Where to copy your Tapis access token in the ICICLE AI portal](assets/access_token_ss.png)

> ⏰ Tokens expire after ~4 hours. If you start seeing `401 Token expired`, refresh the token from the Tapis UI and paste it again.

## Tune ingestion behaviour

Open the *⚙️ Ingestion settings* accordion in the notebook:

- **Collection / Topic / Source** — namespacing inside the vector store. Use a distinct collection per project so retrieval doesn't bleed across documents.
- **Chat model** — switch between `llama4-17b`, `llama-3-70b`, and `mistral-7b`.
- **Top-K retrieval** — how many chunks to pull back per question. Raise it for broader context; lower it to keep prompts tight.
- **Max chunk tokens / Chunk overlap tokens** — chunking budget. Larger chunks preserve more context per vector; overlap reduces "split at a bad spot" misses.

## Preload a token via environment variable

Skip the paste step by exporting `TAPIS_TOKEN` before launching marimo — the token input prefills from `os.environ["TAPIS_TOKEN"]`.

```bash
export TAPIS_TOKEN="eyJ..."
uv run marimo run notebooks/rag_chat_marimo.py
```

## Reset the local environment

If dependencies get out of sync or you want a clean rebuild:

```bash
rm -rf .venv uv.lock
uv sync
```

---

# Explanation

## What the playground does

The notebook is a thin client over three ICICLE Tapis services, glued together behind one access token. Each chat turn runs the full RAG loop:

| Step | Service | Endpoint | What happens |
| --- | --- | --- | --- |
| **1. Embed** | `icicleaiembedserver` | `POST /v1/embed` | Text → 1024-dim normalized vector (Qwen3-Embedding via `llama-cpp-python`). |
| **2. Store / retrieve** | `icicleaivecserver` | `POST /v1/embeddings`, `POST /v1/retrieve` | FastAPI + Qdrant; cosine similarity with MMR reranking. |
| **3. Chat** | `tapisagent` | `POST /chat` | Generates the final answer from the retrieved chunks. |

Every request carries the same `X-Tapis-Token` (sent as both header and cookie), so authenticating once unlocks the whole pipeline.

## Why marimo?

marimo gives a reactive, code-first notebook with first-class UI widgets (`mo.ui.text`, `mo.ui.chat`, `mo.ui.run_button`) and an "app mode" that hides cells — useful for handing the notebook to non-developers without exposing the implementation. Reactivity also means the validation, ingestion, and chat cells re-evaluate cleanly whenever the token or ingest state changes.

## Design choices worth knowing

- **Validate before ingest.** The token cell hits `/v1/model` once and only unlocks downstream cells on a 200. This catches expired or wrong-tenant tokens before any embedding API spend.
- **Token-budget chunking.** A naive word-split with configurable max/overlap. Good enough for demo content; swap in `tiktoken` or a recursive splitter for production-grade ingestion.
- **Source metadata is stored alongside vectors.** `doc_id`, `chunk_index`, `chunk_count`, and a free-form `source` label travel with each vector so retrieval results stay traceable.
- **Retrieved chunks are echoed under every answer** in a collapsible `<details>` block — the demo prioritizes legibility/auditability over a polished chat surface.

## Project layout

```
icicle-chatbook/
├── assets/                  # Images referenced by the notebook (logo, screenshot)
├── notebooks/
│   └── rag_chat_marimo.py   # The marimo notebook
├── pyproject.toml           # uv-managed project metadata + deps
├── uv.lock                  # Pinned dependency lockfile
└── README.md
```
