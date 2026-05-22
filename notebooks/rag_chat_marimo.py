"""ICICLE RAG playground — interactive marimo notebook.

Run as a chat-style app (no code visible):
    uv run marimo run notebooks/rag_chat_marimo.py

Run in editor mode (code visible, hot-reload):
    uv run marimo edit notebooks/rag_chat_marimo.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", app_title="ICICLE AI Chatbook")


@app.cell(hide_code=True)
def _imports():
    import json
    import os
    import uuid
    from pathlib import Path
    from typing import Any

    import marimo as mo
    import requests

    CHAT_BASE_URL = "https://tapisagent.pods.tacc.tapis.io"
    EMBED_BASE_URL = "https://icicleaiembedserver.pods.icicleai.tapis.io"
    VECTOR_BASE_URL = "https://icicleaivecserver.pods.icicleai.tapis.io"
    PORTAL_URL = "https://icicleai.tapis.io"

    # Shared Qdrant collection. The vector service already isolates rows per
    # user via the token's subject, so a fixed collection is safe — you only
    # ever see your own embeddings. Use `topic` (set in the config panel) to
    # organize different documents within your private slice of the collection.
    COLLECTION = "icicle-demo-collection"
    return (
        Any,
        CHAT_BASE_URL,
        COLLECTION,
        EMBED_BASE_URL,
        PORTAL_URL,
        Path,
        VECTOR_BASE_URL,
        mo,
        os,
        requests,
        uuid,
    )


@app.cell(hide_code=True)
def _title(CHAT_BASE_URL, EMBED_BASE_URL, Path, VECTOR_BASE_URL, mo):
    logo_path = Path(__file__).parent.parent / "assets" / "ICICLE_logo.jpg"

    # Header row: logo + page title, side by side. Only these two items go
    # in the hstack so the title doesn't get squashed.
    if logo_path.exists():
        header = mo.hstack(
            [
                mo.image(src=str(logo_path), width=140),
                mo.md("# ICICLE AI Chatbook"),
            ],
            justify="start",
            align="center",
            gap=2,
        )
    else:
        header = mo.md("# ICICLE AI Chatbook")

    # Full-width body below the header. Markdown gets to flow naturally — no hstack.
    body = mo.md(
        f"""
        Paste a document. Ask questions. Get grounded answers.

        Behind the scenes this notebook chains three **ICICLE AI Services** —
        general-purpose APIs you can build your own apps on, not just this demo:

        | Step | Service | What it does | Links |
        | --- | --- | --- | --- |
        | **1. Embed** | `icicleaiembedserver` | Qwen3 text embeddings → 1024-dim vectors | [OpenAPI docs]({EMBED_BASE_URL}/docs) |
        | **2. Store / retrieve** | `icicleaivecserver` | Qdrant-backed vector store + retrieval | [OpenAPI docs]({VECTOR_BASE_URL}/docs) |
        | **3. Chat** | `tapisagent` | Generates answers from the retrieved chunks | [Service]({CHAT_BASE_URL}) |

        All three live behind the same `X-Tapis-Token`. Get yours below 👇
        """
    )

    mo.vstack([header, body], gap=1)
    return


@app.cell(hide_code=True)
def _token_help(PORTAL_URL, Path, mo):
    image_path = Path(__file__).parent.parent / "assets" / "access_token_ss.png"
    if image_path.exists():
        screenshot = mo.image(src=str(image_path), width=480)
    else:
        screenshot = mo.callout(
            mo.md(
                "_(Screenshot of the bottom-left **Copy Access Token** menu would appear here. "
                "Drop an `access_token_ss.png` into `assets/` to enable it.)_"
            ),
            kind="neutral",
        )

    mo.accordion(
        {
            "🔑 How to get your Tapis access token (click to expand)": mo.vstack(
                [
                    mo.md(
                        f"""
                        1. Go to the **ICICLE AI Tapis UI** → [{PORTAL_URL}]({PORTAL_URL})
                        2. **Log in.** You have three ways to authenticate:
                           - Log in if you already have a TACC account
                           - **Sign up** for a TACC account (free, takes a minute)
                           - Log in with **CILogon** using your university account
                        3. Once logged in, click your **username** in the bottom-left corner.
                        4. Select **Copy Access Token**.
                        5. Paste the JWT into the box below.

                        > ⏰ Tokens expire after ~4 hours. If you start seeing `401 Token expired`,
                        > refresh the token from the Tapis UI and paste it again.
                        """
                    ),
                    screenshot,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _token_input(mo, os):
    token_input = mo.ui.text(
        value=os.environ.get("TAPIS_TOKEN", ""),
        placeholder="Paste your X-Tapis-Token here (eyJ...)",
        full_width=True,
        kind="password",
        label="**X-Tapis-Token**",
    )
    validate_button = mo.ui.run_button(
        label="🔐 Validate token",
        kind="info",
        tooltip="Pings the embed service /v1/model endpoint to confirm the token works.",
    )
    mo.vstack([token_input, validate_button])
    return token_input, validate_button


@app.cell(hide_code=True)
def _token_status(EMBED_BASE_URL, mo, requests, token_input, validate_button):
    raw = token_input.value.strip()
    token = None  # default — downstream stays locked unless we set this

    if not validate_button.value:
        # User hasn't clicked Validate yet.
        if raw:
            _status = mo.callout(
                "👆 Token pasted. Click **🔐 Validate token** above to verify it before continuing.",
                kind="neutral",
            )
        else:
            _status = mo.callout(
                "⏳ Paste your token above, then click **🔐 Validate token**.",
                kind="warn",
            )
    elif not raw:
        _status = mo.callout("❌ No token to validate.", kind="danger")
    elif not raw.startswith("eyJ") or raw.count(".") != 2:
        _status = mo.callout(
            "❌ That doesn't look like a JWT. A Tapis access token starts with `eyJ` "
            "and has two dots. Get a fresh one from the Tapis UI and click Validate again.",
            kind="danger",
        )
    else:
        # Hit the embed service /v1/model — auth-required, cheap, fast. Returns 200
        # only if the token is well-formed, signed, unexpired, and from the icicleai tenant.
        try:
            resp = requests.get(
                f"{EMBED_BASE_URL}/v1/model",
                headers={"X-Tapis-Token": raw},
                cookies={"X-Tapis-Token": raw},
                timeout=10,
            )
            if resp.status_code == 200:
                model_info = resp.json()
                _status = mo.callout(
                    mo.md(
                        f"✅ **Token validated.** "
                        f"Model: `{model_info.get('model', '?')}` · "
                        f"dim: `{model_info.get('dim', '?')}` · "
                        f"n_ctx: `{model_info.get('n_ctx', '?')}`. "
                        f"You're good to ingest below."
                    ),
                    kind="success",
                )
                token = raw
            elif resp.status_code == 401:
                _status = mo.callout(
                    "❌ **Token rejected (401).** Likely expired — Tapis access tokens last "
                    "~4 hours. Get a fresh one from the Tapis UI and click Validate again.",
                    kind="danger",
                )
            elif resp.status_code == 403:
                _status = mo.callout(
                    "❌ **Wrong tenant (403).** This service only accepts tokens from the "
                    "`icicleai` tenant. Make sure you're logged into the right Tapis UI.",
                    kind="danger",
                )
            else:
                _status = mo.callout(
                    f"❌ Validation failed [{resp.status_code}]: {resp.text[:200]}",
                    kind="danger",
                )
        except requests.RequestException as exc:
            _status = mo.callout(
                f"❌ **Network error** while validating: `{exc}`. "
                "Is the embed service reachable from here?",
                kind="danger",
            )

    _status
    return (token,)


@app.cell(hide_code=True)
def _config_form(COLLECTION, mo):
    topic = mo.ui.text(
        value="general",
        label="**Topic** (a label that groups this document — e.g. `paper-2024`, `notes`)",
        full_width=True,
    )
    chat_model = mo.ui.dropdown(
        options=["llama4-17b", "llama-3-70b", "mistral-7b"],
        value="llama4-17b",
        label="Chat model",
    )
    top_k = mo.ui.slider(start=1, stop=10, step=1, value=4, label="Top-K retrieval")
    max_chunk_tokens = mo.ui.slider(
        start=100, stop=600, step=50, value=300, label="Max chunk tokens"
    )
    overlap_tokens = mo.ui.slider(
        start=0, stop=150, step=10, value=50, label="Chunk overlap tokens"
    )

    isolation_note = mo.callout(
        mo.md(
            f"All ingests land in the shared Qdrant collection **`{COLLECTION}`**. "
            "The vector service automatically isolates rows by your token's user — "
            "you only ever retrieve your own embeddings. Use **Topic** to organize "
            "different documents within your private slice."
        ),
        kind="info",
    )

    config_panel = mo.accordion(
        {
            "⚙️ Ingestion settings": mo.vstack(
                [
                    topic,
                    mo.hstack([chat_model, top_k]),
                    mo.hstack([max_chunk_tokens, overlap_tokens]),
                    isolation_note,
                ]
            )
        }
    )
    config_panel
    return (
        chat_model,
        max_chunk_tokens,
        overlap_tokens,
        top_k,
        topic,
    )


@app.cell(hide_code=True)
def _helpers(Any, EMBED_BASE_URL, VECTOR_BASE_URL, requests):
    def approx_tokens(text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))

    def chunk_by_token_budget(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        words = text.split()
        if not words:
            return []
        max_words = max(20, int(max_tokens / 1.3))
        overlap_words = max(0, min(max_words - 1, int(overlap_tokens / 1.3)))

        chunks: list[str] = []
        i = 0
        while i < len(words):
            j = min(len(words), i + max_words)
            chunks.append(" ".join(words[i:j]))
            if j == len(words):
                break
            i = max(0, j - overlap_words)
        return chunks

    def extract_embedding(response_json: dict) -> list[float]:
        if isinstance(response_json.get("embedding"), list):
            return response_json["embedding"]
        data = response_json.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            emb = data[0].get("embedding")
            if isinstance(emb, list):
                return emb
        embeddings = response_json.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        raise ValueError(f"Could not parse embedding response: {response_json}")

    def call_embed(
        token: str,
        text: str,
        input_type: str,
        instruction: str | None = None,
    ) -> list[float]:
        headers = {"X-Tapis-Token": token, "Content-Type": "application/json"}
        cookies = {"X-Tapis-Token": token}
        payloads = [
            {
                "input": [text],
                "input_type": input_type,
                "instruction": instruction,
                "normalize": True,
            },
            {
                "input": text,
                "input_type": input_type,
                "instruction": instruction,
                "normalize": True,
            },
        ]
        last_error = None
        for payload in payloads:
            try:
                resp = requests.post(
                    f"{EMBED_BASE_URL}/v1/embed",
                    headers=headers,
                    cookies=cookies,
                    json=payload,
                    timeout=120,
                )
                if resp.status_code == 200:
                    return extract_embedding(resp.json())
                last_error = f"{resp.status_code}: {resp.text[:200]}"
            except requests.RequestException as exc:
                last_error = str(exc)
        raise RuntimeError(f"Embedding call failed. Last error: {last_error}")

    def store_chunk(
        token: str,
        embedding: list[float],
        chunk_text: str,
        collection: str,
        topic: str | None,
        metadata: dict[str, Any],
    ) -> dict:
        headers = {"X-Tapis-Token": token, "Content-Type": "application/json"}
        cookies = {"X-Tapis-Token": token}
        payload = {
            "embedding": embedding,
            "collection": collection,
            "topic": topic or None,
            "chunks": [chunk_text],
            "token_ids": [approx_tokens(chunk_text)],
            "embedding_model": "embed-service-default",
            "metadata": metadata,
        }
        resp = requests.post(
            f"{VECTOR_BASE_URL}/v1/embeddings",
            headers=headers,
            cookies=cookies,
            json=payload,
            timeout=120,
        )
        if resp.status_code != 201:
            raise RuntimeError(f"Store failed [{resp.status_code}]: {resp.text[:300]}")
        return resp.json()

    def retrieve_chunks(
        token: str,
        query_embedding: list[float],
        collection: str,
        topic: str | None,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        headers = {"X-Tapis-Token": token, "Content-Type": "application/json"}
        cookies = {"X-Tapis-Token": token}
        payload: dict[str, Any] = {
            "query_embedding": query_embedding,
            "top_k": top_k,
            "collection": collection,
            "topic": topic or None,
        }
        if metadata_filter:
            payload["filter"] = {"conditions": metadata_filter}
        resp = requests.post(
            f"{VECTOR_BASE_URL}/v1/retrieve",
            headers=headers,
            cookies=cookies,
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Retrieve failed [{resp.status_code}]: {resp.text[:300]}")
        return resp.json().get("results", [])

    def build_rag_message(question: str, context: str) -> str:
        return (
            "You are a RAG assistant.\n"
            "Use only the provided context to answer the user question.\n"
            "If the answer is not in the context, say: "
            "'I don't have enough information in the provided context.'\n"
            "Keep the answer concise and factual.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )

    return (
        build_rag_message,
        call_embed,
        chunk_by_token_budget,
        retrieve_chunks,
        store_chunk,
    )


@app.cell(hide_code=True)
def _file_helpers():
    import io

    # Keep uploads small — this is a demo, and big PDFs mean many embed calls.
    MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB

    def extract_text_from_file(name: str, contents: bytes) -> str:
        """Pull plain text out of an uploaded PDF / DOCX / TXT / MD file."""
        lower = name.lower()
        if lower.endswith(".pdf"):
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(contents))
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
            return "\n\n".join(p for p in pages if p)
        if lower.endswith(".docx"):
            import docx

            document = docx.Document(io.BytesIO(contents))
            return "\n".join(p.text for p in document.paragraphs if p.text.strip())
        # .txt, .md, or anything else — decode as UTF-8 text.
        return contents.decode("utf-8", errors="replace")

    return MAX_UPLOAD_BYTES, extract_text_from_file


@app.cell(hide_code=True)
def _doc_input(mo):
    sample_text = (
        "ICICLE AI provides embedding and vector services for grounded retrieval. "
        "A common pattern is to embed chunked source content, store vectors with "
        "metadata, retrieve top matches for a user question, and send those chunks "
        "to a chat model for constrained answering. The embedding service runs "
        "Qwen3-Embedding via llama-cpp-python. The vector service is FastAPI on top "
        "of Qdrant, with cosine similarity and MMR reranking. The chat service "
        "wraps a model deployment behind a JWT-gated endpoint."
    )
    document_input = mo.ui.text_area(
        value=sample_text,
        placeholder="Paste any text — a PDF excerpt, a wiki page, meeting notes, anything…",
        rows=12,
        full_width=True,
        label="**📄 Document to ingest**",
    )
    file_upload = mo.ui.file(
        filetypes=[".pdf", ".docx", ".txt", ".md"],
        multiple=False,
        kind="area",
        max_size=2 * 1024 * 1024,  # 2 MB cap, enforced client-side
        label="…or upload a file (PDF / DOCX / TXT / MD, ≤ 2 MB). A file takes priority over the text box.",
    )
    ingest_button = mo.ui.run_button(label="🚀 Ingest into vector store", kind="success")

    mo.vstack([document_input, file_upload, ingest_button])
    return document_input, file_upload, ingest_button


@app.cell(hide_code=True)
def _ingest(
    COLLECTION,
    MAX_UPLOAD_BYTES,
    call_embed,
    chunk_by_token_budget,
    document_input,
    extract_text_from_file,
    file_upload,
    ingest_button,
    max_chunk_tokens,
    mo,
    overlap_tokens,
    store_chunk,
    token,
    topic,
    uuid,
):
    # Fail fast on missing/invalid token — don't even try to call the API.
    mo.stop(
        token is None,
        mo.callout(
            mo.md(
                "❌ **Cannot ingest without a valid Tapis token.** "
                "Paste a fresh token in the box above (must start with `eyJ` and have two dots)."
            ),
            kind="danger",
        ),
    )

    mo.stop(
        not ingest_button.value,
        mo.md("_(Click **Ingest** above to chunk → embed → store the document.)_"),
    )

    # An uploaded file takes priority over the pasted text box.
    uploaded = file_upload.value
    if uploaded:
        upload = uploaded[0]
        if len(upload.contents) > MAX_UPLOAD_BYTES:
            mo.stop(
                True,
                mo.callout(
                    f"❌ **`{upload.name}` is too large** "
                    f"({len(upload.contents) / 1_048_576:.1f} MB). Limit is 2 MB — "
                    "upload a smaller file or paste an excerpt instead.",
                    kind="danger",
                ),
            )
        try:
            text_to_ingest = extract_text_from_file(upload.name, upload.contents).strip()
        except Exception as exc:
            mo.stop(
                True,
                mo.callout(
                    f"❌ **Couldn't read `{upload.name}`:** {exc}", kind="danger"
                ),
            )
        source_label = upload.name
        if not text_to_ingest:
            mo.stop(
                True,
                mo.callout(
                    f"❌ **No extractable text in `{upload.name}`.** "
                    "Scanned/image-only PDFs have no text layer — paste the text instead.",
                    kind="warn",
                ),
            )
    else:
        text_to_ingest = document_input.value.strip()
        source_label = "pasted text"
        if not text_to_ingest:
            mo.stop(True, mo.callout("Paste some text or upload a file first.", kind="warn"))

    chunks = chunk_by_token_budget(
        text_to_ingest, max_chunk_tokens.value, overlap_tokens.value
    )

    doc_id = str(uuid.uuid4())
    failures: list[str] = []

    with mo.status.progress_bar(
        total=len(chunks), title="Embedding + storing chunks", remove_on_exit=False
    ) as bar:
        for idx, chunk in enumerate(chunks):
            try:
                vec = call_embed(token, chunk, input_type="document")
                store_chunk(
                    token=token,
                    embedding=vec,
                    chunk_text=chunk,
                    collection=COLLECTION,
                    topic=topic.value,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "chunk_count": len(chunks),
                        "source": source_label,
                    },
                )
            except Exception as exc:
                failures.append(f"❌ Chunk {idx + 1}: {exc}")
            bar.update()

    if failures:
        ingest_summary = mo.callout(
            mo.md(
                f"**Ingested {len(chunks) - len(failures)} of {len(chunks)} chunks.**\n\n"
                + "\n".join(failures)
            ),
            kind="danger",
        )
    else:
        ingest_summary = mo.callout(
            mo.md(
                f"**🎉 Ingested all {len(chunks)} chunks**\n\n"
                f"- Source: `{source_label}`\n"
                f"- Document ID: `{doc_id}`\n"
                f"- Collection: `{COLLECTION}`\n"
                f"- Topic: `{topic.value or '(none)'}`"
            ),
            kind="success",
        )

    ingest_summary
    ingested = len(chunks) - len(failures)
    return (ingested,)


@app.cell(hide_code=True)
def _chat_section(mo):
    mo.md("""
    ## 💬 Ask questions about your document

    The chat below runs the full RAG loop on each message:
    1. Your question → embed service (as a query)
    2. Retrieve top-K nearest chunks from your collection
    3. Stitch chunks + question into a grounded prompt
    4. Send to the chat model and stream back the answer
    """)
    return


@app.cell(hide_code=True)
def _chat(
    CHAT_BASE_URL,
    COLLECTION,
    build_rag_message,
    call_embed,
    chat_model,
    ingested,
    mo,
    requests,
    retrieve_chunks,
    token,
    top_k,
    topic,
):
    mo.stop(
        token is None,
        mo.callout(
            mo.md(
                "❌ **Cannot chat without a valid Tapis token.** "
                "Paste a token above and re-ingest your document."
            ),
            kind="danger",
        ),
    )

    mo.stop(
        ingested == 0,
        mo.callout(
            "Ingest a document above before chatting.",
            kind="neutral",
        ),
    )

    def rag_chat(messages, _config):
        if not messages:
            return "Ask me anything about the document you ingested."

        question = messages[-1].content.strip()
        if not question:
            return "Please type a question."

        try:
            query_vec = call_embed(
                token=token,
                text=question,
                input_type="query",
                instruction="Represent this query for retrieving relevant passages.",
            )
        except Exception as exc:
            return f"❌ Embed call failed: {exc}"

        try:
            results = retrieve_chunks(
                token=token,
                query_embedding=query_vec,
                collection=COLLECTION,
                topic=topic.value,
                top_k=top_k.value,
            )
        except Exception as exc:
            return f"❌ Retrieval failed: {exc}"

        if not results:
            return "_No matching chunks found for that question._"

        context_blocks = []
        for i, item in enumerate(results, start=1):
            chunk_text = " ".join(item.get("chunks") or [])
            score = item.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
            context_blocks.append(f"[Chunk {i} | score={score_str}]\n{chunk_text}")
        rag_context = "\n\n".join(context_blocks)

        try:
            chat_resp = requests.post(
                f"{CHAT_BASE_URL}/chat",
                headers={"X-Tapis-Token": token, "Content-Type": "application/json"},
                cookies={"X-Tapis-Token": token},
                json={
                    "model": chat_model.value,
                    "message": build_rag_message(question, rag_context),
                },
                timeout=120,
            )
        except requests.RequestException as exc:
            return f"❌ Chat request failed: {exc}"

        if chat_resp.status_code != 200:
            return f"❌ Chat HTTP {chat_resp.status_code}: {chat_resp.text[:300]}"

        body = chat_resp.json()
        answer = body.get("answer") or body.get("response") or body
        if not isinstance(answer, str):
            import json as _json
            answer = _json.dumps(answer, indent=2)

        retrieved_summary = "\n".join(
            f"- _Chunk {i}_ (score `{r.get('score'):.3f}`): "
            f"{' '.join(r.get('chunks') or [])[:120]}…"
            for i, r in enumerate(results, start=1)
            if isinstance(r.get("score"), (int, float))
        )
        topic_suffix = f" / topic `{topic.value}`" if topic.value else ""
        return (
            f"{answer}\n\n"
            f"---\n"
            f"_Searched collection `{COLLECTION}`{topic_suffix} — top {top_k.value} chunks._\n\n"
            f"<details><summary>📎 Retrieved chunks</summary>\n\n"
            f"{retrieved_summary}\n\n</details>"
        )

    chat = mo.ui.chat(
        rag_chat,
        prompts=[
            "Summarize the document in two sentences.",
            "What are the main components mentioned?",
            "List any specific technologies referenced.",
        ],
    )
    chat
    return


@app.cell(hide_code=True)
def _footer(mo):
    mo.md("""
    ---
    Built on the **ICICLE AI** Tapis tenant.
    [Embed service](https://icicleaiembedserver.pods.icicleai.tapis.io/docs) ·
    [Vector service](https://icicleaivecserver.pods.icicleai.tapis.io/docs)
    """)
    return


if __name__ == "__main__":
    app.run()
