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
    import re
    import threading
    import time
    import uuid
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from typing import Any

    import marimo as mo
    import requests

    # OpenAI-compatible LiteLLM proxy on the *tacc* tenant. Replaces the old
    # `tapisagent` pod, which has been returning 503 "failed to start".
    LITELLM_BASE_URL = "https://litellm.pods.tacc.tapis.io"
    EMBED_BASE_URL = "https://icicleaiembedserver.pods.icicleai.tapis.io"
    VECTOR_BASE_URL = "https://icicleaivecserver.pods.icicleai.tapis.io"
    PORTAL_URL = "https://icicleai.tapis.io"

    # Used only if GET /v1/models can't be reached — the live roster is
    # authoritative, including for which of the duplicate ids are real.
    FALLBACK_MODELS = [
        "llama4-17b",
        "gpt-oss-120b",
        "Qwen3-32B",
        "Meta-Llama-3.3-70B-Instruct",
        "Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3.2-1B-Instruct",
        "MiniMax-M2.7",
        "gemma-4-31B-it",
    ]

    # Shared Qdrant collection. The vector service already isolates rows per
    # user via the token's subject, so a fixed collection is safe — you only
    # ever see your own embeddings. Use `topic` (set in the config panel) to
    # organize different documents within your private slice of the collection.
    COLLECTION = "icicle-demo-collection"
    return (
        Any,
        COLLECTION,
        EMBED_BASE_URL,
        FALLBACK_MODELS,
        LITELLM_BASE_URL,
        PORTAL_URL,
        Path,
        ThreadPoolExecutor,
        VECTOR_BASE_URL,
        json,
        mo,
        os,
        re,
        requests,
        threading,
        time,
        uuid,
    )


@app.cell(hide_code=True)
def _title(EMBED_BASE_URL, LITELLM_BASE_URL, Path, VECTOR_BASE_URL, mo):
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
        | **3. Chat** | `litellm` | OpenAI-compatible proxy; generates answers from the chunks | [Models]({LITELLM_BASE_URL}/v1/models) |

        All three live behind the same `X-Tapis-Token`. Get yours below 👇
        """
    )

    mo.vstack([header, body], gap=1)
    return


@app.cell(hide_code=True)
def _token_source(mo, os):
    """Where the Tapis token comes from — the pod already knows who you are.

    Served from an ICICLE AI Tapis pod, the browser holds an `X-Tapis-Token`
    cookie for `*.tapis.io` (that cookie is how you got past the login page at
    all), and marimo hands the session's HTTP request to the kernel. So on a pod
    we read the token from there and skip the paste box entirely; run locally
    there is no such cookie and no request, so the UI asks for one.

    `ICICLE_TOKEN_SOURCE`: `auto` (default — cookie if there is one, otherwise
    paste), `cookie` (same, but say so loudly when the cookie is missing), or
    `manual` (never look at the cookie, which is what you want when testing the
    paste flow on a pod).
    """
    TOKEN_SOURCE = os.environ.get("ICICLE_TOKEN_SOURCE", "auto").strip().lower()
    if TOKEN_SOURCE not in ("auto", "cookie", "manual"):
        TOKEN_SOURCE = "auto"

    def _from_request() -> str:
        # `request` is None in edit mode and in any non-served context.
        request = mo.app_meta().request
        if request is None:
            return ""
        # Cookie jars are case-sensitive but proxies are not consistent, and the
        # same value may arrive as a header instead — take whichever is present.
        for source in ((request.cookies or {}), (request.headers or {})):
            for key, value in source.items():
                if key.lower() == "x-tapis-token" and (value or "").strip():
                    return value.strip()
        return ""

    COOKIE_TOKEN = "" if TOKEN_SOURCE == "manual" else _from_request()

    # With a cookie the paste box and the walkthrough are noise — the user is
    # already signed in to Tapis, which is the only way they reached this page.
    SHOW_TOKEN_INPUT = not COOKIE_TOKEN
    return COOKIE_TOKEN, SHOW_TOKEN_INPUT, TOKEN_SOURCE


@app.cell(hide_code=True)
def _token_help(PORTAL_URL, Path, SHOW_TOKEN_INPUT, mo):
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

    _help = mo.accordion(
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
    _help if SHOW_TOKEN_INPUT else None
    return


@app.cell(hide_code=True)
def _token_input(SHOW_TOKEN_INPUT, mo, os):
    # Both elements are always constructed — `_token_status` reads them either
    # way — but on a pod, where the token comes from the session cookie, only the
    # re-check button is shown.
    token_input = mo.ui.text(
        value=os.environ.get("TAPIS_TOKEN", ""),
        placeholder="Paste your X-Tapis-Token here (eyJ...)",
        full_width=True,
        kind="password",
        label="**X-Tapis-Token**",
    )
    validate_button = mo.ui.run_button(
        label="🔐 Validate token" if SHOW_TOKEN_INPUT else "🔄 Re-check Tapis session",
        kind="info",
        tooltip="Pings the embed service /v1/model endpoint to confirm the token works.",
    )
    mo.vstack([token_input, validate_button] if SHOW_TOKEN_INPUT else [validate_button])
    return token_input, validate_button


@app.cell(hide_code=True)
def _token_status(
    COOKIE_TOKEN,
    EMBED_BASE_URL,
    FALLBACK_MODELS,
    LITELLM_BASE_URL,
    TOKEN_SOURCE,
    mo,
    requests,
    token_input,
    validate_button,
):
    # A cookie token validates on load: there is nothing for the user to paste
    # and nothing to click, so waiting for a button press would just be a wall.
    from_cookie = bool(COOKIE_TOKEN)
    raw = (COOKIE_TOKEN or token_input.value).strip()
    token = None  # default — downstream stays locked unless we set this
    chat_models = list(FALLBACK_MODELS)

    def _fetch_models(tok):
        """GET /v1/models → (ids, error). Doubles as a tacc-tenant token check.

        `allow_redirects=False` is load-bearing: unauthenticated, the LiteLLM pod
        302s to the Tapis OAuth page, which then serves a 200 HTML login form. With
        redirects followed, an auth failure would look like a success and only blow
        up later as a JSONDecodeError.
        """
        try:
            resp = requests.get(
                f"{LITELLM_BASE_URL}/v1/models",
                headers={"X-Tapis-Token": tok},
                cookies={"X-Tapis-Token": tok},
                timeout=15,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            return [], f"network error: {exc}"

        if 300 <= resp.status_code < 400:
            return [], "not authenticated for the `tacc` tenant (redirected to login)"
        if resp.status_code != 200:
            return [], f"HTTP {resp.status_code}: {resp.text[:120]}"

        try:
            data = resp.json().get("data") or []
        except ValueError:
            return [], "unexpected response (not JSON)"

        # whisper/tts/embed/rerank are hosted here too but aren't chat models.
        skip = ("whisper", "tts", "embed", "rerank")
        ids, seen = [], set()
        for entry in data:
            mid = (entry or {}).get("id")
            if not mid or mid in seen or any(k in mid.lower() for k in skip):
                continue
            seen.add(mid)
            ids.append(mid)
        return sorted(ids), None if ids else "no chat-capable models listed"

    if not (from_cookie or validate_button.value):
        # User hasn't clicked Validate yet.
        if TOKEN_SOURCE == "cookie":
            _status = mo.callout(
                mo.md(
                    "⚠️ **No `X-Tapis-Token` cookie on this session**, though "
                    "`ICICLE_TOKEN_SOURCE=cookie` expects one. Either this isn't "
                    "running behind the Tapis pod proxy, or your session has "
                    "expired — paste a token below to carry on."
                ),
                kind="warn",
            )
        elif raw:
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
        # Two tenants, two checks: embed/vector live on `icicleai`, LiteLLM on
        # `tacc`. A token can be good for one and not the other, so report them
        # separately — embed alone is still enough to ingest.
        try:
            resp = requests.get(
                f"{EMBED_BASE_URL}/v1/model",
                headers={"X-Tapis-Token": raw},
                cookies={"X-Tapis-Token": raw},
                timeout=10,
            )
            if resp.status_code == 200:
                model_info = resp.json()
                token = raw
                found, models_error = _fetch_models(raw)
                if found:
                    chat_models = found
                _lines = [
                    ("🔐 **Signed in from your Tapis session** — token read from "
                     "the pod's `X-Tapis-Token` cookie, nothing to paste.")
                    if from_cookie
                    else "",
                    f"✅ **Token validated** on `icicleai` (embed + vector). "
                    f"Model: `{model_info.get('model', '?')}` · "
                    f"dim: `{model_info.get('dim', '?')}` · "
                    f"n_ctx: `{model_info.get('n_ctx', '?')}`."
                ]
                if models_error:
                    _lines.append(
                        f"⚠️ **LiteLLM (`tacc`) unavailable** — {models_error}. "
                        f"Ingestion works; answering may not. "
                        f"Falling back to a built-in model list."
                    )
                    _kind = "warn"
                else:
                    _lines.append(
                        f"✅ **LiteLLM (`tacc`) reachable** — {len(chat_models)} chat models."
                    )
                    _kind = "success"
                _status = mo.callout(
                    mo.md("\n\n".join(_l for _l in _lines if _l)), kind=_kind
                )
            elif resp.status_code == 401:
                _status = mo.callout(
                    (
                        "❌ **Session expired (401).** Tapis access tokens last ~4 hours. "
                        "Reload this page to pick up a fresh cookie, signing in again if "
                        "Tapis asks you to."
                        if from_cookie
                        else "❌ **Token rejected (401).** Likely expired — Tapis access "
                        "tokens last ~4 hours. Get a fresh one from the Tapis UI and click "
                        "Validate again."
                    ),
                    kind="danger",
                )
            elif resp.status_code == 403:
                _status = mo.callout(
                    "❌ **Wrong tenant (403).** The embed service only accepts tokens from the "
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
    return chat_models, token


@app.cell(hide_code=True)
def _eval_config(os):
    """Environment-driven config, named to match `icicle-ai-embed-service`.

    Everything here is Tapis-token driven — the MLflow pod is gated the same way
    the AI services are, so metric logging reuses the token you already pasted
    rather than a service account. There is deliberately no local file-store
    fallback: an empty MLFLOW_TRACKING_URI simply disables metrics.
    """

    def _flag(name: str, default: str) -> bool:
        return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")

    MLFLOW_ENABLED = _flag("MLFLOW_ENABLED", "false")
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "").strip().rstrip("/")
    MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "icicle-chatbook").strip()
    MLFLOW_TIMEOUT_SECONDS = float(os.environ.get("MLFLOW_TIMEOUT_SECONDS", "2.0"))

    # The sibling embed service logs only anonymous request shape. The chatbook
    # deliberately diverges — the questions, answers and judge rationales are the
    # point — but set MLFLOW_LOG_CONTENT=0 to match its posture exactly.
    MLFLOW_LOG_CONTENT = _flag("MLFLOW_LOG_CONTENT", "1")

    EVAL_ENABLED_DEFAULT = _flag("ICICLE_EVAL_ENABLED", "1")
    JUDGE_MODEL_ENV = os.environ.get("ICICLE_JUDGE_MODEL", "gpt-oss-120b").strip()

    # GenAI tracing is a separate surface from the run/metric logging above: it
    # feeds MLflow's Traces tab rather than the experiment table, and needs the
    # `mlflow-tracing` SDK because a span tree is not something worth hand-rolling
    # over REST. Defaults to following MLFLOW_ENABLED so one switch turns on both.
    MLFLOW_TRACING_ENABLED = _flag(
        "MLFLOW_TRACING_ENABLED", "true" if MLFLOW_ENABLED else "false"
    )
    return (
        EVAL_ENABLED_DEFAULT,
        JUDGE_MODEL_ENV,
        MLFLOW_ENABLED,
        MLFLOW_EXPERIMENT,
        MLFLOW_LOG_CONTENT,
        MLFLOW_TIMEOUT_SECONDS,
        MLFLOW_TRACING_ENABLED,
        MLFLOW_TRACKING_URI,
    )


@app.cell(hide_code=True)
def _tracing(
    MLFLOW_EXPERIMENT,
    MLFLOW_LOG_CONTENT,
    MLFLOW_TIMEOUT_SECONDS,
    MLFLOW_TRACING_ENABLED,
    MLFLOW_TRACKING_URI,
    os,
    threading,
):
    """MLflow GenAI tracing — the Traces tab, as opposed to the experiment table.

    Runs and metrics (see `EvalSession`) still go over raw REST. Spans do not:
    a trace is a tree with ids, parentage and timing, and hand-rolling that over
    REST would be a reimplementation of the SDK rather than an avoidance of it.

    Auth is the interesting part. The pod is behind the Tapis proxy, which wants
    `X-Tapis-Token`; the SDK only knows `Authorization: Bearer`. Unauthenticated,
    the proxy 302s to the OAuth page and serves a 200 HTML login form — the same
    trap `_api` and `_fetch_models` guard against with allow_redirects=False,
    except here the SDK owns the request and we cannot pass that flag. So the
    token is injected instead, through the one hook MLflow routes *every*
    outgoing tracking request through (`rest_utils.http_request` →
    `resolve_request_headers`).
    """

    # Read per request, not captured at registration: a user who revalidates with
    # a fresh token mid-session must not keep sending the expired one.
    TAPIS_TOKEN_HOLDER = {"value": None}

    class _NullSpan:
        """Stands in when tracing is off, so callers never branch on it."""

        def set_inputs(self, *_a, **_k) -> None:
            pass

        def set_outputs(self, *_a, **_k) -> None:
            pass

        def set_attributes(self, *_a, **_k) -> None:
            pass

    class _NullCtx:
        def __enter__(self):
            return _NullSpan()

        def __exit__(self, *_exc) -> bool:
            return False

    class _SafeSpan:
        """A live span whose setters can never raise into the RAG loop.

        The call sites sit in the middle of `answer_question` — one of them
        outside any try — so an exception from a setter would surface as a failed
        answer or a traceback in the transcript. Recording a span is never worth
        that, so every setter is swallowed here rather than guarded at each site.
        """

        def __init__(self, span) -> None:
            self._span = span

        def _safe(self, method: str, value) -> None:
            try:
                getattr(self._span, method)(value)
            except Exception:
                pass

        def set_inputs(self, v) -> None:
            self._safe("set_inputs", v)

        def set_outputs(self, v) -> None:
            self._safe("set_outputs", v)

        def set_attributes(self, v) -> None:
            self._safe("set_attributes", v)

    class _SpanCtx:
        """Wraps `mlflow.start_span` so inputs can be attached on entry.

        `start_span` takes `attributes`, not `inputs` — inputs live on the span
        object and are set once it exists. Doing that here keeps every call site
        a plain `with tracer.span(...) as span`.
        """

        def __init__(self, cm, inputs) -> None:
            self._cm = cm
            self._inputs = inputs

        def __enter__(self):
            span = _SafeSpan(self._cm.__enter__())
            if self._inputs is not None:
                span.set_inputs(self._inputs)
            return span

        def __exit__(self, *exc) -> bool:
            try:
                return self._cm.__exit__(*exc)
            except Exception:
                # Never let a failed export mask (or invent) an error in the
                # traced code itself.
                return False

    class Tracer:
        """Fail-open span emitter. Mirrors EvalSession's posture exactly: every
        failure is recorded and swallowed, because an observability feature that
        can break an answer is worse than no observability feature."""

        def __init__(self) -> None:
            self.enabled = bool(MLFLOW_TRACING_ENABLED and MLFLOW_TRACKING_URI)
            self.error: str | None = None
            self.ready = False
            self.lock = threading.Lock()
            self.mlflow = None
            self.Document = None
            # One init attempt per token. `_ensure_ready` runs on the answer path,
            # and MLflow retries failed HTTP with backoff — so a sleeping pod would
            # otherwise stall *every* question, several seconds at a time, which is
            # exactly the delay this whole layer promises never to introduce.
            self.attempted_for: str | None = None
            # MLFLOW_LOG_CONTENT=0 means metrics-only, and a span's inputs and
            # outputs are content — questions, answers, chunk text. Honour the
            # same switch here or the flag would silently stop meaning anything.
            self.redact = not MLFLOW_LOG_CONTENT

        def _ensure_ready(self) -> bool:
            """Import, register and resolve the experiment on first real use.

            Deferred because `set_experiment` is an authenticated call: at import
            time there is no token yet, and eagerly failing would disable tracing
            for the whole session over a race with the login step.
            """
            if self.ready or not self.enabled:
                return self.ready
            with self.lock:
                if self.ready:
                    return True
                tok = TAPIS_TOKEN_HOLDER["value"]
                if not tok or self.attempted_for == tok:
                    return False
                self.attempted_for = tok
                try:
                    # Set before the import: MLflow reads these per request, and
                    # the defaults (5 retries with backoff) are tuned for batch
                    # jobs, not for something sitting in front of a chat turn.
                    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
                    os.environ.setdefault(
                        "MLFLOW_HTTP_REQUEST_TIMEOUT", str(int(MLFLOW_TIMEOUT_SECONDS))
                    )

                    import mlflow
                    from mlflow.entities import Document
                    from mlflow.tracking.request_header.abstract_request_header_provider import (
                        RequestHeaderProvider,
                    )
                    from mlflow.tracking.request_header.registry import (
                        _request_header_provider_registry as _registry,
                    )

                    class TapisRequestHeaderProvider(RequestHeaderProvider):
                        def in_context(self) -> bool:
                            return bool(TAPIS_TOKEN_HOLDER["value"])

                        def request_headers(self) -> dict:
                            tok = TAPIS_TOKEN_HOLDER["value"]
                            # Cookie as well as header: the proxy accepts either,
                            # and every hand-written call in this notebook sends
                            # both. `Cookie` is just a header, so one hook covers it.
                            return {
                                "X-Tapis-Token": tok,
                                "Cookie": f"X-Tapis-Token={tok}",
                            }

                    # Registering twice is not idempotent — `resolve_request_headers`
                    # *concatenates* duplicate keys, so a second registration would
                    # send `X-Tapis-Token: "<tok> <tok>"` and fail auth on every
                    # request. A reactive notebook re-runs cells freely, so guard by
                    # class name rather than trusting this to run once.
                    if not any(
                        type(p).__name__ == "TapisRequestHeaderProvider" for p in _registry
                    ):
                        _registry.register(TapisRequestHeaderProvider)

                    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                    mlflow.set_experiment(MLFLOW_EXPERIMENT)
                    self.mlflow = mlflow
                    self.Document = Document
                    self.ready = True
                    self.error = None
                except Exception as exc:
                    self.error = f"tracing unavailable — {exc}"
                return self.ready

        def span(self, name: str, span_type: str, inputs: dict | None = None):
            if not self._ensure_ready():
                return _NullCtx()
            try:
                return _SpanCtx(
                    self.mlflow.start_span(name=name, span_type=span_type),
                    None if self.redact else inputs,
                )
            except Exception as exc:
                self.error = f"span failed — {exc}"
                return _NullCtx()

        def documents(self, results: list[dict]) -> list:
            """Vector-service rows → `Document`s, the shape the RETRIEVER span type
            requires. It is what unlocks MLflow's built-in RAG judges
            (RetrievalGroundedness / RetrievalRelevance / RetrievalSufficiency);
            an untyped span is invisible to them.
            """
            if not self.ready:
                return []
            docs = []
            for i, r in enumerate(results):
                meta = r.get("metadata") or {}
                docs.append(
                    self.Document(
                        id=f"{meta.get('doc_id', '?')}#{meta.get('chunk_index', i)}",
                        # Redacted mode keeps the retrieval *shape* — ids, scores,
                        # provenance — and drops only the text.
                        page_content=(
                            "" if self.redact else " ".join(r.get("chunks") or [])
                        ),
                        metadata={
                            "score": r.get("score"),
                            "source": meta.get("source"),
                            "chunk_index": meta.get("chunk_index"),
                        },
                    )
                )
            return docs

        @property
        def status(self) -> str:
            if not self.enabled:
                return "off"
            if self.error:
                return "error"
            return "ok" if self.ready else "pending"

    tracer = Tracer()
    return TAPIS_TOKEN_HOLDER, tracer


@app.cell(hide_code=True)
def _tracing_token(TAPIS_TOKEN_HOLDER, token):
    # Republish the validated token wherever the SDK can reach it. Reactive, so a
    # revalidation propagates to the header provider without re-registering it.
    TAPIS_TOKEN_HOLDER["value"] = token
    return


@app.cell(hide_code=True)
def _probe_button(mo):
    # Defined apart from the cell that reads `.value`: a click re-runs readers,
    # never the defining cell, so the button itself survives the round trip.
    probe_button = mo.ui.run_button(
        label="🩺 Check which models are up",
        tooltip="Sends a 1-token 'ping' to every listed model — a few seconds.",
    )

    # Remembers the picked models across the re-render a probe (or a token
    # revalidation) causes. Set from the dropdowns' own cell, so no self-loop.
    get_picks, set_picks = mo.state({})
    return get_picks, probe_button, set_picks


@app.cell(hide_code=True)
def _model_health(ThreadPoolExecutor, chat_models, litellm_chat, mo, probe_button, token):
    """Liveness per model: `/v1/models` lists what's configured, not what answers.

    A model can be listed and still be undeployed, out of quota, or 500ing. The
    cheapest honest check is the real thing: one 1-token completion each, in
    parallel because they're all network-bound.
    """
    model_health: dict[str, str | None] = {}
    probe_note = mo.md("")

    if probe_button.value and token:
        def _probe(model: str) -> tuple[str, str | None]:
            out = litellm_chat(
                token,
                model,
                [{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
                timeout=25,
            )
            return model, (str(out["error"])[:150] if out.get("error") else None)

        with mo.status.spinner(title=f"Pinging {len(chat_models)} models…"):
            with ThreadPoolExecutor(max_workers=8) as pool:
                model_health = dict(pool.map(_probe, chat_models))

        _down = {m: e for m, e in model_health.items() if e}
        _up = len(model_health) - len(_down)
        if _down:
            probe_note = mo.md(
                f"✅ {_up} of {len(model_health)} models answered. "
                f"❌ didn't: "
                + " · ".join(f"`{m}`" for m in _down)
                + "\n\n<details><summary>Why they failed</summary>\n\n"
                + "\n".join(f"- **{m}** — {e}" for m, e in _down.items())
                + "\n\n</details>"
            )
        else:
            probe_note = mo.md(f"✅ All {_up} listed models answered.")
    elif probe_button.value:
        probe_note = mo.md("_Validate a token first — the check needs one._")
    return model_health, probe_note


@app.cell(hide_code=True)
def _config_form(
    COLLECTION,
    EVAL_ENABLED_DEFAULT,
    JUDGE_MODEL_ENV,
    MLFLOW_ENABLED,
    MLFLOW_TRACKING_URI,
    chat_models,
    get_picks,
    mo,
    model_health,
    probe_button,
    probe_note,
    set_picks,
):
    topic = mo.ui.text(
        value="general",
        label="**Topic** (a label that groups this document — e.g. `paper-2024`, `notes`)",
        full_width=True,
    )

    # Options come from the live /v1/models roster, so these rebuild once when you
    # validate a token — selections reset at that moment, by design.
    _answer_default = "llama4-17b" if "llama4-17b" in chat_models else chat_models[0]

    # Judge on a different model family than the answerer: a model asked to grade
    # its own output scores it generously.
    _judge_default = JUDGE_MODEL_ENV if JUDGE_MODEL_ENV in chat_models else next(
        (m for m in chat_models if m != _answer_default), _answer_default
    )

    # ✅ / ❌ come from the last availability check; unchecked models are shown
    # bare rather than guessed at. The label carries the mark, the value stays the
    # plain model id, so nothing downstream has to strip anything.
    def _label(model: str) -> str:
        health = model_health.get(model, "unchecked")
        if health == "unchecked":
            return model
        return f"❌ {model}" if health else f"✅ {model}"

    _options = {_label(m): m for m in chat_models}

    def _picked(slot: str, default: str) -> str:
        choice = get_picks().get(slot)
        return _label(choice if choice in chat_models else default)

    chat_model = mo.ui.dropdown(
        options=_options,
        value=_picked("chat", _answer_default),
        label="Chat model",
        on_change=lambda v: set_picks(lambda p: {**p, "chat": v}),
    )
    judge_model = mo.ui.dropdown(
        options=_options,
        value=_picked("judge", _judge_default),
        label="🧪 Judge model",
        on_change=lambda v: set_picks(lambda p: {**p, "judge": v}),
    )
    eval_enabled = mo.ui.switch(
        value=EVAL_ENABLED_DEFAULT, label="🧪 Evaluate answers (G-Eval)"
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

    if MLFLOW_ENABLED and MLFLOW_TRACKING_URI:
        _mlflow_note = mo.callout(
            mo.md(
                f"🧪 Judge scores log to MLflow at `{MLFLOW_TRACKING_URI}`, "
                f"authenticated with the same `X-Tapis-Token`."
            ),
            kind="info",
        )
    else:
        _mlflow_note = mo.callout(
            mo.md(
                "🧪 MLflow logging is **off**. Set `MLFLOW_ENABLED=true` and "
                "`MLFLOW_TRACKING_URI=<tapis mlflow pod>` to record scores. "
                "Scores still show inline while it's off."
            ),
            kind="neutral",
        )

    config_panel = mo.accordion(
        {
            "⚙️ Ingestion settings": mo.vstack(
                [
                    topic,
                    mo.hstack([chat_model, top_k]),
                    mo.hstack([probe_button, probe_note], align="center", gap=1),
                    mo.hstack([max_chunk_tokens, overlap_tokens]),
                    isolation_note,
                ]
            ),
            "🧪 Evaluation settings": mo.vstack(
                [
                    mo.md(
                        "One judge model scores each answer 1–5 (G-Eval) on "
                        "faithfulness to your document, relevance to the question, "
                        "quality of the retrieved passages, and clarity. It runs in "
                        "the background and costs 4 extra model calls per answer."
                    ),
                    mo.hstack([eval_enabled, judge_model]),
                    probe_note,
                    _mlflow_note,
                ]
            ),
        }
    )
    config_panel
    return (
        chat_model,
        eval_enabled,
        judge_model,
        max_chunk_tokens,
        overlap_tokens,
        top_k,
        topic,
    )


@app.cell(hide_code=True)
def _helpers(
    Any, EMBED_BASE_URL, LITELLM_BASE_URL, VECTOR_BASE_URL, requests, time, tracer
):
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

    def litellm_chat(
        token: str,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.5,
        max_tokens: int = 4096,
        logprobs: bool = False,
        top_logprobs: int = 0,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """One OpenAI-compatible call to the LiteLLM proxy.

        Returns ``{"content", "raw", "error"}`` — errors come back as strings so
        every caller (answers *and* the judge) can render them inline.

        `allow_redirects=False` is load-bearing: unauthenticated, the pod 302s to
        the Tapis OAuth page, which then serves a 200 HTML login form. Following
        redirects would make an auth failure look like success and surface much
        later as a confusing JSONDecodeError.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs:
                payload["top_logprobs"] = top_logprobs

        try:
            resp = requests.post(
                f"{LITELLM_BASE_URL}/v1/chat/completions",
                headers={"X-Tapis-Token": token, "Content-Type": "application/json"},
                cookies={"X-Tapis-Token": token},
                json=payload,
                timeout=timeout,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            return {"content": "", "raw": None, "error": f"❌ Chat request failed: {exc}"}

        if 300 <= resp.status_code < 400:
            return {
                "content": "",
                "raw": None,
                "error": (
                    "❌ **Not authenticated for LiteLLM.** The request was redirected to "
                    "the Tapis login page — the token is missing, expired, or not valid "
                    "for the `tacc` tenant."
                ),
            }
        if resp.status_code != 200:
            return {
                "content": "",
                "raw": None,
                "error": f"❌ Chat HTTP {resp.status_code}: {resp.text[:300]}",
            }

        try:
            body = resp.json()
            choice = (body.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content")
        except (ValueError, AttributeError, IndexError) as exc:
            return {
                "content": "",
                "raw": None,
                "error": f"❌ Unexpected chat response shape: {exc}",
            }

        if not isinstance(content, str):
            return {
                "content": "",
                "raw": body,
                "error": "❌ Chat response had no message content.",
            }
        return {"content": content, "raw": body, "error": None}

    def build_rag_messages(
        question: str,
        context: str,
        history: list[dict] | None = None,
        max_turns: int = 3,
        max_answer_chars: int = 400,
    ) -> list[dict]:
        """Compose the OpenAI-style message array for one grounded answer.

        Prior turns go in as genuine `user`/`assistant` messages rather than being
        stitched into one string, so each model applies its own chat template.
        Capped at the last `max_turns` with answers truncated — with no
        clear-conversation control, that cap is what bounds the prompt.
        """
        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are a RAG assistant.\n"
                    "Use only the provided context to answer the user question.\n"
                    "If the answer is not in the context, say: "
                    "'I don't have enough information in the provided context.'\n"
                    "Keep the answer concise and factual."
                ),
            }
        ]
        for turn in [t for t in (history or []) if not t.get("error")][-max_turns:]:
            messages.append({"role": "user", "content": turn.get("question", "")})
            messages.append(
                {"role": "assistant", "content": (turn.get("answer") or "")[:max_answer_chars]}
            )
        messages.append(
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        )
        return messages

    def format_answer_md(
        answer: str,
        results: list[dict],
        *,
        collection: str,
        topic: str | None,
        top_k: int,
    ) -> str:
        """Answer + provenance footer + the collapsible retrieved-chunks block."""
        retrieved_summary = "\n".join(
            f"- _Chunk {i}_ (score `{r.get('score'):.3f}`): "
            f"{' '.join(r.get('chunks') or [])[:120]}…"
            for i, r in enumerate(results, start=1)
            if isinstance(r.get("score"), (int, float))
        )
        topic_suffix = f" / topic `{topic}`" if topic else ""
        return (
            f"{answer}\n\n"
            f"---\n"
            f"_Searched collection `{collection}`{topic_suffix} — top {top_k} chunks._\n\n"
            f"<details><summary>📎 Retrieved chunks</summary>\n\n"
            f"{retrieved_summary}\n\n</details>"
        )

    def build_context_block(results: list[dict]) -> str:
        """The retrieved chunks, formatted as the context the model is grounded on."""
        blocks = []
        for i, item in enumerate(results, start=1):
            chunk_text = " ".join(item.get("chunks") or [])
            score = item.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
            blocks.append(f"[Chunk {i} | score={score_str}]\n{chunk_text}")
        return "\n\n".join(blocks)

    def answer_question(
        token: str,
        question: str,
        *,
        collection: str,
        topic: str | None,
        top_k: int,
        chat_model: str,
        history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """One full RAG round-trip: embed the question → retrieve → chat.

        `history`, when given, is folded into the message array only — the query
        that gets embedded is always the bare question, so chunk selection stays
        reproducible and doesn't drift with the conversation.

        Returns ``{"answer", "results", "context", "timings", "error"}``. Failures
        come back as an ``error`` string rather than an exception so callers can
        render them inline without their own try/except.
        """
        timings = {"embed_ms": 0.0, "retrieve_ms": 0.0, "chat_ms": 0.0}
        blank = {"answer": "", "results": [], "context": "", "timings": timings}

        question = question.strip()
        if not question:
            return {**blank, "error": "Please type a question."}

        with tracer.span("rag_turn", "CHAIN", {"question": question}) as _turn_span:
            out = _answer_question_traced(
                token,
                question,
                collection=collection,
                topic=topic,
                top_k=top_k,
                chat_model=chat_model,
                history=history,
                timings=timings,
                blank=blank,
            )
            _turn_span.set_outputs(None if tracer.redact else out.get("answer"))
            _turn_span.set_attributes(
                {
                    "topic": topic or "",
                    "top_k": top_k,
                    "chunks_retrieved": len(out.get("results") or []),
                    **{k: round(v, 1) for k, v in out["timings"].items()},
                }
            )
            return out

    def _answer_question_traced(
        token: str,
        question: str,
        *,
        collection: str,
        topic: str | None,
        top_k: int,
        chat_model: str,
        history: list[dict] | None,
        timings: dict,
        blank: dict,
    ) -> dict[str, Any]:
        """The body of `answer_question`, split out so the CHAIN span can wrap it
        as a unit and still see the return value."""

        _t0 = time.perf_counter()
        try:
            with tracer.span("embed_query", "EMBEDDING", {"text": question}) as _sp:
                query_vec = call_embed(
                    token=token,
                    text=question,
                    input_type="query",
                    instruction="Represent this query for retrieving relevant passages.",
                )
                _sp.set_attributes({"dim": len(query_vec)})
        except Exception as exc:
            return {**blank, "error": f"❌ Embed call failed: {exc}"}
        timings["embed_ms"] = (time.perf_counter() - _t0) * 1000

        _t0 = time.perf_counter()
        try:
            # RETRIEVER, specifically: the built-in RAG judges key off the span
            # type, and its outputs must be `Document`s for them to see the chunks.
            with tracer.span(
                "retrieve", "RETRIEVER", {"query": question, "top_k": top_k}
            ) as _sp:
                results = retrieve_chunks(
                    token=token,
                    query_embedding=query_vec,
                    collection=collection,
                    topic=topic,
                    top_k=top_k,
                )
                _sp.set_outputs(tracer.documents(results))
        except Exception as exc:
            return {**blank, "error": f"❌ Retrieval failed: {exc}"}
        timings["retrieve_ms"] = (time.perf_counter() - _t0) * 1000

        if not results:
            return {**blank, "error": "_No matching chunks found for that question._"}

        rag_context = build_context_block(results)

        _t0 = time.perf_counter()
        with tracer.span(
            "generate", "LLM", {"question": question, "context": rag_context}
        ) as _sp:
            chat = litellm_chat(
                token,
                chat_model,
                build_rag_messages(question, rag_context, history),
            )
            _sp.set_attributes({"model": chat_model, "chunks_in_context": len(results)})
            _sp.set_outputs(None if tracer.redact else chat.get("content"))
        timings["chat_ms"] = (time.perf_counter() - _t0) * 1000

        if chat["error"]:
            return {
                "answer": "",
                "results": results,
                "context": rag_context,
                "timings": timings,
                "error": chat["error"],
            }

        return {
            "answer": chat["content"],
            "results": results,
            "context": rag_context,
            "timings": timings,
            "error": None,
        }

    return (
        answer_question,
        build_context_block,
        build_rag_messages,
        call_embed,
        chunk_by_token_budget,
        format_answer_md,
        litellm_chat,
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
    # This cell deliberately avoids `mo.stop`: `ingested` has to be bound on
    # every path, because the chat and study panes below both key off it and
    # would otherwise never render their "not ready yet" message.
    def _resolve_source():
        """→ (text, label, error_output); exactly one of text / error is set."""
        uploaded = file_upload.value
        if uploaded:
            # An uploaded file takes priority over the pasted text box.
            upload = uploaded[0]
            if len(upload.contents) > MAX_UPLOAD_BYTES:
                return "", "", mo.callout(
                    f"❌ **`{upload.name}` is too large** "
                    f"({len(upload.contents) / 1_048_576:.1f} MB). Limit is 2 MB — "
                    "upload a smaller file or paste an excerpt instead.",
                    kind="danger",
                )
            try:
                text = extract_text_from_file(upload.name, upload.contents).strip()
            except Exception as exc:
                return "", "", mo.callout(
                    f"❌ **Couldn't read `{upload.name}`:** {exc}", kind="danger"
                )
            if not text:
                return "", "", mo.callout(
                    f"❌ **No extractable text in `{upload.name}`.** "
                    "Scanned/image-only PDFs have no text layer — paste the text instead.",
                    kind="warn",
                )
            return text, upload.name, None

        text = document_input.value.strip()
        if not text:
            return "", "", mo.callout(
                "Paste some text or upload a file first.", kind="warn"
            )
        return text, "pasted text", None

    ingested = 0

    if token is None:
        ingest_summary = mo.callout(
            mo.md(
                "❌ **Cannot ingest without a valid Tapis token.** "
                "Paste a fresh token in the box above (must start with `eyJ` and have two dots)."
            ),
            kind="danger",
        )
    elif not ingest_button.value:
        ingest_summary = mo.md(
            "_(Click **Ingest** above to chunk → embed → store the document.)_"
        )
    else:
        _text_to_ingest, _source_label, _source_error = _resolve_source()
        if _source_error is not None:
            ingest_summary = _source_error
        else:
            _chunks = chunk_by_token_budget(
                _text_to_ingest, max_chunk_tokens.value, overlap_tokens.value
            )
            _doc_id = str(uuid.uuid4())
            _failures: list[str] = []

            with mo.status.progress_bar(
                total=len(_chunks),
                title="Embedding + storing chunks",
                remove_on_exit=False,
            ) as _ingest_bar:
                for _idx, _chunk in enumerate(_chunks):
                    try:
                        _vec = call_embed(token, _chunk, input_type="document")
                        store_chunk(
                            token=token,
                            embedding=_vec,
                            chunk_text=_chunk,
                            collection=COLLECTION,
                            topic=topic.value,
                            metadata={
                                "doc_id": _doc_id,
                                "chunk_index": _idx,
                                "chunk_count": len(_chunks),
                                "source": _source_label,
                            },
                        )
                    except Exception as exc:
                        _failures.append(f"❌ Chunk {_idx + 1}: {exc}")
                    _ingest_bar.update()

            if _failures:
                ingest_summary = mo.callout(
                    mo.md(
                        f"**Ingested {len(_chunks) - len(_failures)} of {len(_chunks)} chunks.**\n\n"
                        + "\n".join(_failures)
                    ),
                    kind="danger",
                )
            else:
                ingest_summary = mo.callout(
                    mo.md(
                        f"**🎉 Ingested all {len(_chunks)} chunks**\n\n"
                        f"- Source: `{_source_label}`\n"
                        f"- Document ID: `{_doc_id}`\n"
                        f"- Collection: `{COLLECTION}`\n"
                        f"- Topic: `{topic.value or '(none)'}`"
                    ),
                    kind="success",
                )
            ingested = len(_chunks) - len(_failures)

    ingest_summary
    return (ingested,)


@app.cell(hide_code=True)
def _geval(Any, litellm_chat, re):
    """G-Eval: task intro → criteria → auto-generated CoT steps → form-filling.

    Faithful to the paper's structure. The one departure is scoring: G-Eval weights
    the score by output-token probability, which needs logprobs. We *ask* for them
    (`logprobs=True`) and compute the real weighted score when the backend passes
    them through; otherwise we fall back to the integer the judge wrote. Which path
    ran is recorded as `mode`, so the two are never silently mixed up.
    """

    GEVAL_DIMENSIONS = {
        "faithfulness": {
            "name": "Sticks to the document",
            "hint": "Is every claim backed by your document, with nothing made up?",
            "needs": ("question", "context", "answer"),
            "task": "You will be given a question, retrieved context, and an answer produced from that context.",
            "criteria": (
                "Faithfulness (1-5) — is every factual claim in the answer supported by the "
                "retrieved context? Penalise any claim that is not stated in or entailed by the "
                "context, however plausible it sounds. An answer that correctly says it lacks "
                "information scores high, not low."
            ),
        },
        "answer_relevance": {
            "name": "Answers the question",
            "hint": "Does it address what you actually asked, without dodging or padding?",
            "needs": ("question", "answer"),
            "task": "You will be given a question and an answer.",
            "criteria": (
                "Answer relevance (1-5) — does the answer actually address what was asked? "
                "Penalise evasion, padding, and answers to a different question. Judge only "
                "relevance to the question, not factual correctness."
            ),
        },
        "context_relevance": {
            "name": "Found the right passages",
            "hint": "Were the passages pulled from your document useful for this question?",
            # Deliberately never sees the answer: this scores the *retriever*, so
            # a good answer must not be able to rescue a bad chunk set.
            "needs": ("question", "context"),
            "task": "You will be given a question and the context chunks retrieved for it.",
            "criteria": (
                "Context relevance (1-5) — how useful is the retrieved context for answering "
                "this question? Penalise chunks that are off-topic or redundant, and penalise "
                "a context set that lacks the information needed to answer at all."
            ),
        },
        "coherence": {
            "name": "Clearly written",
            "hint": "Is it well organised and easy to read?",
            "needs": ("question", "answer"),
            "task": "You will be given a question and an answer.",
            "criteria": (
                "Coherence (1-5) — is the answer well organised and clearly written? Judge "
                "structure and readability only, not factual accuracy or relevance."
            ),
        },
    }

    def geval_steps(chat_fn, dimension: str) -> list[str]:
        """G-Eval step 1: have the judge derive its own evaluation steps.

        The steps depend only on the criteria, never on a specific question or
        answer, so callers cache them for the session — which is both what the
        paper describes and what keeps this to 4 calls per answer instead of 8.
        """
        spec = GEVAL_DIMENSIONS[dimension]
        out = chat_fn(
            [
                {
                    "role": "user",
                    "content": (
                        f"{spec['task']}\n\n"
                        f"Evaluation Criteria:\n{spec['criteria']}\n\n"
                        "Write 3-5 concise, numbered evaluation steps a careful grader would "
                        "follow to apply this criterion. Output only the numbered steps."
                    ),
                }
            ],
            temperature=0.0,
        )
        if out.get("error") or not out.get("content"):
            return []
        return [
            line.strip()
            for line in out["content"].splitlines()
            if re.match(r"^\s*\d+[.)]", line)
        ] or [out["content"].strip()]

    def _weighted_from_logprobs(raw: dict) -> float | None:
        """G-Eval's actual scoring: Σ P(d)·d over the score token's distribution.

        Finds the last bare 1-5 digit token in the completion — which is why the
        judge is asked to end with `SCORE: <n>` — and renormalises the 1-5
        candidates in its top_logprobs.

        Note: some backends apply temperature to the logprobs they return, so the
        calibration of this number is backend-dependent. It is still strictly more
        granular than the integer.
        """
        import math

        try:
            content = (raw.get("choices") or [{}])[0].get("logprobs", {}).get("content")
        except (AttributeError, IndexError):
            return None
        if not content:
            return None

        target = None
        for entry in content:
            if (entry.get("token") or "").strip() in {"1", "2", "3", "4", "5"}:
                target = entry
        if target is None:
            return None

        dist = {}
        for cand in target.get("top_logprobs") or []:
            tok = (cand.get("token") or "").strip()
            if tok in {"1", "2", "3", "4", "5"} and cand.get("logprob") is not None:
                dist[int(tok)] = dist.get(int(tok), 0.0) + math.exp(cand["logprob"])
        if not dist:
            return None
        total = sum(dist.values())
        if total <= 0:
            return None
        return sum(d * w for d, w in dist.items()) / total

    def geval_score(
        chat_fn,
        dimension: str,
        *,
        question: str = "",
        context: str = "",
        answer: str = "",
        steps: list[str] | None = None,
    ) -> dict[str, Any]:
        """G-Eval step 2: form-filling. Never raises."""
        spec = GEVAL_DIMENSIONS[dimension]
        blank = {
            "score": None,
            "integer_score": None,
            "reasoning": "",
            "mode": "integer",
            "error": None,
        }

        parts = [spec["task"], "", f"Evaluation Criteria:\n{spec['criteria']}"]
        if steps:
            parts += ["", "Evaluation Steps:\n" + "\n".join(steps)]
        supplied = {"question": question, "context": context, "answer": answer}
        for field in spec["needs"]:
            parts += ["", f"{field.capitalize()}:\n{supplied[field]}"]
        parts += [
            "",
            "Work through the evaluation steps in two or three sentences, then end your "
            "reply with a final line of exactly this form:",
            "SCORE: <a single integer from 1 to 5>",
        ]
        prompt = "\n".join(parts)

        last_error = None
        for _attempt in range(2):  # one retry: judges sometimes omit the SCORE line
            out = chat_fn(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,
            )
            if out.get("error"):
                last_error = out["error"]
                continue
            text = out.get("content") or ""
            match = re.search(r"SCORE:\s*([1-5])", text) or re.search(
                r"\b([1-5])\b(?!.*\b[1-5]\b)", text, re.S
            )
            if not match:
                last_error = "judge did not return a parseable score"
                continue

            integer_score = int(match.group(1))
            weighted = _weighted_from_logprobs(out.get("raw") or {})
            reasoning = re.sub(r"SCORE:\s*[1-5]\s*$", "", text).strip()
            return {
                "score": round(weighted, 3) if weighted is not None else float(integer_score),
                "integer_score": integer_score,
                "reasoning": reasoning,
                "mode": "weighted" if weighted is not None else "integer",
                "error": None,
            }

        return {**blank, "error": last_error or "judge unavailable"}

    return GEVAL_DIMENSIONS, geval_score, geval_steps


@app.cell(hide_code=True)
def _mlflow(
    Any,
    MLFLOW_EXPERIMENT,
    MLFLOW_LOG_CONTENT,
    MLFLOW_TIMEOUT_SECONDS,
    MLFLOW_TRACKING_URI,
    json,
    requests,
    threading,
    time,
):
    """Fail-open MLflow logger over the REST API, authenticated with X-Tapis-Token.

    Ported from `icicle-ai-embed-service/src/app/metrics.py` (sync `requests`
    instead of async `httpx`). Same reasoning applies: talking to the REST API
    directly means no `mlflow` client dependency, and reusing the caller's
    already-validated token means no service account and no refresh logic.

    Every operation is best-effort. MLflow down, pod asleep, token not authorised
    on the pod, network blip — recorded on the session and swallowed, so metric
    logging can never break or delay an answer.

    Divergences from the sibling service, both deliberate:
      * one run per *session* with turns as steps, not one run per request —
        turn-over-turn trends are the entire point here;
      * content (questions, answers, judge rationales) is logged unless
        MLFLOW_LOG_CONTENT=0, whereas the service logs only anonymous shape.
    """

    # MLflow's log-batch caps: 1000 metrics / 100 params / 100 tags per call.
    _BATCH_CAPS = {"metrics": 1000, "params": 100, "tags": 100}

    class EvalSession:
        def __init__(self) -> None:
            self.enabled = bool(MLFLOW_TRACKING_URI)
            self.base = MLFLOW_TRACKING_URI
            self.experiment_id: str | None = None
            self.run_id: str | None = None
            self.last_error: str | None = None
            self.logged_turns = 0
            self.steps_cache: dict[str, list[str]] = {}
            self.geval_mode: str | None = None
            # Two questions asked in quick succession spawn two eval threads; the
            # lock keeps them from each creating their own run for the session.
            self.lock = threading.Lock()

        @property
        def status(self) -> str:
            if not self.enabled:
                return "off"
            if self.last_error:
                return "unreachable"
            return "ok" if self.run_id else "pending"

        def _api(self, method: str, path: str, token: str, **kw) -> dict:
            resp = requests.request(
                method,
                f"{self.base}/api/2.0/mlflow{path}",
                headers={"X-Tapis-Token": token, "Content-Type": "application/json"},
                cookies={"X-Tapis-Token": token},
                timeout=MLFLOW_TIMEOUT_SECONDS,
                allow_redirects=False,
                **kw,
            )
            if 300 <= resp.status_code < 400:
                raise RuntimeError("redirected to login — token not valid for this pod")
            resp.raise_for_status()
            return resp.json() if resp.content else {}

        def _ensure_experiment(self, token: str) -> str | None:
            if self.experiment_id is not None:
                return self.experiment_id
            # Prefer an existing experiment; create only if absent; re-fetch once
            # in case another client won the race.
            try:
                data = self._api(
                    "GET", "/experiments/get-by-name", token,
                    params={"experiment_name": MLFLOW_EXPERIMENT},
                )
                self.experiment_id = data["experiment"]["experiment_id"]
                return self.experiment_id
            except Exception:
                pass
            try:
                data = self._api(
                    "POST", "/experiments/create", token, json={"name": MLFLOW_EXPERIMENT}
                )
                self.experiment_id = data["experiment_id"]
                return self.experiment_id
            except Exception:
                try:
                    data = self._api(
                        "GET", "/experiments/get-by-name", token,
                        params={"experiment_name": MLFLOW_EXPERIMENT},
                    )
                    self.experiment_id = data["experiment"]["experiment_id"]
                    return self.experiment_id
                except Exception as exc:
                    self.last_error = f"could not resolve experiment: {exc}"
                    return None

        def ensure_run(self, token: str, session_id: str, params: dict) -> str | None:
            """Create the session run lazily, on the first turn actually evaluated."""
            if self.run_id or not self.enabled:
                return self.run_id
            with self.lock:
                if self.run_id:  # another eval thread won the race
                    return self.run_id
                return self._create_run(token, session_id, params)

        def _create_run(self, token: str, session_id: str, params: dict) -> str | None:
            experiment_id = self._ensure_experiment(token)
            if experiment_id is None:
                return None
            ts = int(time.time() * 1000)
            try:
                run = self._api(
                    "POST", "/runs/create", token,
                    json={
                        "experiment_id": experiment_id,
                        "start_time": ts,
                        "tags": [
                            {"key": "mlflow.runName", "value": f"chatbook-{session_id[:8]}"},
                            {"key": "session_id", "value": session_id},
                            {"key": "source", "value": "icicle-chatbook"},
                        ],
                    },
                )
                self.run_id = run["run"]["info"]["run_id"]
                self._log_batch(token, params=[
                    {"key": k, "value": str(v)[:5000]} for k, v in params.items()
                ])
                self.last_error = None
            except Exception as exc:
                self.last_error = f"run create failed: {exc}"
            return self.run_id

        def _log_batch(self, token: str, **kinds) -> None:
            """POST /runs/log-batch, chunked to stay under the API's caps."""
            for kind, items in kinds.items():
                cap = _BATCH_CAPS[kind]
                for i in range(0, len(items), cap):
                    self._api(
                        "POST", "/runs/log-batch", token,
                        json={"run_id": self.run_id, kind: items[i : i + cap]},
                    )

        def log_turn(
            self, token: str, step: int, metrics: dict, content: dict | None
        ) -> None:
            """Log one turn's metrics (and optionally its content). Never raises."""
            if not self.enabled or not self.run_id:
                return
            ts = int(time.time() * 1000)
            try:
                self._log_batch(token, metrics=[
                    {"key": k, "value": float(v), "timestamp": ts, "step": step}
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                ])
                if content and MLFLOW_LOG_CONTENT:
                    self._log_content(token, step, content)
                self.logged_turns += 1
                self.last_error = None
            except Exception as exc:
                self.last_error = f"log failed: {exc}"
            self._mark_finished(token)

        def _mark_finished(self, token: str) -> None:
            """Close the run after every turn, re-closing it as later turns land.

            A notebook session has no shutdown event we can hook — the tab is
            closed, the kernel is killed, the pod scales down — so a run left open
            until "the end" is a run left RUNNING forever, with its duration
            climbing long after the last question. Terminating on each turn keeps
            the run's end_time honest whatever happens next: MLflow gates further
            logging on lifecycle_stage (active vs deleted), not on status, so a
            FINISHED run still accepts the next turn's metrics.
            """
            if not self.run_id:
                return
            try:
                self._api(
                    "POST", "/runs/update", token,
                    json={
                        "run_id": self.run_id,
                        "status": "FINISHED",
                        "end_time": int(time.time() * 1000),
                    },
                )
            except Exception:
                # Fail-open like everything else here: a run left RUNNING is
                # cosmetic, and must never surface as a logging error.
                pass

        def _log_content(self, token: str, step: int, content: dict) -> None:
            """Questions, answers, chunks and judge rationales.

            Tries the artifact proxy first; many deployments don't enable it, so
            falls back to truncated run tags. Both paths are best-effort.
            """
            body = json.dumps(content, indent=2).encode("utf-8")
            try:
                resp = requests.post(
                    f"{self.base}/api/2.0/mlflow-artifacts/artifacts"
                    f"/{self.run_id}/turns/{step:03d}.json",
                    headers={"X-Tapis-Token": token},
                    cookies={"X-Tapis-Token": token},
                    data=body,
                    timeout=MLFLOW_TIMEOUT_SECONDS,
                    allow_redirects=False,
                )
                if resp.status_code < 300:
                    return
            except requests.RequestException:
                pass

            tags = [{"key": f"turn_{step:03d}_question", "value": str(content.get("question", ""))[:4900]}]
            for dim, detail in (content.get("scores") or {}).items():
                reasoning = (detail or {}).get("reasoning") or ""
                if reasoning:
                    tags.append(
                        {"key": f"turn_{step:03d}_{dim}_reasoning", "value": reasoning[:4900]}
                    )
            self._log_batch(token, tags=tags)

    return (EvalSession,)


@app.cell(hide_code=True)
def _chat_section(mo):
    mo.md("""
    ## 💬 Ask questions about your document

    Every question runs the same RAG loop:
    1. Your question → embed service (as a query)
    2. Retrieve top-K nearest chunks from your collection
    3. Stitch chunks + question into a grounded prompt
    4. Send to the chat model and read back the answer

    There are **two ways to ask, into one conversation**. Type a question and send
    it straight away, or — while you're reading a long answer — jot questions into
    the 📓 study notes and send them as a batch when you're done. The **📚 Study
    mode** switch just shows or hides the notes; nothing is lost either way.
    """)
    return


@app.cell(hide_code=True)
def _study_state(mo):
    # Notes pad items: {"id", "kind": "question" | "note", "text"}.
    #
    # allow_self_loops=True is required here: the cell that renders the pad also
    # owns the per-item ✖ buttons, so it has to re-run after its own setter fires
    # or the list goes stale. It terminates — a re-run only rebuilds the buttons,
    # it never calls the setter again.
    get_notes, set_notes = mo.state([], allow_self_loops=True)

    # The one conversation. Both ways of asking append here:
    # [{"question", "answer", "results", "error", "topic", "top_k"}].
    get_history, set_history = mo.state([])

    # Batched questions staged behind the confirmation gate, or None when idle.
    # allow_self_loops=True because `_study_confirm` both renders the gate and
    # clears it: without it, clearing from that cell never re-renders it and the
    # dialog lingers after the batch has already been sent.
    get_pending, set_pending = mo.state(None, allow_self_loops=True)

    # The single work order the executor consumes: {"questions": [...], "mode": ...}
    # or None. Every entry point — composer, starter chip, confirmed batch — sets
    # this, and setting it is what re-runs the executor.
    get_work, set_work = mo.state(None)

    # turn_id -> {"status", "scores", "error"}. Written from the background eval
    # thread; marimo re-runs the transcript when it lands (see _eval_runner).
    get_evals, set_evals = mo.state({})
    return (
        get_evals,
        get_history,
        get_notes,
        get_pending,
        get_work,
        set_evals,
        set_history,
        set_notes,
        set_pending,
        set_work,
    )


@app.cell(hide_code=True)
def _note_form(mo, set_notes, uuid):
    ASK_LATER = "❓ Ask later"

    # A form (rather than a bare text area + button) so `clear_on_submit` empties
    # the box for us — there's no way to reset a text area's value from Python.
    # `.batch()` lays the widgets out through a markdown template; `mo.ui.dictionary`
    # would render them inside a JSON-style tree with the raw keys showing.
    note_form = (
        mo.md("{text}\n\n{kind}")
        .batch(
            text=mo.ui.text_area(
                placeholder="Type a question or a note…",
                rows=3,
                full_width=True,
            ),
            kind=mo.ui.radio(
                options=[ASK_LATER, "📝 Just a note"],
                value=ASK_LATER,
                inline=True,
            ),
        )
        .form(
            submit_button_label="➕ Add",
            clear_on_submit=True,
            bordered=False,
            validate=lambda v: (
                None if (v or {}).get("text", "").strip() else "Write something first."
            ),
            on_change=lambda v: (
                set_notes(
                    lambda ns: ns
                    + [
                        {
                            "id": str(uuid.uuid4()),
                            "kind": "question" if v["kind"] == ASK_LATER else "note",
                            "text": v["text"].strip(),
                        }
                    ]
                )
                if v and v.get("text", "").strip()
                else None
            ),
        )
    )
    return (note_form,)


@app.cell(hide_code=True)
def _study_pad(get_notes, mo, set_notes, set_pending):
    notes = get_notes()
    queued_questions = [n for n in notes if n["kind"] == "question"]

    # mo.ui.array is what makes a run-time-determined number of elements work:
    # it clones each button (preserving on_change) and routes frontend updates
    # back to the right one.
    remove_buttons = mo.ui.array(
        [
            mo.ui.run_button(
                label="✖",
                tooltip="Remove from pad",
                on_change=(
                    lambda _v, note_id=n["id"]: set_notes(
                        lambda ns: [x for x in ns if x["id"] != note_id]
                    )
                ),
            )
            for n in notes
        ]
    )

    def _rows(kind):
        return [
            mo.hstack(
                [mo.md(n["text"]), remove_buttons[i]],
                widths=[1, 0],
                align="start",
                gap=0.5,
            )
            for i, n in enumerate(notes)
            if n["kind"] == kind
        ]

    kept_notes = [n for n in notes if n["kind"] == "note"]
    sections = []
    if queued_questions:
        sections += [
            mo.md(f"**❓ To ask ({len(queued_questions)})**"),
            *_rows("question"),
        ]
    if kept_notes:
        sections += [mo.md(f"**📝 My notes ({len(kept_notes)})**"), *_rows("note")]
    pad_rows = (
        mo.vstack(sections, gap=0.25)
        if sections
        else mo.md(
            "_Jot down questions as you read, then ask them all at once. "
            "Notes just stay here for you._"
        )
    )

    ask_mode = mo.ui.checkbox(label="Answer them together in one reply")

    send_button = mo.ui.run_button(
        label=(
            f"📤 Ask {len(queued_questions)} question"
            f"{'' if len(queued_questions) == 1 else 's'}"
        ),
        kind="success",
        disabled=not queued_questions,
        full_width=True,
        tooltip="You'll get to review them before anything is sent.",
        # The questions move out of the pad and into the gate *here*, on click —
        # not in the executor. The executor must never call set_notes, because
        # allow_self_loops=True would then make it re-run itself forever.
        on_change=lambda _v: (
            set_notes(lambda ns: [n for n in ns if n["kind"] != "question"]),
            set_pending(
                {
                    "questions": [n["text"] for n in queued_questions],
                    "mode": "Ask together" if ask_mode.value else "Ask separately",
                }
            ),
        ),
    )

    # The combine option only means something with two or more questions, and the
    # send button only once there's something to send.
    send_block = mo.vstack(
        ([ask_mode] if len(queued_questions) > 1 else [])
        + ([send_button] if queued_questions else []),
        gap=0.5,
    )
    return ask_mode, pad_rows, remove_buttons, send_block, send_button


@app.cell(hide_code=True)
def _study_confirm(get_pending, mo, set_notes, set_pending, set_work, uuid):
    pending = get_pending()

    # Hands the staged batch to the work order and clears the gate. The executor
    # re-runs because `work` changed — not because this button was clicked — which
    # is what lets the composer and the starter chips share the same executor.
    # Both handlers read the gate through `get_pending()` rather than closing over
    # the `pending` above: whatever the render state, once the gate is empty a
    # second click is a no-op, so a batch can never be sent twice.
    def _confirm():
        staged = get_pending()
        if staged is None:
            return
        set_pending(None)
        set_work(staged)

    def _cancel():
        staged = get_pending()
        if staged is None:
            return
        set_pending(None)
        set_notes(
            lambda ns: ns
            + [
                {"id": str(uuid.uuid4()), "kind": "question", "text": q}
                for q in staged.get("questions", [])
            ]
        )

    confirm_button = mo.ui.run_button(
        label="✅ Yes, send",
        kind="success",
        on_change=lambda _v: _confirm(),
    )

    cancel_button = mo.ui.run_button(
        label="✖️ Cancel",
        on_change=lambda _v: _cancel(),
    )

    if pending is None:
        confirm_block = mo.md("")
    else:
        listed = "\n".join(
            f"{i}. {q}" for i, q in enumerate(pending["questions"], start=1)
        )
        mode_note = (
            "Each question gets its own answer."
            if pending["mode"] == "Ask separately"
            else "They'll be answered together in one reply."
        )
        confirm_block = mo.callout(
            mo.vstack(
                [
                    mo.md(
                        f"**Send {len(pending['questions'])} question"
                        f"{'' if len(pending['questions']) == 1 else 's'} to the model?**\n\n"
                        f"{listed}\n\n_{mode_note}_"
                    ),
                    mo.hstack(
                        [confirm_button, cancel_button], justify="start", gap=0.5
                    ),
                ],
                gap=0.5,
            ),
            kind="warn",
        )
    return cancel_button, confirm_block, confirm_button


@app.cell(hide_code=True)
def _eval_session(EvalSession, uuid):
    # No volatile dependencies, so this is constructed once and survives every
    # re-render — which is what makes "one MLflow run per session" hold.
    eval_session = EvalSession()
    session_id = str(uuid.uuid4())
    return eval_session, session_id


@app.cell(hide_code=True)
def _eval_runner(
    GEVAL_DIMENSIONS,
    MLFLOW_ENABLED,
    eval_session,
    geval_score,
    geval_steps,
    litellm_chat,
    mo,
    session_id,
    set_evals,
    time,
):
    def evaluate_turns(token, turns, *, judge_model, params, first_step):
        """Judge a batch of turns and log them. Runs inside mo.Thread.

        Calling a marimo state setter from an mo.Thread is supported and is what
        makes this work without polling: marimo marks the reading cells stale and
        re-runs them (see `register_state_update` in marimo/_runtime/runtime.py),
        so scores appear in the transcript on their own.
        """
        thread = mo.current_thread()

        def chat_fn(messages, **kw):
            return litellm_chat(token, judge_model, messages, max_tokens=700, **kw)

        def _mark(turn_id, payload):
            set_evals(lambda e: {**e, turn_id: payload})

        for offset, turn in enumerate(turns):
            if thread.should_exit:
                return
            if turn.get("error"):
                continue  # nothing to judge

            turn_id = turn["id"]
            _mark(turn_id, {"status": "running", "scores": {}, "error": None})

            scores, judge_started = {}, time.perf_counter()
            for dimension in GEVAL_DIMENSIONS:
                if thread.should_exit:
                    return
                # Steps depend only on the criteria, so they're generated once per
                # session — the paper's design, and it halves the call count.
                if dimension not in eval_session.steps_cache:
                    eval_session.steps_cache[dimension] = geval_steps(chat_fn, dimension)
                scores[dimension] = geval_score(
                    chat_fn,
                    dimension,
                    question=turn.get("question", ""),
                    context=turn.get("context", ""),
                    answer=turn.get("answer", ""),
                    steps=eval_session.steps_cache[dimension],
                )
            judge_ms = (time.perf_counter() - judge_started) * 1000

            graded = [d for d in scores.values() if d.get("score") is not None]
            if graded and eval_session.geval_mode is None:
                eval_session.geval_mode = graded[0]["mode"]

            _mark(
                turn_id,
                {
                    "status": "done" if graded else "failed",
                    "scores": scores,
                    "error": None if graded else "judge returned no usable scores",
                },
            )

            if not (MLFLOW_ENABLED and eval_session.enabled):
                continue

            eval_session.ensure_run(
                token, session_id, {**params, "geval_mode": eval_session.geval_mode}
            )
            retrieval = [
                r.get("score")
                for r in (turn.get("results") or [])
                if isinstance(r.get("score"), (int, float))
            ]
            timings = turn.get("timings") or {}
            metrics = {
                **{d: v["score"] for d, v in scores.items() if v.get("score") is not None},
                **{
                    f"{d}_integer": v["integer_score"]
                    for d, v in scores.items()
                    if v.get("integer_score") is not None
                },
                "chunks_returned": len(turn.get("results") or []),
                "answer_chars": len(turn.get("answer") or ""),
                "latency_judge_ms": judge_ms,
                **{f"latency_{k}": v for k, v in timings.items()},
            }
            if retrieval:
                metrics["retrieval_score_mean"] = sum(retrieval) / len(retrieval)
                metrics["retrieval_score_max"] = max(retrieval)
                metrics["retrieval_score_min"] = min(retrieval)

            eval_session.log_turn(
                token,
                first_step + offset,
                metrics,
                {
                    "question": turn.get("question", ""),
                    "answer": turn.get("answer", ""),
                    "chunks": turn.get("results") or [],
                    "scores": scores,
                    "config": params,
                },
            )

    return (evaluate_turns,)


@app.cell(hide_code=True)
def _composer(mo, set_work):
    # A form (not a bare text area + button) because `clear_on_submit` is the only
    # way to empty an input from Python.
    composer = mo.ui.form(
        mo.ui.text_area(
            placeholder="Ask about the document…",
            rows=2,
            full_width=True,
        ),
        submit_button_label="Ask",
        clear_on_submit=True,
        bordered=False,
        validate=lambda v: None if (v or "").strip() else "Type a question first.",
        on_change=lambda v: (
            set_work({"questions": [v.strip()], "mode": "Ask separately"})
            if v and v.strip()
            else None
        ),
    )

    remember = mo.ui.switch(
        value=True,
        label="Remember conversation",
    )

    # Unlike mo.ui.chat's prompt chips, these *ask* the question outright rather
    # than pre-filling a box.
    STARTERS = [
        "Summarize the document in two sentences.",
        "What are the main components mentioned?",
        "List any specific technologies referenced.",
    ]
    starter_chips = mo.ui.array(
        [
            mo.ui.run_button(
                label=text,
                on_change=(
                    lambda _v, q=text: set_work(
                        {"questions": [q], "mode": "Ask separately"}
                    )
                ),
            )
            for text in STARTERS
        ]
    )
    return composer, remember, starter_chips


@app.cell(hide_code=True)
def _study_run(
    COLLECTION,
    answer_question,
    chat_model,
    eval_enabled,
    evaluate_turns,
    get_history,
    get_work,
    judge_model,
    max_chunk_tokens,
    mo,
    overlap_tokens,
    remember,
    set_history,
    set_work,
    token,
    top_k,
    topic,
    uuid,
):
    # Driven by the `work` state rather than any one button's .value — that is what
    # lets the composer, the starter chips and a confirmed batch all land here.
    work = get_work()
    mo.stop(not work, mo.md(""))
    mo.stop(
        token is None,
        mo.callout(
            "❌ Token is no longer valid — revalidate it above and try again.",
            kind="danger",
        ),
    )

    questions = work["questions"]
    if work["mode"] == "Ask together":
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
        batch = [
            "Answer each of the following questions about the document, "
            "labelling each answer with its number:\n" + numbered
        ]
    else:
        batch = list(questions)

    # Snapshot the history once: every question in this batch sees the same prior
    # conversation, not each other's answers.
    prior = get_history() if remember.value else None
    _first_step = len(get_history())

    def _run_one(question_text):
        return {
            "id": str(uuid.uuid4()),  # stable handle for matching eval results back
            "question": question_text,
            # Snapshot topic/top_k with the turn so an old answer keeps showing the
            # settings it was actually retrieved under.
            "topic": topic.value,
            "top_k": top_k.value,
            **answer_question(
                token,
                question_text,
                collection=COLLECTION,
                topic=topic.value,
                top_k=top_k.value,
                chat_model=chat_model.value,
                history=prior,
            ),
        }

    _turns = []
    if len(batch) == 1:
        with mo.status.spinner(title="Thinking…"):
            _turns.append(_run_one(batch[0]))
    else:
        with mo.status.progress_bar(
            total=len(batch), title="Answering your questions", remove_on_exit=True
        ) as _bar:
            for question_text in batch:
                _turns.append(_run_one(question_text))
                _bar.update()

    set_history(lambda h: h + _turns)

    # Judge in the background: the answers are already rendered by the time this
    # starts, and the thread pushes scores back through `set_evals`.
    if eval_enabled.value and any(not t.get("error") for t in _turns):
        mo.Thread(
            target=evaluate_turns,
            args=(token, _turns),
            kwargs={
                "judge_model": judge_model.value,
                "params": {
                    "chat_model": chat_model.value,
                    "judge_model": judge_model.value,
                    "top_k": top_k.value,
                    "max_chunk_tokens": max_chunk_tokens.value,
                    "overlap_tokens": overlap_tokens.value,
                    "topic": topic.value,
                    "collection": COLLECTION,
                    "memory_enabled": remember.value,
                    "dimensions": "faithfulness,answer_relevance,context_relevance,coherence",
                },
                "first_step": _first_step,
            },
            daemon=True,
        ).start()

    # Safe to clear from this cell: `work` keeps allow_self_loops=False, so setting
    # it here does not re-trigger this cell.
    set_work(None)
    return


@app.cell(hide_code=True)
def _study_transcript(
    GEVAL_DIMENSIONS, COLLECTION, format_answer_md, get_evals, get_history, mo
):
    _turns = get_history()
    _evals = get_evals()

    def _score_row(turn):
        """The 🧪 badge under an answer — arrives on its own from the eval thread."""
        entry = _evals.get(turn.get("id"))
        if not entry:
            return ""
        if entry["status"] == "running":
            return "\n\n🧪 _evaluating…_"
        if entry["status"] == "failed":
            return f"\n\n🧪 _eval unavailable — {entry.get('error', 'unknown')}_"

        def _dot(score):
            return "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"

        scored = {
            d: round(v["score"], 1)
            for d, v in entry["scores"].items()
            if v.get("score") is not None
        }
        if not scored:
            return "\n\n🧪 _eval unavailable — no scores came back_"

        summary = " · ".join(
            f"{_dot(sc)} {GEVAL_DIMENSIONS[d]['name']} **{sc:g}/5**"
            for d, sc in scored.items()
        )
        # Plain-language meaning of each check first, then the judge's own words.
        explained = "\n".join(
            f"- {_dot(scored[d])} **{GEVAL_DIMENSIONS[d]['name']} — {scored[d]:g}/5.** "
            f"{GEVAL_DIMENSIONS[d]['hint']}"
            + (
                f"\n  <br>_Judge: {entry['scores'][d]['reasoning']}_"
                if entry["scores"][d].get("reasoning")
                else ""
            )
            for d in scored
        )
        block = (
            f"\n\n---\n\n🧪 **Answer check** &nbsp;{summary}"
            "\n\n<details><summary>What do these scores mean?</summary>\n\n"
            "An AI judge rates each answer from 1 (poor) to 5 (great) on four "
            "questions. 🟢 4–5 good · 🟡 3–3.9 so-so · 🔴 below 3 worth double-checking.\n\n"
            f"{explained}\n\n</details>"
        )
        return block

    def _render(turn):
        if turn.get("error"):
            return mo.callout(
                mo.md(f"**💬 {turn['question']}**\n\n{turn['error']}"), kind="danger"
            )
        return mo.callout(
            mo.md(
                f"**💬 {turn['question']}**\n\n"
                + format_answer_md(
                    turn["answer"],
                    turn["results"],
                    collection=COLLECTION,
                    topic=turn.get("topic"),
                    top_k=turn.get("top_k"),
                )
                + _score_row(turn)
            ),
            kind="neutral",
        )

    if not _turns:
        transcript_block = mo.callout(
            mo.md(
                "**Nothing asked yet.**\n\n"
                "Ask a question below — or turn on **📚 Study mode** to jot down "
                "questions while you read and ask them all at once."
            ),
            kind="neutral",
        )
    else:
        # Newest first: marimo has no scroll-to-bottom, so the answer you just
        # asked for has to already be at the top of the pane.
        latest = _render(_turns[-1])
        older = list(reversed(_turns[:-1]))
        if older:
            # Keys must be unique — the same question can be asked twice — so each
            # is numbered with its turn index.
            transcript_block = mo.vstack(
                [
                    latest,
                    mo.accordion(
                        {
                            f"{len(_turns) - i - 1} · 💬 {t['question']}": _render(t)
                            for i, t in enumerate(older)
                        },
                        multiple=True,
                    ),
                ],
                gap=0.75,
            )
        else:
            transcript_block = latest
    return (transcript_block,)


@app.cell(hide_code=True)
def _layout_mode(mo):
    # In its own cell so toggling re-runs only the layout below. History lives in
    # `mo.state`, so the flip is a pure re-render — nothing is lost, which is
    # exactly what a stateful chat widget inside tabs could not do.
    layout_mode = mo.ui.switch(label="📚 Study mode", value=False)
    return (layout_mode,)


@app.cell(hide_code=True)
def _chat_surface(
    composer,
    confirm_block,
    eval_session,
    get_evals,
    get_history,
    get_notes,
    GEVAL_DIMENSIONS,
    ingested,
    layout_mode,
    MLFLOW_ENABLED,
    mo,
    note_form,
    pad_rows,
    remember,
    send_block,
    starter_chips,
    token,
    tracer,
    transcript_block,
):
    def session_markdown() -> bytes:
        lines = ["# ICICLE AI Chatbook — study session", ""]
        turns = get_history()
        if turns:
            lines += ["## Questions & answers", ""]
            evals = get_evals()
            for t in turns:
                lines += [
                    f"### 💬 {t['question']}",
                    "",
                    t.get("error") or t.get("answer", ""),
                    "",
                ]
                entry = evals.get(t.get("id")) or {}
                scored = [
                    f"{GEVAL_DIMENSIONS[d]['name']} {round(v['score'], 1):g}/5"
                    for d, v in (entry.get("scores") or {}).items()
                    if v.get("score") is not None
                ]
                if scored:
                    lines += [f"_🧪 Answer check — {' · '.join(scored)}_", ""]
        # Same plain style as "Questions & answers": a heading and bullets, no icons.
        kept = get_notes()
        for heading, kind in (("Notes", "note"), ("Questions not yet asked", "question")):
            items = [n["text"] for n in kept if n["kind"] == kind]
            if items:
                lines += [f"## {heading}", ""] + [f"- {text}" for text in items] + [""]
        return "\n".join(lines).encode("utf-8")

    download_button = mo.download(
        data=session_markdown,
        filename="study-session.md",
        mimetype="text/markdown",
        label="⬇️ Export session (.md)",
    )

    if not MLFLOW_ENABLED or not eval_session.enabled:
        _mlflow_status = ""
    elif eval_session.status == "unreachable":
        _mlflow_status = f"⬡ _MLflow unreachable — {eval_session.last_error}_"
    elif eval_session.run_id:
        _mlflow_status = f"⬡ _MLflow: {eval_session.logged_turns} turns logged_"
    else:
        _mlflow_status = "⬡ _MLflow ready_"

    # Tracing is a separate surface with a separate failure mode — it can be off,
    # or on and unable to reach the pod — so it gets its own indicator rather than
    # being folded into the metrics one.
    if tracer.status == "off":
        pass
    elif tracer.status == "error":
        _mlflow_status += f" · 🕸 _tracing failed — {tracer.error}_"
    elif tracer.status == "ok":
        _mlflow_status += " · 🕸 _tracing on_"
    else:
        _mlflow_status += " · 🕸 _tracing pending_"

    if token is None:
        chat_surface = mo.callout(
            mo.md(
                "❌ **Cannot chat without a valid Tapis token.** "
                "Paste a token above and re-ingest your document."
            ),
            kind="danger",
        )
    elif ingested == 0:
        chat_surface = mo.callout(
            "Ingest a document above before asking questions.", kind="neutral"
        )
    else:
        # Starter chips only earn their space before the first question.
        _starters = (
            mo.hstack(list(starter_chips), justify="start", wrap=True, gap=0.5)
            if not get_history()
            else mo.md("")
        )
        _conversation = mo.vstack(
            [
                mo.hstack(
                    [mo.md("#### 💬 Chat"), layout_mode],
                    justify="space-between",
                    align="center",
                ),
                transcript_block,
                # The gate sits above the composer so a staged batch is never
                # hidden, in either view.
                confirm_block,
                _starters,
                composer,
                mo.hstack(
                    [remember, download_button, mo.md(_mlflow_status)],
                    justify="start",
                    gap=1,
                    align="center",
                ),
            ],
            gap=0.5,
        )

        if not layout_mode.value:
            chat_surface = _conversation
        else:
            chat_surface = mo.hstack(
                [
                    _conversation,
                    mo.vstack(
                        [
                            mo.md(
                                "#### 📓 Study notes\n\n"
                                "_Queue follow-up questions while you read._"
                            ),
                            note_form,
                            pad_rows,
                            send_block,
                        ],
                        gap=0.5,
                    ),
                ],
                widths=[2, 1],
                align="start",
                gap=1.5,
            )

    chat_surface
    return (chat_surface,)


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
