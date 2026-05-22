# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — builder: resolve and install dependencies into a self-contained
# virtual environment using uv (fast, reproducible from uv.lock).
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Copy the uv binary from its official image — no curl/install script needed.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first (cached layer) using only the lockfiles, so app
# code changes don't bust the dependency cache.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------------------
# Stage 2 — runtime: slim image with only the venv + app, runs as non-root.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Put the venv on PATH so "marimo" resolves without uv at runtime.
#
# HOME and the XDG dirs point at /tmp so marimo can always write its
# config/cache/data — Tapis Pods (and k8s generally) may run the container as
# an arbitrary UID and/or with a read-only root filesystem; /tmp stays
# writable in both cases, so this works regardless of the runtime UID.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    XDG_CONFIG_HOME=/tmp/.config \
    XDG_CACHE_HOME=/tmp/.cache \
    XDG_DATA_HOME=/tmp/.local/share

# Unprivileged user — never run the app as root.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

# Bring in the prebuilt virtual environment and only the files the app needs.
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser notebooks/ ./notebooks/
COPY --chown=appuser:appuser assets/ ./assets/

USER appuser

EXPOSE 8080

# Serve the notebook as a read-only app, reachable from outside the container.
# The Tapis token is supplied by the end user in the UI at runtime — nothing
# secret is baked into this image.
ENTRYPOINT ["marimo", "run", "notebooks/rag_chat_marimo.py", \
            "--host", "0.0.0.0", "--port", "8080", \
            "--headless", "--no-token"]
