# ============================================================
#  Stage 0 — base: shared OS + non-root user
# ============================================================
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv" \
    UV_PROJECT_ENVIRONMENT="/opt/venv"

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

# ============================================================
#  Stage 1 — deps: install uv + system build tools
# ============================================================
FROM base AS deps

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libpq-dev \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Lockfile copied first — layer cache is only invalidated when
# lockfile changes, not on every source code change.
COPY uv.lock pyproject.toml ./

# ============================================================
#  Stage 2 — dev: all deps, hot-reload entrypoint
# ============================================================
FROM deps AS dev

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project && \
    find /opt/venv -type f -name "*.a" -delete && \
    rm -rf /opt/venv/include

# Source is bind-mounted in docker compose. Copying here makes
# the image usable standalone without a volume mount.
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000
CMD ["fastapi", "dev", "src/api/main.py", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================
#  Stage 3 — builder: production deps only
# ============================================================
FROM deps AS builder

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev && \
    find /opt/venv -type f -name "*.a" -delete && \
    rm -rf /opt/venv/include

# ============================================================
#  Stage 4 — runner: minimal production image
# ============================================================
FROM base AS runner

# Only the runtime shared library — no compiler or build tools
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# --chown ensures appuser can write to /app/storage
# at runtime without permission errors.
COPY --chown=appuser:appuser . /app

# Remove non-runtime artifacts. .dockerignore is the first line of
# defence — this is a safety net for anything that slips through.
RUN find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /app/tests /app/.pytest_cache /app/.ruff_cache /app/.env

# Pre-create storage dir so the app can write files immediately.
RUN mkdir -p /app/storage && chown appuser:appuser /app/storage

USER appuser
EXPOSE 8000

# Baked-in healthcheck works on any runtime: Render, VPS, K8s.
# Uses stdlib only — no curl dependency in the production image.
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["fastapi", "run", "src/api/main.py", "--port", "8000"]