# ============================================================
#  Stage 0 — base: Shared OS setup and security
# ============================================================
FROM python:3.12-slim AS base

# Python best practices & virtualenv path
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv" \
    UV_PROJECT_ENVIRONMENT="/opt/venv"

WORKDIR /app

# Security: Shared non-root user setup
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser /app

# ============================================================
#  Stage 1 — deps: install uv and build dependencies
# ============================================================
FROM base AS deps

# UV optimizations & CPU-only torch index
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" \
    PATH="/root/.local/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT="/opt/venv"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libpq-dev \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock pyproject.toml ./

# ============================================================
#  Stage 2 — dev: full deps + hot-reload
# ============================================================
FROM deps AS dev

# Install ALL dependencies with cache mount and strip
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project && \
    find /opt/venv -type f -name "*.a" -delete && \
    rm -rf /opt/venv/include

# Source code is bind-mounted in dev, but copied for standalone use
COPY . .

USER appuser
EXPOSE 8000
CMD ["fastapi", "dev", "src/api/main.py", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================
#  Stage 3 — builder: production deps only
# ============================================================
FROM deps AS builder

# Install production dependencies and strip virtualenv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev && \
    find /opt/venv -type f -name "*.a" -delete && \
    rm -rf /opt/venv/include

# ============================================================
#  Stage 4 — runner: minimal production image
# ============================================================
FROM base AS runner

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only the virtualenv and source code
COPY --from=builder /opt/venv /opt/venv
COPY . /app

# Final cleanup of non-runtime files
RUN find . -type d -name "__pycache__" -exec rm -rf {} + && \
    rm -rf tests/ .pytest_cache/ .ruff_cache/

USER appuser
EXPOSE 8000
CMD ["fastapi", "run", "src/api/main.py", "--port", "8000"]
