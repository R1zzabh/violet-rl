# docker/backend/Dockerfile — VIOLET

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- deps ----
COPY backend/pyproject.toml backend/README.md ./
# Optional: a lockfile (poetry.lock / requirements.txt) would make this reproducible.
# For now install the package + optional deps we need in production.
RUN pip install --upgrade pip \
    && pip install ".[api]" \
    && pip install ".[dev]" \
    && rm -rf /root/.cache/pip

# ---- app ----
COPY backend/violet ./violet
COPY backend/tests ./tests
COPY backend/scripts ./scripts
COPY backend/configs ./configs

# Non-root user
RUN addgroup --system violet \
    && adduser --system --ingroup violet violet \
    && chown -R violet:violet /app
USER violet

EXPOSE 8000

# Default: run the API server. Override for training jobs.
CMD ["sh", "-c", "uvicorn violet.api.app:app --host 0.0.0.0 --port 8000 --log-level info"]
