# docker/backend/Dockerfile — VIOLET
#
# Build context is the REPOSITORY ROOT (see docker-compose.yml):
#   docker build -f docker/backend/Dockerfile .
#
# There is no backend/configs directory in this repo: all scenario/config
# data lives in code (violet.scenarios) and runs/ artifacts are mounted at
# runtime. Nothing is copied from a configs path.
#
# The API image contains runtime dependencies only. Training dependencies live
# in the isolated CPU worker image.

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- deps ----
# Copy the package source before installing so hatchling can build it.
COPY backend/pyproject.toml backend/README.md ./
COPY backend/violet ./violet
COPY backend/migrations ./migrations
COPY backend/scripts/provision_runtime_role.py ./scripts/provision_runtime_role.py
COPY backend/scripts/provision_sqli_runtime_role.py ./scripts/provision_sqli_runtime_role.py
COPY backend/alembic-control.ini ./alembic-control.ini
COPY backend/alembic-sqli.ini ./alembic-sqli.ini

RUN pip install --upgrade pip \
    && pip install ".[api]" \
    && rm -rf /root/.cache/pip

# Non-root user
RUN addgroup --system --gid 10001 violet \
    && adduser --system --uid 10001 --ingroup violet violet \
    && chown -R violet:violet /app
USER violet

EXPOSE 8000

# Default: run the API server. Override for training jobs.
CMD ["sh", "-c", "uvicorn violet.api.app:app --host 0.0.0.0 --port 8000 --log-level info"]
