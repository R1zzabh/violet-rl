# docker/backend/Dockerfile — VIOLET
#
# Build context is the REPOSITORY ROOT (see docker-compose.yml):
#   docker build -f docker/backend/Dockerfile .
#
# There is no backend/configs directory in this repo: all scenario/config
# data lives in code (violet.scenarios) and runs/ artifacts are mounted at
# runtime. Nothing is copied from a configs path.
#
# Note on the pip installs below: the package itself is copied first and
# installed non-editable; `.[dev]` provides the test tooling so the image can
# also run the pytest suite (`docker run violet-backend pytest tests -q`).

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
COPY backend/.wheelhouse /tmp/torch-wheelhouse

RUN pip install --upgrade pip \
    && if ls /tmp/torch-wheelhouse/*.whl >/dev/null 2>&1; then \
         pip install /tmp/torch-wheelhouse/*.whl; \
       else \
         pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch>=2.1"; \
       fi \
    && pip install ".[api,dev,rl]" \
    && rm -rf /tmp/torch-wheelhouse \
    && rm -rf /root/.cache/pip

# ---- app (tests + scripts only; package already installed above) ----
COPY backend/tests ./tests
COPY backend/scripts ./scripts

# Non-root user
RUN addgroup --system --gid 10001 violet \
    && adduser --system --uid 10001 --ingroup violet violet \
    && chown -R violet:violet /app
USER violet

EXPOSE 8000

# Default: run the API server. Override for training jobs.
CMD ["sh", "-c", "uvicorn violet.api.app:app --host 0.0.0.0 --port 8000 --log-level info"]
