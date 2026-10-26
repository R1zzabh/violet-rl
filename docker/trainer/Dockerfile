FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY backend/pyproject.toml backend/README.md ./
COPY backend/violet ./violet
RUN pip install --upgrade pip \
    && pip install ".[api]" \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1" \
    && addgroup --system --gid 10003 trainer \
    && adduser --system --uid 10003 --ingroup trainer trainer \
    && mkdir -p /app/runs && chown -R trainer:trainer /app
USER 10003:10003
CMD ["python", "-m", "violet.range.trainer_main"]
