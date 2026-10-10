FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN pip install --upgrade pip \
    && pip install "httpx>=0.27" "pydantic>=2.6" \
    && addgroup --system --gid 10001 executor \
    && adduser --system --uid 10001 --ingroup executor executor \
    && mkdir -p /spool/jobs /spool/processing /spool/results /spool/records \
    && chown -R executor:executor /app /spool \
    && chmod 2770 /spool /spool/jobs /spool/processing /spool/results /spool/records

COPY backend/violet/__init__.py backend/violet/_version.py ./violet/
COPY backend/violet/range ./violet/range

USER 10001:10001
CMD ["python", "-m", "violet.range.executor_main"]
