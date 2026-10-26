FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN pip install --upgrade pip \
    && pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "sqlalchemy>=2.0" "psycopg2-binary>=2.9" "pydantic>=2.6" \
    && addgroup --system --gid 10001 sqli-share \
    && adduser --system --uid 10004 --ingroup sqli-share sqli-target \
    && mkdir -p /control/inbox /control/results /var/lib/violet/sqli-evidence \
    && chown -R sqli-target:sqli-share /app /control/results /var/lib/violet/sqli-evidence \
    && chmod 0755 /control/inbox

COPY backend/violet/__init__.py backend/violet/_version.py ./violet/
COPY backend/violet/range/__init__.py backend/violet/range/spool.py ./violet/range/
COPY backend/violet/sqli_lab ./violet/sqli_lab

USER 10004:10001
EXPOSE 8002
CMD ["uvicorn", "violet.sqli_lab.app:app", "--host", "0.0.0.0", "--port", "8002", "--no-access-log"]
