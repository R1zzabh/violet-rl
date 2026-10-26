FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN pip install --upgrade pip \
    && pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "sqlalchemy>=2.0" "psycopg2-binary>=2.9" "pydantic>=2.6" "alembic>=1.13" \
    && addgroup --system --gid 10001 violet-share \
    && adduser --system --uid 10002 --ingroup violet-share ledger \
    && mkdir -p /control/inbox /control/results /var/lib/violet/evidence \
    && chown -R ledger:violet-share /app /control/results /var/lib/violet/evidence \
    && chmod 0755 /control/inbox

COPY backend/violet/__init__.py backend/violet/_version.py ./violet/
COPY backend/violet/range/__init__.py backend/violet/range/contracts.py backend/violet/range/spool.py ./violet/range/
COPY backend/violet/target_app ./violet/target_app
COPY backend/migrations/target ./migrations/target
COPY backend/scripts/provision_runtime_role.py ./scripts/provision_runtime_role.py
COPY backend/alembic-target.ini ./alembic-target.ini

USER 10002:10001
EXPOSE 8001
CMD ["uvicorn", "violet.target_app.app:app", "--host", "0.0.0.0", "--port", "8001", "--no-access-log"]
