# VIOLET backend

The backend contains the graph simulator, its simulator-only PPO trainer, the
FastAPI service, and the separate Violet Ledger HTTP range. Range actions are
typed contracts; the executor maps them to fixed operations against the local
`violet-ledger` service.

## Install

```bash
pip install -e ".[api,dev]"
```

Install the optional `rl` extra to run simulator PPO (`pip install -e ".[rl]"`).
The current real-range slice has no learning trainer. Ray is not required by the
custom simulator PPO implementation.

## Run the local real range

Start Docker Compose from the repository root after configuring `.env` from
`.env.example`. The target is reachable only on Compose's internal range
network. The executor cannot access either PostgreSQL service; Violet Ledger
can access only its separate synthetic target database.

```bash
docker compose up --build
```

Check `GET http://127.0.0.1:8000/range/health`, then run the complete bounded
IDOR/control acceptance path:

```bash
python scripts/range_smoke.py
```

The range API exposes `POST /range/jobs`, `GET /range/jobs/{job_id}`,
`POST /range/reset`, `POST /range/controls`, `GET /range/controls/{id}`, and
`POST /range/controls/{id}/verify` for ownership-control verification, plus
`GET /range/runs/{run_id}/events?after_seq=N` for sequence-based replay.
Job actions contain operation IDs and catalog references; they do not contain
URLs or arbitrary paths. The executor enforces a 40-request episode budget, a
five-minute episode deadline, one in-flight request, a five-second request
timeout, and a ten-request-per-second ceiling.

## Storage boundary

The two PostgreSQL services have separate named volumes. Violet Ledger's
database contains only Alice/Bob test fixtures and target control state. The
VIOLET PostgreSQL database stores range runs, episodes, typed job/control
outboxes, target evidence, and a sequence-numbered, hash-linked event timeline.
The API commits each job/control before placing it in the volume queue and
reconciles pending records after restart. Result and acknowledgement records
are added to the canonical timeline before replay returns them. The control
verifier requires post-acknowledgement Alice `403` and Bob `200` transactions
with target evidence before it records a verified outcome. Target access
evidence and control acknowledgements are mounted read-only into the API.

The current SQLAlchemy schema is bootstrapped with `create_all()` at API
startup; Alembic migrations and immutable database permissions are still
outstanding. Simulator runs and PPO artifacts remain file-backed, and the
real-range API does not yet persist policy observations, training transitions,
or rewards.

Export a persisted real-range run from the API container:

```bash
docker compose exec backend python scripts/export_range_run.py <run-id>
```

The bundle is written beneath `backend/runs/<run-id>/`. Export verifies the
event hash chain and redacts session handles, cookies, tokens, and credential
values before writing JSONL plus a SHA-256 manifest.

## Verify

```bash
pytest tests/unit/test_range_contracts.py -q
pytest tests/unit/test_control_verifier.py -q
pytest tests/integration/test_real_target.py -q
pytest tests/integration/test_persistence_replay.py -q
```

The simulator suite does not prove real-range learning. The persistence test
checks event ordering/hash links and repository restart; the Compose smoke test
also verifies HTTP evidence and sequence replay through the API. A frontend
replay UI is not connected yet.
