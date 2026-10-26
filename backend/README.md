# VIOLET backend

The backend contains a graph simulator and two distinct real HTTP exercises:
the SQLi learning lab and the preserved Violet Ledger IDOR range. Simulator
runs do not produce evidence about target traffic or real-range learning.

## Install and run

```bash
pip install -e ".[api,dev]"
pip install -e ".[rl]"  # required by the CPU range trainer
```

Configure `.env` at the repository root from `.env.example` with separate owner
and restricted runtime passwords for the control, Ledger, and SQLi databases,
plus a separate `VIOLET_TRAINER_TOKEN`, then start Compose:

```bash
docker compose up --build
```

Each target is reachable only on its private range network. Executors have no
database or API network; the trainer reaches only the API through
`trainer-control`; migration services attach only to their respective database
networks. The SQLi runtime role can only read its fixture table. API and
frontend ports bind to loopback.

## Real-range workflow

Open `/sqli` for the SQLi lab's target status, PPO training, evaluation, and
persisted instructor replay. Open `/range` for the preserved Ledger's typed
manual workbench, Blue SOC, and historical training/evaluation views. SQLi
training uses separate Red and Blue PPO actors, critics, optimizers, random
streams, and policy versions. The pair stays fixed within each episode;
accepted transitions are stored before either role updates at the boundary.
Unknown or rejected outcomes never enter PPO data. Red selects bounded action
components that the executor composes into a request parameter; Blue acts on
sanitized HTTP telemetry and the target applies its typed control. The no-defense
comparator is evaluation-only.

The Ledger training flow updates separate Red and Blue policies after the same
completed episode. Teaching runs can be short; research training
requires 10,000 completed episodes per seed. `/range/{training_run_id}` replays
the persisted event stream. The API validates the exact observation, full
policy sample, and decision before a policy action. Manual requests use the
same bounded executor and remain outside PPO rollouts. Blue receives sanitized
telemetry without verifier labels or session aliases.

The Ledger verifier confirms the synthetic IDOR only after correlated target
evidence matches the seeded canary. A control is “applied” on target
acknowledgement and “verified” only after follow-up foreign-denial and owner
access probes. Rewards link to verifier/detector/control evidence. The event
stream replays committed sequence numbers and resumes from `Last-Event-ID`.

The Ledger CPU worker runs independent masked categorical PPO policies and saves both
optimizer states with a schema-checked joint checkpoint. Evaluation loads a
frozen pair without optimizer updates and compares it with a scripted or random
pair on paired seeds. Reports include verified task outcomes, legitimate-read
availability, and uncertainty. One training seed cannot pass the five-seed
research gate and is never labeled an improvement.

```bash
python scripts/range_smoke.py
python scripts/sqli_range_smoke.py
python scripts/range_isolation_smoke.py
```

The Ledger smoke verifies the seeded IDOR, hardening effect, owner availability,
correlated evidence, hash-linked replay, and cursor resume. The SQLi smoke
checks real target requests before/after parameterization, synthetic evidence,
legitimate-query availability, and event replay. The isolation smoke checks
both Compose network boundaries, published ports, SQLi read-only permissions,
and approved/prohibited destinations.

## Persistence

`range-migrate` and `target-migrate` own separate Alembic streams with explicit
table revisions. `sqli-migrate` provisions the SQLi fixture schema and its
read-only runtime role. API/target PostgreSQL startup validates migrated schemas.
SQLite test databases self-initialize. Migration services use owner credentials;
API and target processes use separate restricted runtime roles. Ledger grants
retain existing behavior; SQLi uses a read-only `SELECT` grant. Append-only SQL
permissions for the event store remain hardening work.

The canonical range event chain is stored in VIOLET PostgreSQL. Target
middleware produces structured JSONL evidence; artifact exports verify event
hashes and redact secrets. Graph simulator run artifacts, metrics, and
checkpoints remain beneath `backend/runs/<run_id>/`. The Compose trainer writes
SQLi PPO checkpoints and metrics to its trainer-owned `sqli-training-runs`
volume so the restricted worker can save and reload them without changing
ownership of existing local research artifacts.

## Simulator-only training

`violet train` and the current graph PPO implementation operate only on the
PettingZoo graph simulator. They do not send Violet Ledger HTTP requests and
their reward/evaluation artifacts must remain labeled simulator-only.

See [`docs/REAL_RANGE_ARCHITECTURE.md`](docs/REAL_RANGE_ARCHITECTURE.md) for
event semantics, network boundaries, endpoint details, and current limitations.
