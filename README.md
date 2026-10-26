# VIOLET

VIOLET is a research project for measuring attacker and defender strategies. It
keeps the graph simulator separate from two isolated HTTP exercises: the new
SQLi learning lab and the preserved Violet Ledger IDOR range.

## Current implementation

| Area | State |
|---|---|
| Graph simulator | Existing PettingZoo-style simulator with scripted policies and a custom PPO trainer. Its nodes and actions remain simulated. |
| SQLi learning lab | A separate HTTP service and database seeded with synthetic records. Red and Blue use independent PPO policies; Red composes bounded query inputs and Blue controls query parameterization, scoped rate limiting, and temporary guest isolation. |
| Violet Ledger | Preserved real FastAPI process with synthetic Alice/Bob accounts and documents in its own target database. The existing exercise is authenticated IDOR; historical event replay remains available. |
| HTTP executor | A separate container accepts typed route, login, document-access, and health actions. Its destination and routes are fixed in code. |
| Blue controls | Typed reset, ownership-hardening, actor block, and rate-limit commands reach the target through a volume queue. Only ownership hardening is policy-eligible; verification also checks Alice's legitimate read. |
| VIOLET event storage | Range jobs, full policy samples and transitions, detections, rewards, training lifecycle and hash-linked events are stored in PostgreSQL. Separate Alembic streams manage control and target databases. Run events and episode timelines replay committed records. |
| Range learning | A dedicated CPU worker holds separate Red and Blue policies fixed within each episode, then updates both with masked categorical PPO at the episode boundary. Teaching runs may be short; research runs require 10,000 completed episodes per training seed. |
| Evaluation | Frozen candidate and baseline episodes run on paired seeds. Reports include verified attack/prevention, legitimate-read availability, and paired uncertainty. A supported-improvement claim also requires five independent training seeds and the availability criterion. |
| Dashboard | `/sqli` launches the live SQLi walkthrough; `/sqli/{id}` replays the stored episode and plots measured outcomes separately from PPO metrics. `/range` preserves Violet Ledger training and replay. Graph simulator pages remain separately labeled. |

Each executor sends HTTP requests to exactly one fixed target: the archived
Ledger executor to Violet Ledger, and the SQLi executor to the SQLi service.
Neither accepts arbitrary hosts, URLs, ports, paths, shell commands, or
callbacks. The SQLi target database is separate and grants its runtime role
`SELECT` only. Internal Docker networks prevent public egress; UI and API ports
bind to loopback, and databases and targets have no published ports. Compose
uses a separate no-masquerade bridge for the loopback UI/API ports.

## Start the local stack

Docker Compose requires separate, non-empty owner and restricted runtime
passwords for the control, Ledger, and SQLi databases, plus a separate random
`VIOLET_TRAINER_TOKEN`:

```bash
cp .env.example .env
```

Then start the stack:

```bash
docker compose up --build
```

The dashboard is at [http://127.0.0.1:3000](http://127.0.0.1:3000), the API is
at [http://127.0.0.1:8000](http://127.0.0.1:8000), and both HTTP targets are
reachable only from their fixed executors on internal networks. SQLi target
health is reported by `GET /range/sqli/target/status`; Ledger health remains at
`GET /range/health`.

The VIOLET database has its own volume for range run and event records; the
simulator's existing metrics and checkpoints remain file-backed. Each target
database stores only its own synthetic records. `POST /range/reset`
re-seeds target records, clears sessions and controls, and leaves VIOLET run
and event records alone. PostgreSQL schemas are managed by separate explicit
Alembic streams; API and target processes connect with restricted runtime
roles.

SQLi PPO checkpoints and metrics use a separate trainer-owned
`sqli-training-runs` Docker volume, so training and checkpoint evaluation do
not change the ownership or contents of existing `backend/runs` artifacts.

## Run the real-target acceptance slice

Install the backend's API and development extras, then run the Compose-backed
smoke script from the backend directory:

```bash
cd backend
pip install -e ".[api,dev]"
python scripts/range_smoke.py
```

The Ledger script resets fixtures with seed 17, logs in as Alice using a synthetic
credential reference, requests Bob's document through the executor, and
requires the independent verifier to match the seeded Bob canary. It then
applies ownership hardening and checks that Alice receives HTTP 403 while Alice
and Bob receive HTTP 200 for their own documents. It submits those requests to
the control verifier and checks the persisted event sequence, evidence
correlation, hash links, and an `after_seq` replay. The isolation smoke checks
the Compose service boundaries and prohibited destinations.

The SQLi acceptance probe is verifier traffic, never learned behavior or
training data. It sends a bounded tautology action through the fixed executor,
verifies synthetic foreign rows and correlated target evidence, applies query
parameterization, resends the same action class, and checks a legitimate lookup.
It also confirms that the event timeline can replay the persisted results:

```bash
cd backend
python scripts/sqli_range_smoke.py
```

Focused local tests are:

```bash
pytest tests/unit/test_range_contracts.py -q
pytest tests/unit/test_control_verifier.py -q
pytest tests/integration/test_real_target.py -q
pytest tests/integration/test_persistence_replay.py -q
python scripts/range_isolation_smoke.py
```

Range runs, events, job results, and control acknowledgements are persisted in
PostgreSQL before they are returned by the API. Target access evidence is
stored in the `target-evidence` Docker volume, mounted read-only into the API,
and imported into the event timeline when a result is read or replayed.
Executor jobs/results and target control messages use separate named volumes;
the database outbox is reconciled into those queues after API restart. The
Separate Alembic migration services initialize the control and target schemas.
The exporter checks the source event hash chain before writing a redacted JSONL
timeline and SHA-256 manifest. The backend records versioned observations,
full policy samples and transitions, telemetry-derived alerts,
evidence-attributed rewards, role-specific PPO updates, and checkpoint digests.
Episode listings, paginated timelines, and update records are read-only. Manual
workbench requests are auditable but never enter PPO rollouts. The range console
is separate from simulator training.

## Simulator training

The graph simulator and its existing training flow remain separate from the
real HTTP range. Install the `rl` extra for PyTorch and follow the commands in
`backend/README.md`. Simulator rewards and saved evaluations must not be
presented as real-target results.

## Repository map

- `backend/violet/env.py`, `network.py`, and `transition.py`: graph simulator.
- `backend/violet/algorithms/ppo.py`: current simulator PPO implementation.
- `backend/violet/range/`: versioned Violet Ledger actions, executor, verifier, and spool.
- `backend/violet/sqli_lab/`: versioned composed SQLi action grammar, isolated target, executor, controls, and separate Red/Blue PPO training loop.
- `backend/violet/persistence/`: SQLAlchemy event store, replay, and export.
- `backend/violet/target_app/`: Violet Ledger, synthetic database, controls, and telemetry.
- `backend/violet/api/`: simulator API and local range endpoints.
- `docker-compose.yml`, `docker/target/`, `docker/sqli-target/`, and both executor images: local isolation boundary.
- `backend/tests/`: simulator and real-target tests.
- `backend/docs/development_log.md`: dated implementation history and measured results.

## Remaining milestones

Remaining research work includes five-seed aggregation and held-out evaluation,
broader independently verified control effects, narrower database grants,
append-only SQL enforcement, and full browser/service restart acceptance. The
current v2 range code has not completed a formal 10,000-episode campaign, so it
supports no learning-improvement claim. Earlier experiment reports are specific
to their recorded code and reward versions.
