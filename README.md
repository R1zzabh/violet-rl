# VIOLET

VIOLET is a research project for measuring attacker and defender strategies. It
now has two separate execution modes: the existing graph simulator and a local
HTTP training range called Violet Ledger.

## Current implementation

| Area | State |
|---|---|
| Graph simulator | Existing PettingZoo-style simulator with scripted policies and a custom PPO trainer. Its nodes and actions remain simulated. |
| Violet Ledger | A real FastAPI process with synthetic Alice/Bob accounts and documents in a separate target database. The initial exercise is authenticated IDOR. |
| HTTP executor | A separate container accepts typed route, login, document-access, and health actions. Its destination and routes are fixed in code. |
| Blue controls | Typed reset, ownership-hardening, actor block, and rate-limit commands reach the target through a volume queue. Ownership hardening can be verified and persisted only after Alice is denied and Bob remains served. |
| VIOLET event storage | Range runs, episodes, typed job/control outboxes, target evidence, and hash-linked events are stored in PostgreSQL. `GET /range/runs/{run_id}/events?after_seq=N` replays the committed timeline; `scripts/export_range_run.py` writes a redacted JSONL bundle with checksums. The initial schema is created on API startup; Alembic migrations are not implemented. |
| Range learning | Separate Red/Blue PPO policies train through typed executor jobs and persist transitions, verifier-based rewards, and actual PPO updates. No improvement claim is made without held-out results. |
| Dashboard | `/range-training` shows persisted real-range runs and hash-linked event replay; it does not execute historical actions. |

The local range sends HTTP requests to the Violet Ledger container only. It does
not accept arbitrary hosts, URLs, ports, paths, request bodies, shell commands,
or external callbacks. The executor and target use internal Docker networks;
UI and API ports bind to loopback, and databases have no published ports. This
Docker engine does not publish ports from an `internal: true` bridge, so Compose
uses a separate no-masquerade bridge for the loopback UI/API ports. The
executor and target are not attached to that bridge.

## Start the local stack

Docker Compose requires separate, non-empty passwords for the two databases.
Copy the example and set each value to a distinct random alphanumeric string:

```bash
cp .env.example .env
```

Then start the stack:

```bash
docker compose up --build
```

The dashboard is at [http://127.0.0.1:3000](http://127.0.0.1:3000), the API is
at [http://127.0.0.1:8000](http://127.0.0.1:8000), and the Violet Ledger target
is reachable only from the executor on the internal range network. Its health
check is available through `GET /range/health`.

The VIOLET database has its own volume for range run and event records; the
simulator's existing metrics and checkpoints remain file-backed. The target
database stores only disposable synthetic target data. `POST /range/reset`
re-seeds target records, clears sessions and controls, and leaves VIOLET run
and event records alone.

## Run the real-target acceptance slice

Install the backend's API and development extras, then run the Compose-backed
smoke script from the backend directory:

```bash
cd backend
pip install -e ".[api,dev]"
python scripts/range_smoke.py
```

The script resets fixtures with seed 17, logs in as Alice using a synthetic
credential reference, requests Bob's document through the executor, and
requires the independent verifier to match the seeded Bob canary. It then
applies ownership hardening and checks that Alice receives HTTP 403 while Bob
still receives HTTP 200. It submits those requests to the control verifier and
checks the persisted event sequence, evidence correlation, hash links, and an
`after_seq` replay.

Focused local tests are:

```bash
pytest tests/unit/test_range_contracts.py -q
pytest tests/unit/test_control_verifier.py -q
pytest tests/integration/test_real_target.py -q
pytest tests/integration/test_persistence_replay.py -q
```

Range runs, events, job results, and control acknowledgements are persisted in
PostgreSQL before they are returned by the API. Target access evidence is
stored in the `target-evidence` Docker volume, mounted read-only into the API,
and imported into the event timeline when a result is read or replayed.
Executor jobs/results and target control messages use separate named volumes;
the database outbox is reconciled into those queues after API restart. The
initial SQLAlchemy schema is created at startup rather than managed by
Alembic. The exporter checks the source event hash chain before writing a
redacted JSONL timeline and SHA-256 manifest. The real-range dashboard reads
the same persisted event history.

## Simulator training

The graph simulator and its existing training flow remain separate from the
real HTTP range. Install the `rl` extra for PyTorch and follow the commands in
`backend/README.md`. Simulator rewards and saved evaluations must not be
presented as real-target results.

## Initial real-range PPO runner

Build the CPU-only training runtime once, then run the trainer inside Compose:

```powershell
docker compose build backend
docker compose run --rm backend python -m violet.range.train_real --api http://backend:8000 --episodes 3 --red-steps 6 --seed 42 --run-dir runs/real_range_ppo
```

Red and Blue have separate policies. Red selects existing typed requests; Blue
chooses no-op or ownership hardening, the only control with an independent
verifier today. Checkpoints and an episode JSONL are saved under the run
directory, and each transition is appended to the PostgreSQL event chain.
Resume both policies with `--resume --red-checkpoint ... --blue-checkpoint ...`;
use `--evaluate` with both checkpoints for deterministic inference without
updates. Compare the four baseline pairings over identical held-out seeds with:

```powershell
docker compose run --rm backend python -m violet.range.evaluate_real --api http://backend:8000 --episodes 20 --seed 10000 --red-checkpoint runs/real_range_ppo/red.pt --blue-checkpoint runs/real_range_ppo/blue.pt
```

Open `http://localhost:3000/range-training` for persisted events and updates.
This local HTTP experiment does not demonstrate production network effectiveness.
See `docs/real-range-training.md` and
`docs/rl-algorithm.md`.

## Repository map

- `backend/violet/env.py`, `network.py`, and `transition.py`: graph simulator.
- `backend/violet/algorithms/ppo.py`: current simulator PPO implementation.
- `backend/violet/range/`: versioned actions, executor, verifier, and spool.
- `backend/violet/persistence/`: SQLAlchemy event store, replay, and export.
- `backend/violet/target_app/`: Violet Ledger, synthetic database, controls, and telemetry.
- `backend/violet/api/`: simulator API and local range endpoints.
- `docker-compose.yml`, `docker/target/`, and `docker/executor/`: local isolation boundary.
- `backend/tests/`: simulator and real-target tests.
- `backend/docs/development_log.md`: dated implementation history and measured results.

## Remaining milestones

Remaining work includes Alembic migration management, longer controlled
experiments with uncertainty estimates, and diagnosis of policy quality from
measured held-out results. The quick three-episode run validates the execution
path and optimizer updates; it is not enough to establish learning
improvement.
