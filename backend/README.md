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
The real-range slice includes a separate CPU PPO trainer for Red and Blue. Ray
is not required by the custom simulator or real-range PPO implementations.

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
outstanding. Simulator runs remain file-backed. Real-range observations,
actions, outcomes, rewards, and actual optimizer updates are recorded in the
hash-linked event timeline.

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
also verifies HTTP evidence and sequence replay through the API. The read-only
frontend event timeline is available at `/range-training` and never executes
historical actions.

## Initial real-range PPO runner

Rebuild the backend image once to install the declared CPU-only RL extra:

```bash
docker compose build backend
```

With the Compose stack running, run the trainer in the backend's Python 3.12
container (the API service is addressed as `backend` on the Compose network):

```bash
docker compose run --rm backend python -m violet.range.train_real --api http://backend:8000 --episodes 3 --red-steps 6 --seed 42 --run-dir runs/real_range_ppo
```

The runner has separate Red and Blue actor/critic networks and checkpoints.
Per-step transitions are persisted by `POST /range/training/transitions` in the
existing event chain. Blue is limited to no-op and ownership hardening because
only ownership hardening has an independent verifier proving Alice's denial
and Bob's continued service. Use `--resume --red-checkpoint ...
--blue-checkpoint ...` to continue or `--evaluate` with both checkpoints for
deterministic inference. Compare the five scripted, random, and learned pairings
plus a passive-Blue calibration cell on common held-out fixture seeds with:

```bash
docker compose build backend
docker compose run --rm backend python -m violet.range.evaluate_real --api http://backend:8000 --episodes 20 --seed 10000 \
  --red-checkpoint runs/real_range_ppo_final/red.pt \
  --blue-checkpoint runs/real_range_ppo_final/blue.pt \
  --training-seeds 42,43,44 \
  --output runs/real_range_eval/audit_comparison_with_calibration.json
```

The report contains raw episode records, per-role action frequencies, selected
action probability, policy entropy, and independent-verifier labels. The
passive-Blue cell checks the known unopposed attack. Random actions are sampled
uniformly from the valid action mask using the evaluation
seed. This synthetic HTTP experiment is not evidence of production-network
effectiveness. See `docs/rl_effectiveness_audit.md` and
`docs/rl_experiment_results.md` for the task limitations and measured results.
