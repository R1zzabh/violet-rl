# Real-range training

## Start the range

From the repository root, set distinct disposable passwords in `.env` (the example intentionally leaves both blank), then run:

```powershell
Copy-Item .env.example .env
# Edit VIOLET_DB_PASSWORD and TARGET_DB_PASSWORD to distinct random alphanumeric values.
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/range/health
```

## Train

The supported CPU-only runtime is the Python 3.12 backend container. Build it
once, then run the command from the repository root:

```powershell
docker compose build backend
docker compose run --rm --no-deps backend python -m violet.range.train_real --api http://backend:8000 --episodes 3 --red-steps 6 --seed 42 --run-dir runs/real_range_ppo
```

The image installs the declared `rl` extra with an official CPU wheel. An
optional Linux CPU wheelhouse can shorten builds; the Dockerfile falls back to
the official CPU index when it is absent. The script writes Red/Blue
checkpoints and JSONL episode outcomes beneath `backend/runs/real_range_ppo/`.
Transitions, verifier outcomes, real optimizer updates, and completion status
are also persisted in PostgreSQL. Inspect them at
`GET /range/runs/{run_id}/events` or in the read-only dashboard at
`http://127.0.0.1:3000/range-training`.

Resume both role policies and their optimizer state:

```powershell
docker compose run --rm --no-deps backend python -m violet.range.train_real --api http://backend:8000 --resume --red-checkpoint runs/real_range_ppo/red.pt --blue-checkpoint runs/real_range_ppo/blue.pt --episodes 3 --red-steps 6 --seed 45 --run-dir runs/real_range_ppo
```

Evaluate both frozen checkpoints with deterministic actions and no optimizer updates:

```powershell
docker compose run --rm --no-deps backend python -m violet.range.train_real --api http://backend:8000 --evaluate --red-checkpoint runs/real_range_ppo/red.pt --blue-checkpoint runs/real_range_ppo/blue.pt --episodes 10 --red-steps 6 --seed 100 --run-dir runs/real_range_eval
```

Evaluation results are written as raw episode JSONL and transition records are
marked `mode=evaluation`; no optimizer updates occur. Resume restores optimizer
and saved RNG state. For all four scripted/learned combinations over a common
20-seed held-out range, run:

```powershell
docker compose run --rm --no-deps backend python -m violet.range.evaluate_real --api http://backend:8000 --episodes 20 --seed 10000 --red-checkpoint runs/real_range_ppo/red.pt --blue-checkpoint runs/real_range_ppo/blue.pt
```

The report includes raw episodes, exact denominators, and seeded bootstrap
intervals. A small fixed synthetic exercise can yield degenerate intervals and
does not establish learned improvement.

## Existing simulator and tests

The graph simulator remains a separate experiment:

```powershell
cd backend
python -m violet.cli train --updates 2 --seed 42
python -m pytest -q
python scripts/range_smoke.py
```

The Compose smoke script exercises the real HTTP target and persistence. The host Python test suite requires the declared `.[api,rl,dev]` extras. Docker-backed tests require Docker Engine and a configured `.env`.

The runner's scope is a local synthetic HTTP task. Do not infer learned
improvement from a successful training run alone or present these results as
production-network effectiveness.
