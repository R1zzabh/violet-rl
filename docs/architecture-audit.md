# Architecture audit

## Baseline and repository state

The repository is `R1zzabh/violet-rl`, branch `main`. The original baseline suite was attempted with `python -m pytest -q` from `backend/` before edits. Collection stopped in `tests/conftest.py` because the host Python environment does not have `pettingzoo` installed (`ModuleNotFoundError`); no baseline pass count is available. Docker Compose v5.4.0 is present. The current host also lacks Gymnasium and FastAPI, so host-side API or simulator tests cannot run without installing the declared extras.

## Existing architecture and execution flow

- `backend/violet/env.py`, `network.py`, `transition.py`, and `agents/` implement the graph simulator. Agents choose integer actions against simulator observations; scripted agents are hand-written rules.
- `backend/violet/algorithms/ppo.py` implements PyTorch clipped PPO for simulator Red against a frozen Blue opponent. It collects simulator transitions, calculates GAE, updates a categorical actor and critic, and saves checkpoints and JSONL metrics. It is not wired to Violet Ledger.
- The local range is a separate FastAPI target (`target_app/`) with seeded Alice/Bob records. A separately isolated executor consumes strict Pydantic `RangeAction` jobs and makes requests to fixed routes on `violet-ledger`.
- Blue control messages use a distinct volume queue. The independent verifier checks fixture canaries and ownership control using executor results plus target evidence.
- `api/range.py` commits jobs and controls to PostgreSQL before queueing them. `persistence/repository.py` stores run and episode records, ordered hash-linked events, results, acknowledgements, and target evidence. Event replay reads persisted events; it does not re-execute actions.
- The Next.js frontend has simulator pages plus `/range-training`, a read-only view of real-range runs and the hash-linked event replay API.
- Compose separates the executor/target range and target database from the control plane. Host API and frontend ports bind to loopback. The executor has no control-plane database mount or network attachment.

## Baseline limitations identified

Before this change, there was no real-range PPO, no role-scoped range observation contract, no persisted learning transition, and no range training/evaluation runner. Simulator PPO represented terminal and truncated steps with one `done` bit, so a time limit could not bootstrap a final value. Existing real-range controls are typed, but only ownership hardening has an independent end-to-end verifier.

## Changes in this increment

Added separate Red/Blue categorical PPO models over versioned fixed-size range observations, an alternating HTTP training runner, and strict API event endpoints for transitions, actual optimizer updates, and episode/run lifecycle status in the existing hash chain. The runner uses typed executor jobs and the existing ownership-hardening control; objective labels come from the independent verifier. Time-limit GAE distinguishes terminal bootstrapping from trajectory boundaries. A held-out comparison script covers the four scripted/learned pairings, and the dashboard shows persisted outcomes without executing replayed actions. See [rl-algorithm.md](rl-algorithm.md), [real-range-training.md](real-range-training.md), and [reward-design.md](reward-design.md).

## Remaining work

Runtime validation on a CPU-only Linux container is complete: Python 3.12.15, PyTorch 2.14.1+cpu, tensor result 5.0, 102 backend tests passing, and the requested 3-episode HTTP run persisted 36 transitions and six optimizer updates. The 20-episode-per-pairing comparison found no verified Red successes and all Blue preventions verified in this easy fixed fixture; it does not establish policy improvement. Rate-limit/block controls remain outside the learnable action mask because they lack independent outcome verifiers. Existing database schema creation still uses `create_all()` rather than migrations.
