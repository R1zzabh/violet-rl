# VIOLET — Multi-Agent RL Cyber Attack & Defense Simulator

🟣 **VIOLET** (Versatile Intelligent Opponent Learning for Evolving Threats) is a
**simulated cyber network** modeled as a Partially Observable Markov Game, where
autonomous attacker and defender agents learn sequential strategies through
competitive multi-agent reinforcement learning.

## What it is (and isn't)

**It IS:**
- A Markov game simulator with scripted and learnable Red/Blue agents
- A research platform for measuring whether RL produces better cyber strategies than baselines
- An evidence-based system: every "learned behavior" claim is backed by measured trajectories

**It is NOT:**
- A real hacker AI
- A system that attacks external networks
- A tool for real exploitation

All activity happens inside the simulated environment. No real malware. No real targets.

## Three-phase roadmap

| Phase | Red | Blue | Goal |
|-------|-----|------|------|
| 1 | Scripted (fixed strategies) | PPO (learning) | Verify environment, training loop, baselines |
| 2 | Multiple scripted strategies | PPO (trained across strategies) | Generalization, cross-strategy evaluation |
| 3 | PPO (learning) | PPO (learning) | Self-play, non-stationarity, cross-play matrix |

## Stack

- **AI/RL:** Python, PyTorch, PettingZoo (ParallelEnv), Ray RLlib (PPO)
- **Backend:** FastAPI, Pydantic, SQLAlchemy, PostgreSQL, WebSockets
- **Frontend:** Next.js 16, TypeScript, Tailwind CSS, shadcn/ui, Recharts, Lucide
- **Theme:** Neo-Brutalism (black borders, zero-radius, heavy shadows, DM Sans + Space Mono)

## Quick start

```bash
# Clone
git clone <repo> && cd violet

# Full stack (backend + frontend + postgres)
docker compose up

# Or develop locally
pip install -e ".[dev]"
npm install
npm run dev
```

Open the dashboard at the frontend URL. The `/dashboard` shows the latest stored experiment.

## Project structure

```
violet/
├── backend/                 # Python package
│   └── violet/
│       ├── env/            # PettingZoo environment, network, nodes, transitions
│       ├── agents/         # Red/Blue agents, baselines, RL wrappers
│       ├── rewards/        # Configurable reward engine
│       ├── scenarios/      # Network topologies
│       ├── training/       # Training loops, checkpoints
│       ├── evaluation/     # Independent evaluation
│       ├── analytics/      # Behavior discovery, metrics
│       ├── api/            # FastAPI + WebSocket
│       └── experiments/    # Experiment runner
├── frontend/               # Next.js dashboard
├── configs/                # YAML configs (phase1/2/3, rewards, training)
├── experiments/            # Run outputs
├── artifacts/              # Checkpoints, metrics, evaluations
├── docker/                 # Dockerfiles
└── README.md
```

## Status

| Component | Status |
|-----------|--------|
| Repository bootstrap | Implemented |
| Cyber simulator core | In development |
| Red mechanics | Not started |
| Blue mechanics | Not started |
| Environment tests | Not started |
| Phase 1 RL (PPO Blue vs scripted Red) | Not started |
| Dashboard V1 | Not started |
| Phase 2 (multiple Red strategies) | Not started |
| Phase 3 (RL Red vs RL Blue, self-play) | Not started |
| Deployment | Not started |

## Documentation

See `docs/` for architecture, environment, agents, rewards, training, evaluation, experiments.

## Rules

- Never fabricate results, learning curves, or benchmark numbers.
- Every reported metric has a reproducible source.
- Phase N+1 only starts after Phase N passes its tests, baselines, and reproducibility checks.
