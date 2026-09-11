# VIOLET Backend

Python package for the VIOLET multi-agent RL cyber attack & defense simulator.

This is the `violet` Python package: PettingZoo environment, transition engine,
agents, rewards, analytics, training, and the FastAPI service.

See the repository root `README.md` for full documentation, setup, and the
three-phase roadmap.

## Install (development)

```bash
pip install -e ".[dev]"
```

Optional extras: `rl` (PyTorch + Ray RLlib), `api` (FastAPI + SQLAlchemy),
`viz` (matplotlib), or `all`.

## Tests

```bash
pytest tests/unit -q        # fast unit tests
pytest tests/ -q            # full suite
```
