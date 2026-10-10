# tests/conftest.py — VIOLET

import pytest

from violet.env import VIOLETEnv
from violet.scenarios import v1_enterprise_network


@pytest.fixture
def env():
    """Provide a fresh VIOLET environment for each test."""
    return VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=20, seed=42)


@pytest.fixture
def seeded_rng():
    """Provide a deterministic RNG for tests that need explicit randomness."""
    import random

    rng = random.Random(42)
    return rng
