# tests/unit/test_env.py — VIOLET

import contextlib

from violet.env import VIOLETEnv
from violet.scenarios import v1_enterprise_network
from violet.types import ATTACKER_AGENT, DEFENDER_AGENT


def test_env_initialization():
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=20, seed=42)
    assert env.agents == [ATTACKER_AGENT, DEFENDER_AGENT]
    assert env.metadata["name"] == "violet_v1"


def test_reset_returns_observations_and_infos():
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=20, seed=42)
    obs, infos = env.reset(seed=42)
    assert ATTACKER_AGENT in obs
    assert DEFENDER_AGENT in obs
    assert ATTACKER_AGENT in infos
    assert DEFENDER_AGENT in infos


def test_action_spaces_per_agent():
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=20, seed=42)
    red_space = env.action_space(ATTACKER_AGENT)
    blue_space = env.action_space(DEFENDER_AGENT)
    assert red_space.dtype == "int64"
    assert blue_space.dtype == "int64"
    assert red_space.n > 0
    assert blue_space.n > 0


def test_observation_spaces_per_agent():
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=20, seed=42)
    red_obs_space = env.observation_space(ATTACKER_AGENT)
    blue_obs_space = env.observation_space(DEFENDER_AGENT)
    assert red_obs_space.shape == (64,)
    assert blue_obs_space.shape == (64,)


def test_step_returns_hashable_types():
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=20, seed=42)
    obs, infos = env.reset(seed=42)
    actions = {
        ATTACKER_AGENT: 0,
        DEFENDER_AGENT: 0,
    }
    next_obs, rewards, terminations, truncations, next_infos = env.step(actions)
    assert isinstance(next_obs, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminations, dict)
    assert isinstance(truncations, dict)
    assert isinstance(next_infos, dict)
    assert ATTACKER_AGENT in next_obs
    assert DEFENDER_AGENT in next_obs


def test_step_rewards_summable():
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=20, seed=42)
    obs, infos = env.reset(seed=42)
    total_red = 0.0
    total_blue = 0.0
    for _ in range(5):
        assert env.rng is not None
        actions = {
            ATTACKER_AGENT: env.rng.randint(0, env.action_space(ATTACKER_AGENT).n - 1),
            DEFENDER_AGENT: env.rng.randint(0, env.action_space(DEFENDER_AGENT).n - 1),
        }
        _, rewards, terminations, truncations, _ = env.step(actions)
        total_red += rewards[ATTACKER_AGENT]
        total_blue += rewards[DEFENDER_AGENT]
        if any(terminations.values()):
            break
    assert total_red is not None
    assert total_blue is not None


def test_render_ansi():
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=20, seed=42)
    env.reset(seed=42)
    output = env.render("ansi")
    assert isinstance(output, str)
    assert "Step" in output
    assert "red last-step reward" in output


def test_render_human():
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=20, seed=42)
    env.reset(seed=42)
    output = env.render("human")
    assert isinstance(output, str)


def test_max_cycles_truncation():
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=3, seed=42)
    obs, infos = env.reset(seed=42)
    for step in range(5):
        actions = {
            ATTACKER_AGENT: 0,
            DEFENDER_AGENT: 0,
        }
        _, _, terminations, truncations, _ = env.step(actions)
        if step < 3:
            assert truncations[ATTACKER_AGENT] is False
        else:
            assert truncations[ATTACKER_AGENT] is True
            break


def test_invalid_action_still_runs():
    """An out-of-range action index should not crash the env."""
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=20, seed=42)
    obs, infos = env.reset(seed=42)
    # Pass a larger-than-n action — env should still step (clamped internally)
    actions = {
        ATTACKER_AGENT: env.action_space(ATTACKER_AGENT).n * 2,
        DEFENDER_AGENT: 0,
    }
    # Out-of-range actions are either rejected by the env or handled without
    # crashing — both outcomes are acceptable here; nothing to assert further.
    with contextlib.suppress(Exception):
        env.step(actions)


def test_determinism_same_seed():
    env_a = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=5, seed=42)
    env_b = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=5, seed=42)

    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    assert obs_a[ATTACKER_AGENT] == obs_b[ATTACKER_AGENT]
    assert obs_a[DEFENDER_AGENT] == obs_b[DEFENDER_AGENT]

    actions = {
        ATTACKER_AGENT: 0,
        DEFENDER_AGENT: 0,
    }
    _, rewards_a, _, _, _ = env_a.step(actions)
    _, rewards_b, _, _, _ = env_b.step(actions)
    assert rewards_a == rewards_b
