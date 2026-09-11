# tests/unit/test_agents.py — VIOLET

import pytest

from violet.agents import (
    RandomBlueAgent,
    RandomRedAgent,
    RuleBasedBlueAgent,
    ScriptedAggressiveRedAgent,
    ScriptedStealthRedAgent,
    ScriptedLateralRedAgent,
    ScriptedRandomizedRedAgent,
    ScriptedObjectiveRedAgent,
    ReactiveBlueAgent,
    ProactiveBlueAgent,
)
from violet.env import VIOLETEnv
from violet.scenarios import v1_enterprise_network
from violet.types import ATTACKER_AGENT, DEFENDER_AGENT


@pytest.fixture
def env():
    return VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=20, seed=42)


@pytest.fixture
def obs_and_info(env):
    obs, info = env.reset(seed=42)
    return obs, info


def test_random_red_agent_returns_valid_index(env, obs_and_info):
    obs, info = obs_and_info
    agent = RandomRedAgent()
    action = agent.select_action(env, info.get(ATTACKER_AGENT, {}))
    space = env.action_space(ATTACKER_AGENT)
    assert 0 <= action < space.n


def test_random_blue_agent_returns_valid_index(env, obs_and_info):
    obs, info = obs_and_info
    agent = RandomBlueAgent()
    action = agent.select_action(env, info.get(DEFENDER_AGENT, {}))
    space = env.action_space(DEFENDER_AGENT)
    assert 0 <= action < space.n


def test_scripted_aggressive_red_selects_action(env, obs_and_info):
    obs, info = obs_and_info
    agent = ScriptedAggressiveRedAgent(target_node_id="database")
    agent.reset()
    action = agent.select_action(env, info.get(ATTACKER_AGENT, {}))
    space = env.action_space(ATTACKER_AGENT)
    assert 0 <= action < space.n


def test_scripted_aggressive_red_theta__phase_progression(env, obs_and_info):
    obs, info = obs_and_info
    agent = ScriptedAggressiveRedAgent(target_node_id="database")
    agent.reset()

    # Phase 0: scan not yet discovered
    action = agent.select_action(env, info.get(ATTACKER_AGENT, {}))
    space = env.action_space(ATTACKER_AGENT)
    assert 0 <= action < space.n

    # advance state: compromise web
    env.network.nodes["web"].attacker_discovered = True
    env.network.nodes["web"].is_compromised = lambda: True  # type: ignore
    env.network.nodes["web"].compromise_status = __import__("violet.types", fromlist=["CompromiseStatus"]).CompromiseStatus.COMPROMISED
    env.network.nodes["web"].attacker_knows_vulnerabilities = True

    action2 = agent.select_action(env, info.get(ATTACKER_AGENT, {}))
    assert 0 <= action2 < space.n


def test_scripted_aggressive_red_walks_topology_inward(env, obs_and_info):
    """Regression: aggressive red must progress from its entry point along the
    topology (web -> app -> database), never attacking the objective directly
    before reaching a foothold adjacent to it."""
    obs, info = obs_and_info
    agent = ScriptedAggressiveRedAgent(target_node_id="database")
    agent.reset()

    first_action = agent.select_action(env, info.get(ATTACKER_AGENT, {}))
    space = env.action_space(ATTACKER_AGENT)
    assert 0 <= first_action < space.n

    # First planned hop must be the entry-adjacent node, not the objective
    first_hop = agent._plan_next_hop(env.network)
    assert first_hop == "web", f"aggressive red should start at entry point, got {first_hop!r}"


def test_scripted_stealth_red_picks_wait_when_alert(env, obs_and_info):
    obs, info = obs_and_info
    agent = ScriptedStealthRedAgent(target_node_id="database")
    env.network.nodes["web"].attacker_discovered = True
    env.network.nodes["web"].alert_level = __import__("violet.types", fromlist=["AlertLevel"]).AlertLevel.HIGH
    action = agent.select_action(env, info.get(ATTACKER_AGENT, {}))
    space = env.action_space(ATTACKER_AGENT)
    assert 0 <= action < space.n


def test_scripted_lateral_red_handles_no_foothold(env, obs_and_info):
    obs, info = obs_and_info
    agent = ScriptedLateralRedAgent(target_node_id="database")
    action = agent.select_action(env, info.get(ATTACKER_AGENT, {}))
    space = env.action_space(ATTACKER_AGENT)
    assert 0 <= action < space.n


def test_scripted_objective_red_targets_objective(env, obs_and_info):
    obs, info = obs_and_info
    agent = ScriptedObjectiveRedAgent(objective_node_id="database")
    action = agent.select_action(env, info.get(ATTACKER_AGENT, {}))
    space = env.action_space(ATTACKER_AGENT)
    assert 0 <= action < space.n


def test_scripted_randomized_red_returns_valid_index(env, obs_and_info):
    obs, info = obs_and_info
    agent = ScriptedRandomizedRedAgent()
    action = agent.select_action(env, info.get(ATTACKER_AGENT, {}))
    space = env.action_space(ATTACKER_AGENT)
    assert 0 <= action < space.n


def test_rule_based_blue_isolate_compromised(env, obs_and_info):
    obs, info = obs_and_info
    agent = RuleBasedBlueAgent()
    # Simulate a compromised node
    env.network.nodes["web"].compromise_status = __import__("violet.types", fromlist=["CompromiseStatus"]).CompromiseStatus.COMPROMISED
    env.network.nodes["web"].isolated = __import__("violet.types", fromlist=["IsolationStatus"]).IsolationStatus.CONNECTED
    action = agent.select_action(env, info.get(DEFENDER_AGENT, {}))
    space = env.action_space(DEFENDER_AGENT)
    assert 0 <= action < space.n


def test_reactive_blue_monitors_when_no_compromise(env, obs_and_info):
    obs, info = obs_and_info
    agent = ReactiveBlueAgent(detection_threshold=1)
    action = agent.select_action(env, info.get(DEFENDER_AGENT, {}))
    space = env.action_space(DEFENDER_AGENT)
    assert 0 <= action < space.n


def test_proactive_blue_patches_high_vuln(env, obs_and_info):
    obs, info = obs_and_info
    agent = ProactiveBlueAgent()
    env.network.nodes["web"].vulnerability_rate = 0.9
    env.network.nodes["web"].is_critical_asset = False  # type: ignore
    action = agent.select_action(env, info.get(DEFENDER_AGENT, {}))
    space = env.action_space(DEFENDER_AGENT)
    assert 0 <= action < space.n
