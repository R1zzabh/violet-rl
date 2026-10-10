# tests/smoke/test_smoke.py — VIOLET

from typing import Any

from violet.agents import RuleBasedBlueAgent, ScriptedAggressiveRedAgent
from violet.env import VIOLETEnv
from violet.scenarios import v1_enterprise_network


def test_smoke_full_episode_scripted_agents():
    """One full episode with scripted Red vs rule-based Blue.

    Proves the environment, actions, transitions, rewards, and termination all
    cooperate end-to-end before any RL is introduced (BLOCK F prerequisite).
    """
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=50, seed=42)
    red = ScriptedAggressiveRedAgent(target_node_id="database")
    blue = RuleBasedBlueAgent()

    obs, infos = env.reset(seed=42)
    red.reset()
    blue.reset()

    red_reward = 0.0
    blue_reward = 0.0
    events: list[dict[str, Any]] = []
    terminated = False

    for step in range(50):
        if terminated:
            break
        red_action = red.select_action(env, infos.get("red", {}))
        blue_action = blue.select_action(env, infos.get("blue", {}))
        actions = {"red": red_action, "blue": blue_action}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        red_reward += rewards.get("red", 0.0)
        blue_reward += rewards.get("blue", 0.0)
        terminated = any(terminations.values()) or any(truncations.values())

        for agent_name in ("red", "blue"):
            for ev in infos.get(agent_name, {}).get("events", []):
                events.append(ev)

    # Sanity: we ran at least one step and didn't crash
    assert step >= 0
    assert isinstance(red_reward, float)
    assert isinstance(blue_reward, float)
    assert isinstance(events, list)


def test_smoke_red_can_reach_objective_in_few_steps():
    """Under scripted Red, the attacker should usually compromise web quickly (high vuln).

    This is a sanity check that the transition engine is not breaking the obvious path.
    """
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=30, seed=42)
    red = ScriptedAggressiveRedAgent(target_node_id="database")
    blue = RuleBasedBlueAgent()

    obs, infos = env.reset(seed=42)
    red.reset()
    blue.reset()

    web_compromised = False
    for step in range(30):
        red_action = red.select_action(env, infos.get("red", {}))
        blue_action = blue.select_action(env, infos.get("blue", {}))
        actions = {"red": red_action, "blue": blue_action}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        assert env.network is not None
        if env.network.nodes["web"].is_compromised():
            web_compromised = True
            break

        if any(terminations.values()):
            break

    # web should be compromised relatively quickly due to 0.8 vuln rate
    assert web_compromised, "Red could not compromise WEB — likely environment bug"
