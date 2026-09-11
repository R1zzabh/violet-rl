#!/usr/bin/env python3
"""Standalone VIOLET smoke test — exercises env + agents + transition + rewards end-to-end."""

import sys
from pathlib import Path

# Ensure the violet package is importable from this script's location
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from violet.agents import ScriptedAggressiveRedAgent, RuleBasedBlueAgent
from violet.env import VIOLETEnv
from violet.scenarios import v1_enterprise_network
from violet.types import ATTACKER_AGENT, DEFENDER_AGENT

ENV = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=50, seed=42)
RED = ScriptedAggressiveRedAgent(target_node_id="database")
BLUE = RuleBasedBlueAgent()

print("=" * 70)
print("VIOLET SMOKE TEST — scripted Red vs rule-based Blue")
print("=" * 70)

obs, infos = ENV.reset(seed=42)
RED.reset()
BLUE.reset()
print(f"Reset OK. Agents: {list(ENV.agents)}")
print(f"Red action space n: {ENV.action_space(ATTACKER_AGENT).n}")
print(f"Blue action space n: {ENV.action_space(DEFENDER_AGENT).n}")
print(f"Red obs shape: {len(obs[ATTACKER_AGENT])}")
print(f"Blue obs shape: {len(obs[DEFENDER_AGENT])}")
print()

red_reward = 0.0
blue_reward = 0.0
events: list[dict] = []
terminated = False

for step in range(50):
    if terminated:
        break
    red_action = RED.select_action(ENV, infos.get(ATTACKER_AGENT, {}))
    blue_action = BLUE.select_action(ENV, infos.get(DEFENDER_AGENT, {}))
    actions = {ATTACKER_AGENT: red_action, DEFENDER_AGENT: blue_action}
    obs, rewards, terminations, truncations, infos = ENV.step(actions)

    red_reward += rewards.get(ATTACKER_AGENT, 0.0)
    blue_reward += rewards.get(DEFENDER_AGENT, 0.0)
    terminated = any(terminations.values()) or any(truncations.values())

    for agent_name in (ATTACKER_AGENT, DEFENDER_AGENT):
        for ev in infos.get(agent_name, {}).get("events", []):
            events.append(ev)
            print(f"  [{ev['step']}] {ev['agent']} -> {ev['action']} {ev['target'] or ''} : {ev['result']}")

    if any(terminations.values()):
        print(f"Terminated at step {step + 1}")
        break

print()
print(f"Episode finished at step {step + 1}")
print(f"Red cumulative reward: {red_reward:.2f}")
print(f"Blue cumulative reward: {blue_reward:.2f}")
print(f"Total events: {len(events)}")

print()
print("Network final state:")
print(ENV.render("ansi"))

# Assertions (so this can be used as a real test)
assert step >= 0, "Episode did not run"
assert isinstance(red_reward, float)
assert isinstance(blue_reward, float)
assert isinstance(events, list)

# WEB should be compromised quickly (0.8 vuln) under scripted Red
web = ENV.network.nodes["web"]
assert web.attacker_discovered, "WEB should have been discovered"
print()
print("ASSERTIONS PASSED ✅")
print("  - WEB discovered:", web.attacker_discovered)
print("  - WEB compromised:", web.is_compromised())
print("  - DATABASE critical asset status:", ENV.network.nodes["database"].compromise_status.value)

sys.exit(0)
