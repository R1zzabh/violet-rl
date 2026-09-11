# violet/agents/__init__.py

from violet.agents.base import Agent, is_attacker, is_defender
from violet.agents.red import (
    RandomRedAgent,
    ScriptedAggressiveRedAgent,
    ScriptedStealthRedAgent,
    ScriptedLateralRedAgent,
    ScriptedRandomizedRedAgent,
    ScriptedObjectiveRedAgent,
)
from violet.agents.blue import (
    RandomBlueAgent,
    RuleBasedBlueAgent,
    ReactiveBlueAgent,
    ProactiveBlueAgent,
)
from violet.agents.rl import RLAgent, RLCheckpoint, TrainingRunMetadata

__all__ = [
    "Agent",
    "is_attacker",
    "is_defender",
    # red
    "RandomRedAgent",
    "ScriptedAggressiveRedAgent",
    "ScriptedStealthRedAgent",
    "ScriptedLateralRedAgent",
    "ScriptedRandomizedRedAgent",
    "ScriptedObjectiveRedAgent",
    # blue
    "RandomBlueAgent",
    "RuleBasedBlueAgent",
    "ReactiveBlueAgent",
    "ProactiveBlueAgent",
    # rl
    "RLAgent",
    "RLCheckpoint",
    "TrainingRunMetadata",
]
