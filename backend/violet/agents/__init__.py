# violet/agents/__init__.py

from violet.agents.base import Agent, is_attacker, is_defender
from violet.agents.blue import (
    ProactiveBlueAgent,
    RandomBlueAgent,
    ReactiveBlueAgent,
    RuleBasedBlueAgent,
)
from violet.agents.red import (
    RandomRedAgent,
    ScriptedAggressiveRedAgent,
    ScriptedLateralRedAgent,
    ScriptedObjectiveRedAgent,
    ScriptedRandomizedRedAgent,
    ScriptedStealthRedAgent,
)
from violet.agents.rl import RLAgent, RLCheckpoint, TrainingRunMetadata

__all__ = [
    "Agent",
    "ProactiveBlueAgent",
    "RLAgent",
    "RLCheckpoint",
    "RandomBlueAgent",
    "RandomRedAgent",
    "ReactiveBlueAgent",
    "RuleBasedBlueAgent",
    "ScriptedAggressiveRedAgent",
    "ScriptedLateralRedAgent",
    "ScriptedObjectiveRedAgent",
    "ScriptedRandomizedRedAgent",
    "ScriptedStealthRedAgent",
    "TrainingRunMetadata",
    "is_attacker",
    "is_defender",
]
