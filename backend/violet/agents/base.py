# violet/agents/base.py — agent abstraction

from __future__ import annotations

from typing import Any, Dict, List, Optional

from violet.types import ATTACKER_AGENT, DEFENDER_AGENT


class Agent:
    """Base class for VIOLET agents.

    Implementations return an action index compatible with the environment's
    discrete action space for their agent type.
    """

    agent_name: str

    def select_action(
        self,
        observation: Any,
        info: Dict[str, Any],
    ) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state at episode start."""
        ...


def is_attacker(agent_name: str) -> bool:
    return agent_name == ATTACKER_AGENT


def is_defender(agent_name: str) -> bool:
    return agent_name == DEFENDER_AGENT
