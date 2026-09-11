# violet/types.py — shared types across the simulator

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Agent identifiers
# ---------------------------------------------------------------------------
ATTACKER_AGENT = "red"
DEFENDER_AGENT = "blue"

AGENT_NAMES = (ATTACKER_AGENT, DEFENDER_AGENT)

# ---------------------------------------------------------------------------
# Action type enums
# ---------------------------------------------------------------------------
class RedActionType(enum.Enum):
    SCAN = "SCAN"
    EXPLOIT = "EXPLOIT"
    PRIVILEGE_ESCALATE = "PRIVILEGE_ESCALATE"
    LATERAL_MOVE = "LATERAL_MOVE"
    EXFILTRATE = "EXFILTRATE"
    WAIT = "WAIT"


class BlueActionType(enum.Enum):
    MONITOR = "MONITOR"
    PATCH = "PATCH"
    BLOCK = "BLOCK"
    ISOLATE = "ISOLATE"
    RESET = "RESET"
    DEPLOY_HONEYPOT = "DEPLOY_HONEYPOT"
    RESTORE = "RESTORE"
    WAIT = "WAIT"


# ---------------------------------------------------------------------------
# Node / asset status enums
# ---------------------------------------------------------------------------
class CompromiseStatus(enum.Enum):
    NONE = "none"
    DISCOVERED = "discovered"
    COMPROMISED = "compromised"
    PRIVILEGES_ESCALATED = "privileges_escalated"
    EXFILTRATED = "exfiltrated"


class IsolationStatus(enum.Enum):
    CONNECTED = "connected"
    ISOLATED = "isolated"


class ServiceStatus(enum.Enum):
    ACTIVE = "active"
    DOWN = "down"
    PATCHED = "patched"


class AlertLevel(enum.Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Direction / connectivity
# ---------------------------------------------------------------------------
class Direction(enum.Enum):
    INCOMING = "incoming"
    OUTGOING = "outgoing"
    BIDIRECTIONAL = "bidirectional"


# ---------------------------------------------------------------------------
# Target spec for parameterized actions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionTarget:
    """References a specific node/session for an action."""

    node_id: str
    session_id: str | None = None
    param: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")


# ---------------------------------------------------------------------------
# Concrete action (decoded form used internally by the environment)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    agent: str
    action_type: RedActionType | BlueActionType
    target: ActionTarget | None = None

    def is_red(self) -> bool:
        return self.agent == ATTACKER_AGENT

    def is_blue(self) -> bool:
        return self.agent == DEFENDER_AGENT


# ---------------------------------------------------------------------------
# Environment event (structured, emitted on every meaningful transition)
# ---------------------------------------------------------------------------
@dataclass
class Event:
    step: int
    timestamp: float
    agent: str
    action: str
    target: str | None
    result: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "agent": self.agent,
            "action": self.action,
            "target": self.target,
            "result": self.result,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Episode-level result stored for evaluation / replay
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    episode_id: str
    seed: int
    config_version: str
    length: int
    red_reward: float
    blue_reward: float
    winner: str | None
    red_stats: dict[str, Any] = field(default_factory=dict)
    blue_stats: dict[str, Any] = field(default_factory=dict)
    joint_stats: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
