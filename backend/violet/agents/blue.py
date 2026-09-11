# violet/agents/blue.py — blue (defender) agent implementations

from __future__ import annotations

import random
from typing import Any, Dict, List

from violet.env import VIOLETEnv
from violet.types import DEFENDER_AGENT, BlueActionType


class BlueAgent:
    """Base class for blue agents that returns a discrete action index suitable for
    the environment's blue action space."""

    agent_name = DEFENDER_AGENT

    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        ...


class RandomBlueAgent(BlueAgent):
    """Uniformly random defender — baseline 1 (per spec)."""

    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        action_space = env.action_space(DEFENDER_AGENT)
        if env.rng is not None:
            return env.rng.randint(0, action_space.n - 1)
        return random.randint(0, action_space.n - 1)


class RuleBasedBlueAgent(BlueAgent):
    """Rule-based reactive defender — baseline 2 (per spec).

    Logic:
        - If a node is compromised and not isolated → ISOLATE
        - If a node has a high alert and is not compromised → MONITOR
        - If a node is patched and compromised → RESET
        - Otherwise → WAIT
    """

    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        if env.network is None:
            raise RuntimeError("Environment not initialized")
        node_ids = list(env.network.nodes.keys())

        # Priority 1: isolate compromised, non-isolated nodes
        for nid in node_ids:
            node = env.network.nodes[nid]
            if node.is_compromised() and node.isolated.value == "connected":
                target_idx = node_ids.index(nid)
                type_idx = list(BlueActionType).index(BlueActionType.ISOLATE)
                return target_idx * len(BlueActionType) + type_idx

        # Priority 2: monitor nodes with alerts
        for nid in node_ids:
            node = env.network.nodes[nid]
            if node.alert_level.value != "none":
                target_idx = node_ids.index(nid)
                type_idx = list(BlueActionType).index(BlueActionType.MONITOR)
                return target_idx * len(BlueActionType) + type_idx

        # Priority 3: reset compromised patched nodes
        for nid in node_ids:
            node = env.network.nodes[nid]
            if node.is_compromised() and node.service_status.value == "patched":
                target_idx = node_ids.index(nid)
                type_idx = list(BlueActionType).index(BlueActionType.RESET)
                return target_idx * len(BlueActionType) + type_idx

        # Priority 4: patch nodes with moderate vulnerability and no alert
        for nid in node_ids:
            node = env.network.nodes[nid]
            if not node.is_compromised() and node.alert_level.value == "none":
                if node.vulnerability_rate > 0.5:
                    target_idx = node_ids.index(nid)
                    type_idx = list(BlueActionType).index(BlueActionType.PATCH)
                    return target_idx * len(BlueActionType) + type_idx

        # Default: monitor the most critical asset
        critical = [nid for nid in node_ids if env.network.nodes[nid].is_critical_asset]
        if critical:
            target_idx = node_ids.index(critical[0])
            type_idx = list(BlueActionType).index(BlueActionType.MONITOR)
            return target_idx * len(BlueActionType) + type_idx

        # Fallback: wait
        target_idx = 0
        type_idx = list(BlueActionType).index(BlueActionType.WAIT)
        return target_idx * len(BlueActionType) + type_idx


class ReactiveBlueAgent(BlueAgent):
    """Reactive blue agent — waits for detection, then responds.

    Priority:
        - Detect first (MONITOR), then respond to compromised nodes.
    """

    def __init__(self, detection_threshold: int = 1):
        self.detection_threshold = detection_threshold
        self.detected_count = 0

    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        if env.network is None:
            raise RuntimeError("Environment not initialized")
        node_ids = list(env.network.nodes.keys())
        compromised_count = sum(1 for n in env.network.nodes.values() if n.is_compromised())
        self.detected_count = compromised_count

        if compromised_count >= self.detection_threshold:
            # Respond to highest-value compromised node
            compromised_nodes = [
                (nid, n.value)
                for nid, n in env.network.nodes.items()
                if n.is_compromised()
            ]
            compromised_nodes.sort(key=lambda x: -x[1])
            target_id = compromised_nodes[0][0]
            target_idx = node_ids.index(target_id)
            if env.network.nodes[target_id].isolated.value == "connected":
                type_idx = list(BlueActionType).index(BlueActionType.ISOLATE)
            else:
                type_idx = list(BlueActionType).index(BlueActionType.RESET)
            return target_idx * len(BlueActionType) + type_idx
        else:
            # Monitoring phase
            # pick a node with highest vulnerability to monitor
            target_id = max(node_ids, key=lambda nid: env.network.nodes[nid].vulnerability_rate)
            target_idx = node_ids.index(target_id)
            type_idx = list(BlueActionType).index(BlueActionType.MONITOR)
            return target_idx * len(BlueActionType) + type_idx


class ProactiveBlueAgent(BlueAgent):
    """Proactive blue agent — pre-emptively patches high-vuln nodes and monitors critical assets."""

    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        if env.network is None:
            raise RuntimeError("Environment not initialized")
        node_ids = list(env.network.nodes.keys())

        # Pre-emptive patch: highest vulnerability non-critical node
        candidates = [
            (nid, n.vulnerability_rate)
            for nid, n in env.network.nodes.items()
            if not n.is_critical_asset and not n.is_compromised()
        ]
        if candidates:
            candidates.sort(key=lambda x: -x[1])
            target_id = candidates[0][0]
            target_idx = node_ids.index(target_id)
            type_idx = list(BlueActionType).index(BlueActionType.PATCH)
            return target_idx * len(BlueActionType) + type_idx

        # Critical asset monitoring
        critical = [nid for nid in node_ids if env.network.nodes[nid].is_critical_asset]
        if critical:
            target_idx = node_ids.index(critical[0])
            type_idx = list(BlueActionType).index(BlueActionType.MONITOR)
            return target_idx * len(BlueActionType) + type_idx

        # Fallback
        target_idx = 0
        type_idx = list(BlueActionType).index(BlueActionType.WAIT)
        return target_idx * len(BlueActionType) + type_idx
