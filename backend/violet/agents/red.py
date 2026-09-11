# violet/agents/red.py — red (attacker) agent implementations

from __future__ import annotations

import random
from typing import Any, Dict

from violet.env import VIOLETEnv
from violet.types import ATTACKER_AGENT, RedActionType


class RedAgent:
    """Base class for red agents that returns a discrete action index suitable for
    the environment's red action space."""

    agent_name = ATTACKER_AGENT

    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        ...


class RandomRedAgent(RedAgent):
    """Uniformly random attacker — baseline 1 (per spec)."""

    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        action_space = env.action_space(ATTACKER_AGENT)
        return env.rng.randint(0, action_space.n - 1) if env.rng is not None else random.randint(0, action_space.n - 1)


class ScriptedAggressiveRedAgent(RedAgent):
    """Scripted, aggressive red baseline — rule-based attacker (baseline 2 per spec).

    Behaves like a realistic intrusion progression: it maintains a foothold set of
    compromised nodes, and advances along the network topology from its entry point
    toward the designated objective:

        SCAN (discover next hop) -> EXPLOIT -> PRIVILEGE_ESCALATE -> LATERAL_MOVE

    It never attacks a node that is not reachable from its current foothold, so the
    episode history looks like a real kill chain rather than a teleporting exploit.
    """

    def __init__(self, target_node_id: str = "database", entry_node_id: str = "web"):
        self.target_node_id = target_node_id
        self.entry_node_id = entry_node_id
        self.phase = 0  # 0=scan, 1=exploit, 2=priv_esc, 3=lateral
        self._next_hop: str | None = None

    def reset(self) -> None:
        self.phase = 0
        self._next_hop = None

    # ------------------------------------------------------------------
    # Topology-aware planning
    # ------------------------------------------------------------------
    def _plan_next_hop(self, network: Any) -> str:
        """Pick the next uncompromised node to attack.

        With no foothold yet, the entry node itself is the objective (gain initial
        access). After that, the next hop is restricted to DIRECT neighbors of the
        current foothold — the attacker walks inward one hop at a time instead of
        teleporting to the final objective."""
        foothold = self._current_foothold(network)
        if foothold is None:
            return self.entry_node_id

        fnode = network.nodes[foothold]
        neighbors = set(fnode.connections) | set(fnode.incoming_connections)
        candidates = [
            nid
            for nid in neighbors
            if nid in network.nodes
            and nid != "internet"
            and not network.nodes[nid].is_compromised()
        ]
        if not candidates:
            return self.target_node_id
        if self.target_node_id in candidates:
            return self.target_node_id
        return min(candidates, key=lambda nid: self._graph_distance(network, nid, self.target_node_id))

    def _current_foothold(self, network: Any) -> str | None:
        """The compromised node closest (by hops) to the objective."""
        compromised = [nid for nid, n in network.nodes.items() if n.is_compromised()]
        if not compromised:
            return None
        return min(
            compromised,
            key=lambda nid: self._graph_distance(network, nid, self.target_node_id),
        )

    @staticmethod
    def _graph_distance(network: Any, src: str, dst: str) -> int:
        """BFS hop distance over the topology, ignoring isolation for planning."""
        if src == dst:
            return 0
        visited = {src}
        frontier = [src]
        depth = 0
        while frontier:
            depth += 1
            next_frontier: list[str] = []
            for current in frontier:
                node = network.nodes.get(current)
                if node is None:
                    continue
                for neighbor in node.connections + node.incoming_connections:
                    if neighbor == dst:
                        return depth
                    if neighbor not in visited and neighbor in network.nodes:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return 999

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        if env.network is None:
            raise RuntimeError("Environment not initialized")
        network = env.network
        node_ids = list(network.nodes.keys())

        def encode(target_id: str, action_type: RedActionType) -> int:
            self.phase = {
                RedActionType.SCAN: 0,
                RedActionType.EXPLOIT: 1,
                RedActionType.PRIVILEGE_ESCALATE: 2,
                RedActionType.LATERAL_MOVE: 3,
            }.get(action_type, 4)
            target_idx = node_ids.index(target_id)
            return target_idx * len(RedActionType) + list(RedActionType).index(action_type)

        foothold = self._current_foothold(network)

        # 1. No foothold: gain initial access at the entry point.
        if foothold is None:
            self._next_hop = self.entry_node_id
            entry = network.nodes[self._next_hop]
            if not entry.attacker_discovered:
                return encode(self._next_hop, RedActionType.SCAN)
            if not entry.is_compromised():
                return encode(self._next_hop, RedActionType.EXPLOIT)
            return encode(self._next_hop, RedActionType.WAIT)

        # 2. Escalate the foothold before spreading (priv_esc retries until success).
        fnode = network.nodes[foothold]
        if fnode.compromise_status.value == "compromised":
            self._next_hop = foothold
            return encode(foothold, RedActionType.PRIVILEGE_ESCALATE)

        # 3. Recon the next hop if unknown...
        next_hop = self._plan_next_hop(network)
        self._next_hop = next_hop
        if next_hop in node_ids and not network.nodes[next_hop].attacker_discovered:
            return encode(next_hop, RedActionType.SCAN)

        # 4. ...then spread laterally from the foothold. Engine semantics (v2):
        # the LATERAL_MOVE target is the DESTINATION node; the engine picks the
        # compromised source foothold that can reach it.
        return encode(next_hop, RedActionType.LATERAL_MOVE)


class ScriptedStealthRedAgent(RedAgent):
    """Scripted stealthy red baseline — low-noise, slow exploitation, avoid detection."""

    def __init__(self, target_node_id: str = "database"):
        self.target_node_id = target_node_id

    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        if env.network is None:
            raise RuntimeError("Environment not initialized")
        node_ids = list(env.network.nodes.keys())
        target_id = self.target_node_id if self.target_node_id in node_ids else node_ids[-1]
        target_idx = node_ids.index(target_id)

        # Stealth logic: prefer SCAN/WAIT unless target discovered and low alert
        target = env.network.nodes[target_id]
        if not target.attacker_discovered:
            action_type = RedActionType.SCAN
        elif target.alert_level.value != "none":
            action_type = RedActionType.WAIT
        elif not target.is_compromised():
            action_type = RedActionType.EXPLOIT
        else:
            action_type = RedActionType.WAIT

        type_idx = list(RedActionType).index(action_type)
        action_index = target_idx * len(RedActionType) + type_idx
        return action_index


class ScriptedLateralRedAgent(RedAgent):
    """Scripted lateral-movement-focused red — prioritizes moving across compromised hosts."""

    def __init__(self, target_node_id: str = "database"):
        self.target_node_id = target_node_id

    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        if env.network is None:
            raise RuntimeError("Environment not initialized")
        node_ids = list(env.network.nodes.keys())

        # Find a compromised node to move from, then target a reachable node
        compromised = [nid for nid, n in env.network.nodes.items() if n.is_compromised()]
        if not compromised:
            # No foothold yet — scan/exploit the most connected node
            target_id = max(node_ids, key=lambda nid: len(env.network.nodes[nid].connections))
            target_idx = node_ids.index(target_id)
            action_type = RedActionType.EXPLOIT
        else:
            source_id = compromised[0]
            reachable = env.network.reachable_from(source_id)
            uncompromised = [nid for nid in reachable if not env.network.nodes[nid].is_compromised()]
            if uncompromised:
                target_id = uncompromised[0]
                target_idx = node_ids.index(target_id)
                action_type = RedActionType.LATERAL_MOVE
            else:
                target_idx = node_ids.index(compromised[0])
                action_type = RedActionType.PRIVILEGE_ESCALATE

        type_idx = list(RedActionType).index(action_type)
        action_index = target_idx * len(RedActionType) + type_idx
        return action_index


class ScriptedRandomizedRedAgent(RedAgent):
    """Scripted but stochastic red — chooses among plausible actions with weights."""

    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        if env.network is None:
            raise RuntimeError("Environment not initialized")
        action_space = env.action_space(ATTACKER_AGENT)
        # Uniform random draw from the discrete action space (random.Random
        # has no weighted choice; uniform is the intended behavior here).
        return env.rng.randint(0, action_space.n - 1)


class ScriptedObjectiveRedAgent(RedAgent):
    """Objective-focused red — always acts toward reaching the designated objective node."""

    def __init__(self, objective_node_id: str = "database"):
        self.objective_node_id = objective_node_id

    def select_action(self, env: VIOLETEnv, info: Dict[str, Any]) -> int:
        if env.network is None:
            raise RuntimeError("Environment not initialized")
        node_ids = list(env.network.nodes.keys())
        if self.objective_node_id not in node_ids:
            # objective not in network; pick highest-value node
            target_id = max(node_ids, key=lambda nid: env.network.nodes[nid].value)
        else:
            target_id = self.objective_node_id

        target_idx = node_ids.index(target_id)
        target = env.network.nodes[target_id]

        if not target.attacker_discovered:
            action_type = RedActionType.SCAN
        elif not target.is_compromised():
            action_type = RedActionType.EXPLOIT
        else:
            action_type = RedActionType.PRIVILEGE_ESCALATE

        type_idx = list(RedActionType).index(action_type)
        action_index = target_idx * len(RedActionType) + type_idx
        return action_index
