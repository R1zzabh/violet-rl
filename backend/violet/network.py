# violet/network.py — cyber network graph and node model

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from violet.types import (
    AlertLevel,
    CompromiseStatus,
    Direction,
    IsolationStatus,
    ServiceStatus,
)


@dataclass
class Vulnerability:
    """A known vulnerability on a node: optional deterministic exploit check."""

    id: str
    description: str
    base_success_rate: float  # 0..1, modified by attacker capability / defender state
    privilege_escalation_available: bool = False


@dataclass
class Credential:
    """Credential knowledge held by the attacker (part of attacker observation)."""

    id: str
    node_id: str
    privilege_level: str  # e.g. "none", "user", "admin"
    discovered: bool = False


@dataclass
class Service:
    """A running service on a node."""

    name: str
    status: ServiceStatus = ServiceStatus.ACTIVE
    port: int | None = None
    exposed_to: list[str] = field(default_factory=list)


@dataclass
class Node:
    """A logical host/asset in the simulated network.

    Properties match the spec example:
        WEB: value=20, vulnerability=0.8, compromised=false, ...
    """

    id: str
    label: str
    node_type: str = "host"  # host | router | firewall | honeypot | database | gateway
    value: float = 10.0  # importance/criticality
    vulnerability_rate: float = 0.5  # base exploit success rate if no specific vuln
    compromise_status: CompromiseStatus = CompromiseStatus.NONE
    alert_level: AlertLevel = AlertLevel.NONE
    service_status: ServiceStatus = ServiceStatus.ACTIVE
    isolated: IsolationStatus = IsolationStatus.CONNECTED
    is_honeypot: bool = False
    is_critical_asset: bool = False

    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    services: list[Service] = field(default_factory=list)
    credentials: list[Credential] = field(default_factory=list)

    connections: list[str] = field(default_factory=list)  # node IDs this node can reach
    incoming_connections: list[str] = field(default_factory=list)  # who can reach this node

    metadata: dict[str, Any] = field(default_factory=dict)

    # attacker knowledge (separate from ground truth — supports partial observability)
    attacker_discovered: bool = False
    attacker_knows_vulnerabilities: bool = False
    attacker_knows_credentials: bool = False

    # defender knowledge: has Blue's monitoring detected THIS node's compromise?
    # Cleared on reset (clean node = stale awareness). Prevents detection-reward farming.
    defender_aware_of_compromise: bool = False

    def is_compromised(self) -> bool:
        return self.compromise_status in (
            CompromiseStatus.COMPROMISED,
            CompromiseStatus.PRIVILEGES_ESCALATED,
            CompromiseStatus.EXFILTRATED,
        )

    def can_reach(self, other_id: str, network: "Network") -> bool:
        """Can this node reach *other_id* given isolation and connectivity?"""
        if self.isolated == IsolationStatus.ISOLATED:
            return False
        if other_id == self.id:
            return True
        if other_id in self.connections:
            target = network.nodes.get(other_id)
            if target is None:
                return False
            if target.isolated == IsolationStatus.ISOLATED:
                return False
            return True
        # indirect reachability (multi-hop) — used for lateral movement
        return network._is_reachable(self.id, other_id, visited=set())

    def to_observation(self) -> dict[str, Any]:
        """Public (attacker-visible) snapshot of this node."""
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type,
            "value": self.value,
            "compromised": self.is_compromised(),
            "compromise_status": self.compromise_status.value,
            "alert_level": self.alert_level.value,
            "service_status": self.service_status.value,
            "isolated": self.isolated.value,
            "is_honeypot": self.is_honeypot,
            "is_critical_asset": self.is_critical_asset,
            "discovered": self.attacker_discovered,
            "connections": list(self.connections),  # only known connections
            "vulnerabilities_known": self.attacker_knows_vulnerabilities,
            "credentials_known": self.attacker_knows_credentials,
        }


@dataclass
class Network:
    """The simulated enterprise network topology and node state.

    Internal (god-mode) state is richer than any single agent's observation.
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    firewall_rules: list[dict[str, Any]] = field(default_factory=list)
    active_alerts: dict[str, AlertLevel] = field(default_factory=dict)
    episode_step: int = 0

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def connect_bidirectional(self, a: str, b: str) -> None:
        """Add bidirectional connectivity between two existing nodes."""
        if a not in self.nodes or b not in self.nodes:
            raise KeyError(f"Both nodes must exist: {a!r}, {b!r}")
        self.nodes[a].connections.append(b)
        self.nodes[b].connections.append(a)
        self.nodes[a].incoming_connections.append(b)
        self.nodes[b].incoming_connections.append(a)

    def connect_unidirectional(self, source: str, target: str) -> None:
        """Add unidirectional connectivity source -> target."""
        if source not in self.nodes or target not in self.nodes:
            raise KeyError(f"Both nodes must exist: {source!r}, {target!r}")
        self.nodes[source].connections.append(target)
        self.nodes[target].incoming_connections.append(source)

    def clone(self) -> "Network":
        """Deep clone for reset / determinism."""
        return copy.deepcopy(self)

    def _is_reachable(self, src: str, dst: str, visited: set[str]) -> bool:
        if src == dst:
            return True
        if src in visited:
            return False
        visited.add(src)
        node = self.nodes.get(src)
        if node is None or node.isolated == IsolationStatus.ISOLATED:
            return False
        for nid in node.connections:
            if self._is_reachable(nid, dst, visited):
                return True
        return False

    def reachable_from(self, src: str) -> set[str]:
        """Set of node IDs reachable from *src* (excluding isolated targets)."""
        reachable: set[str] = set()
        stack = [src]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            node = self.nodes.get(current)
            if node is None:
                continue
            if node.isolated == IsolationStatus.ISOLATED and current != src:
                continue
            for nid in node.connections:
                target = self.nodes.get(nid)
                if target is None:
                    continue
                if target.isolated == IsolationStatus.ISOLATED:
                    continue
                if nid not in reachable:
                    reachable.add(nid)
                    stack.append(nid)
        return reachable

    def compromised_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.is_compromised())

    def critical_assets_compromised(self) -> int:
        return sum(
            1
            for n in self.nodes.values()
            if n.is_critical_asset and n.is_compromised()
        )

    def total_value(self) -> float:
        return sum(n.value for n in self.nodes.values())

    def compromised_value(self) -> float:
        return sum(n.value for n in self.nodes.values() if n.is_compromised())

    def has_honeypot(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        return node is not None and node.is_honeypot

    def reset_step(self) -> None:
        self.episode_step = 0

    def snapshot(self) -> dict[str, Any]:
        """Full (internal) snapshot. Exposed via render / debugging, not agent obs."""
        return {
            "step": self.episode_step,
            "nodes": {
                nid: {
                    "id": n.id,
                    "label": n.label,
                    "node_type": n.node_type,
                    "value": n.value,
                    "compromise_status": n.compromise_status.value,
                    "alert_level": n.alert_level.value,
                    "service_status": n.service_status.value,
                    "isolated": n.isolated.value,
                    "is_honeypot": n.is_honeypot,
                    "is_critical_asset": n.is_critical_asset,
                    "connections": n.connections,
                    "incoming": n.incoming_connections,
                    "vulnerabilities": [v.id for v in n.vulnerabilities],
                    "services": [s.name for s in n.services if s.status == ServiceStatus.ACTIVE],
                }
                for nid, n in self.nodes.items()
            },
            "firewall_rules": list(self.firewall_rules),
            "active_alerts": {k: v.value for k, v in self.active_alerts.items()},
        }
