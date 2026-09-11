# violet/transition.py — action execution and state transition engine

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from violet.network import Network
from violet.rewards import RewardEngine
from violet.types import (
    Action,
    ActionTarget,
    AlertLevel,
    ATTACKER_AGENT,
    BlueActionType,
    CompromiseStatus,
    DEFENDER_AGENT,
    IsolationStatus,
    RedActionType,
    ServiceStatus,
)


def _die_roll(rate: float, rng: random.Random) -> bool:
    """Stochastic success check against *rate* in [0,1]."""
    return rng.random() < rate


@dataclass
class TransitionContext:
    """Mutable context accumulated during a step."""

    step: int = 0
    timestamp: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    # flags consumed by the reward engine
    detected: bool = False
    honeypot_triggered: bool = False
    terminated: bool = False
    attack_detected: bool = False
    attack_contained: bool = False
    host_recovered: bool = False
    critical_asset_protected: bool = False
    attacker_removed: bool = False
    unnecessary_isolation: bool = False
    unnecessary_patch: bool = False
    unnecessary_reset: bool = False
    unnecessary_block: bool = False
    unnecessary_restore: bool = False
    false_positive: bool = False
    service_downtime: bool = False
    critical_asset_compromised: bool = False
    new_state_changed: str = ""  # descriptive label of a major state change
    result: str = ""  # action-level result label


@dataclass
class TransitionEngine:
    """Applies actions to the network, emits events, returns reward context."""

    network: Network
    reward_engine: RewardEngine
    seed: int | None = None

    def _rng(self, seed: int | None = None) -> random.Random:
        if seed is not None:
            return random.Random(seed)
        if self.seed is not None:
            return random.Random(self.seed)
        return random.Random()

    # ------------------------------------------------------------------
    # Attacker actions
    # ------------------------------------------------------------------
    def execute_red_action(
        self,
        action: Action,
        context: TransitionContext,
        rng: random.Random | None = None,
    ) -> TransitionContext:
        if not action.is_red():
            raise ValueError("execute_red_action requires a red action")
        if action.target is None:
            raise ValueError("red actions require a target node")

        target = action.target.node_id
        if target not in self.network.nodes:
            context.result = "invalid_target"
            self._emit_event(context, action, "invalid_target", "target not in network")
            return context

        node = self.network.nodes[target]
        rng = rng or self._rng()

        if action.action_type == RedActionType.SCAN:
            return self._red_scan(action, node, context, rng)
        if action.action_type == RedActionType.EXPLOIT:
            return self._red_exploit(action, node, context, rng)
        if action.action_type == RedActionType.PRIVILEGE_ESCALATE:
            return self._red_priv_esc(action, node, context, rng)
        if action.action_type == RedActionType.LATERAL_MOVE:
            return self._red_lateral(action, node, context, rng)
        if action.action_type == RedActionType.EXFILTRATE:
            return self._red_exfiltrate(action, node, context, rng)
        if action.action_type == RedActionType.WAIT:
            return self._red_wait(action, node, context, rng)

        context.result = "unknown_action"
        self._emit_event(context, action, "unknown_action", action.action_type.value)
        return context

    def _red_scan(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Scan reveals basic info about a reachable node."""
        if not node.attacker_discovered:
            node.attacker_discovered = True
            context.result = "recon_success"
            self._emit_event(context, action, "recon_success", f"discovered {node.id}")
        else:
            context.result = "recon_already_known"
            self._emit_event(context, action, "recon_already_known", f"{node.id} already known")
        return context

    def _red_exploit(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Attempt to exploit a vulnerability on the target node."""
        if node.is_compromised():
            context.result = "exploit_failed"
            self._emit_event(context, action, "exploit_failed", "already compromised")
            return context

        # Reachability gate: the attacker can only exploit a node it can actually
        # reach —
        #   * an INTERNET-FACING node (initial access from outside), or
        #   * a node ADJACENT to a non-isolated compromised foothold.
        # Isolated targets accept nothing. Without this gate an attacker could
        # compromise the database directly from the internet (observed: an
        # untrained PPO policy "won" 10/10 episodes in ~4 steps by hammering
        # EXPLOIT_DATABASE — a pure environment loophole, not learning).
        has_foothold = any(
            n.is_compromised() and n.isolated != IsolationStatus.ISOLATED
            for n in self.network.nodes.values()
        )
        if node.isolated == IsolationStatus.ISOLATED:
            reachable = False
        elif has_foothold:
            adjacent = set(node.connections) | set(node.incoming_connections)
            reachable = any(
                nid in adjacent
                for nid, n in self.network.nodes.items()
                if n.is_compromised() and n.isolated != IsolationStatus.ISOLATED
            )
        else:
            reachable = "internet" in node.incoming_connections
        if not reachable:
            context.result = "exploit_failed"
            self._emit_event(context, action, "exploit_failed", f"no network path to {node.id}")
            return context

        if node.is_honeypot:
            context.honeypot_triggered = True
            context.result = "honeypot_triggered"
            # Honeypot: attack fails and attacker gets a penalty
            node.alert_level = AlertLevel.HIGH
            self.network.active_alerts[node.id] = AlertLevel.HIGH
            self._emit_event(
                context,
                action,
                "honeypot_triggered",
                f"decoy detected on {node.id}",
            )
            return context

        # Attempt exploit: use the most favorable known vulnerability if any,
        # otherwise fall back to the node's base vulnerability rate.
        vuln_rate = node.vulnerability_rate
        if node.vulnerabilities:
            vuln_rate = max(v.base_success_rate for v in node.vulnerabilities)

        success = _die_roll(vuln_rate, rng)

        if success:
            node.compromise_status = CompromiseStatus.COMPROMISED
            node.attacker_knows_vulnerabilities = True
            context.result = "exploit_success"
            self._emit_event(
                context,
                action,
                "exploit_success",
                f"compromised {node.id}",
                details={"vuln_rate_used": vuln_rate},
            )
            # Compromise of a critical asset is a major state change
            if node.is_critical_asset:
                context.critical_asset_compromised = True
                context.new_state_changed = "critical_asset_compromised"
        else:
            context.result = "exploit_failed"
            self._emit_event(
                context,
                action,
                "exploit_failed",
                f"exploit failed on {node.id}",
                details={"vuln_rate_used": vuln_rate},
            )

        return context

    def _red_priv_esc(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Attempt privilege escalation on an already-compromised node."""
        if not node.is_compromised():
            context.result = "priv_esc_failed"
            self._emit_event(context, action, "priv_esc_failed", "node not compromised")
            return context

        if node.compromise_status == CompromiseStatus.PRIVILEGES_ESCALATED:
            context.result = "priv_esc_already"
            self._emit_event(context, action, "priv_esc_already", "already escalated")
            return context

        success = _die_roll(0.7, rng)
        if success:
            node.compromise_status = CompromiseStatus.PRIVILEGES_ESCALATED
            context.result = "priv_esc_success"
            self._emit_event(context, action, "priv_esc_success", f"rooted {node.id}")
        else:
            context.result = "priv_esc_failed"
            self._emit_event(context, action, "priv_esc_failed", f"priv esc failed on {node.id}")

        return context

    def _red_lateral(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Move laterally to the target (destination) node.

        Semantics (v3, hop-by-hop): the action target is the DESTINATION, and the
        source must be a DIRECTLY ADJACENT compromised foothold (an actual one-hop
        lateral movement). Multi-hop "movement" in a single action is rejected —
        the attacker must compromise each intermediate hop. No random destination
        or random source selection remains."""
        destination_id = node.id

        if node.is_compromised():
            context.result = "lateral_failed"
            self._emit_event(context, action, "lateral_failed", f"{destination_id} already compromised")
            return context

        if node.is_honeypot:
            context.honeypot_triggered = True
            context.result = "honeypot_triggered"
            node.alert_level = AlertLevel.HIGH
            self.network.active_alerts[destination_id] = AlertLevel.HIGH
            self._emit_event(context, action, "honeypot_triggered", f"decoy on {destination_id}")
            return context

        # Hop-by-hop: the source must be a compromised DIRECT NEIGHBOR of the
        # destination (either direction of an edge counts as adjacency), and NEITHER
        # endpoint may be isolated — isolation severs the attacker's ability to
        # project movement. Regression guard: without the isolation check, an
        # attacker could tunnel through an isolated foothold (observed seed=42).
        # List order (not set) keeps source selection deterministic across runs.
        destination_node = self.network.nodes[destination_id]
        if destination_node.isolated == IsolationStatus.ISOLATED:
            context.result = "lateral_failed"
            self._emit_event(context, action, "lateral_failed", f"{destination_id} is isolated")
            return context

        adjacent = list(destination_node.connections) + [
            nid for nid in destination_node.incoming_connections
            if nid not in destination_node.connections
        ]
        sources = [
            nid
            for nid in adjacent
            if self.network.nodes[nid].is_compromised()
            and self.network.nodes[nid].isolated != IsolationStatus.ISOLATED
        ]
        if not sources:
            context.result = "lateral_failed"
            self._emit_event(
                context,
                action,
                "lateral_failed",
                f"no adjacent compromised foothold for hop to {destination_id}",
            )
            return context

        # Deterministic source: first adjacent foothold in insertion order.
        source_id = sources[0]

        success = _die_roll(node.vulnerability_rate, rng)
        if success:
            node.compromise_status = CompromiseStatus.COMPROMISED
            context.result = "lateral_success"
            context.new_state_changed = f"lateral_to_{destination_id}"
            if node.is_critical_asset:
                context.critical_asset_compromised = True
                context.new_state_changed = "critical_asset_compromised"
            self._emit_event(
                context,
                action,
                "lateral_success",
                f"moved {source_id} -> {destination_id}",
                details={"source": source_id, "destination": destination_id},
            )
        else:
            context.result = "lateral_failed"
            self._emit_event(
                context,
                action,
                "lateral_failed",
                f"lateral failed {source_id} -> {destination_id}",
                details={"source": source_id, "destination": destination_id},
            )

        return context

    def _red_exfiltrate(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Attempt to exfiltrate data from a compromised node (terminal-ish objective)."""
        if not node.is_compromised():
            context.result = "exfiltration_failed"
            self._emit_event(context, action, "exfiltration_failed", "node not compromised")
            return context

        if node.is_honeypot:
            context.honeypot_triggered = True
            context.result = "honeypot_triggered"
            self._emit_event(context, action, "honeypot_triggered", f"decoy on {node.id}")
            return context

        success = _die_roll(0.6, rng)
        if success:
            node.compromise_status = CompromiseStatus.EXFILTRATED
            context.result = "exfiltration_success"
            self._emit_event(context, action, "exfiltration_success", f"exfiltrated {node.id}")
        else:
            context.result = "exfiltration_failed"
            self._emit_event(context, action, "exfiltration_failed", f"exfiltration failed {node.id}")

        return context

    def _red_wait(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        context.result = "wait"
        self._emit_event(context, action, "wait", "attacker waits")
        return context

    # ------------------------------------------------------------------
    # Defender actions
    # ------------------------------------------------------------------
    def execute_blue_action(
        self,
        action: Action,
        context: TransitionContext,
        rng: random.Random | None = None,
    ) -> TransitionContext:
        if not action.is_blue():
            raise ValueError("execute_blue_action requires a blue action")
        if action.target is None:
            raise ValueError("blue actions require a target node")

        target = action.target.node_id
        if target not in self.network.nodes:
            context.result = "invalid_target"
            self._emit_event(context, action, "invalid_target", "target not in network")
            return context

        node = self.network.nodes[target]
        rng = rng or self._rng()

        if action.action_type == BlueActionType.MONITOR:
            return self._blue_monitor(action, node, context, rng)
        if action.action_type == BlueActionType.PATCH:
            return self._blue_patch(action, node, context, rng)
        if action.action_type == BlueActionType.BLOCK:
            return self._blue_block(action, node, context, rng)
        if action.action_type == BlueActionType.ISOLATE:
            return self._blue_isolate(action, node, context, rng)
        if action.action_type == BlueActionType.RESET:
            return self._blue_reset(action, node, context, rng)
        if action.action_type == BlueActionType.DEPLOY_HONEYPOT:
            return self._blue_deploy_honeypot(action, node, context, rng)
        if action.action_type == BlueActionType.RESTORE:
            return self._blue_restore(action, node, context, rng)
        if action.action_type == BlueActionType.WAIT:
            return self._blue_wait(action, node, context, rng)

        context.result = "unknown_action"
        self._emit_event(context, action, "unknown_action", action.action_type.value)
        return context

    def _blue_monitor(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Monitoring can detect compromise with some probability.

        Detection-reward semantics: `attack_detected` fires only on the FIRST
        detection of this node's compromise (defender_aware_of_compromise flag).
        Repeat detections on a known-compromised node raise/refresh the alert but
        pay no reward — this closes the detection-farming loop where a defender
        could re-monitor an isolated compromised node indefinitely for +5/step.
        """
        detect_prob = 0.1 if not node.is_compromised() else 0.5
        detected = _die_roll(detect_prob, rng)
        if detected:
            node.alert_level = AlertLevel.HIGH
            self.network.active_alerts[node.id] = AlertLevel.HIGH
            if node.is_compromised() and not node.defender_aware_of_compromise:
                node.defender_aware_of_compromise = True
                context.detected = True
                context.attack_detected = True
                context.result = "monitor_detected"
                self._emit_event(context, action, "monitor_detected", f"detected compromise on {node.id}")
            elif not node.is_compromised():
                # False positive: alert raised, but no detection reward
                context.false_positive = True
                context.result = "monitor_false_positive"
                self._emit_event(context, action, "monitor_false_positive", f"false alarm on {node.id}")
            else:
                context.result = "monitor_repeat_detect"
                self._emit_event(context, action, "monitor_repeat_detect", f"{node.id} compromise already known")
        else:
            context.result = "monitor_no_detect"
            self._emit_event(context, action, "monitor_no_detect", f"no activity on {node.id}")
        return context

    def _blue_patch(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Patch a node's vulnerability (removes known vulns, reduces base rate)."""
        if node.service_status == ServiceStatus.DOWN:
            context.service_downtime = True
            context.unnecessary_patch = True
            context.result = "patch_failed"
            self._emit_event(context, action, "patch_failed_down", f"{node.id} already down")
            return context

        if node.service_status == ServiceStatus.PATCHED:
            context.unnecessary_patch = True
            context.result = "patch_already"
            self._emit_event(context, action, "patch_already", f"{node.id} already patched")
            return context

        success = _die_roll(0.8, rng)
        if success:
            # Patch reduces vulnerability rate AND mitigates ALL listed
            # vulnerabilities. Regression guard: the previous filter kept
            # privilege-escalation-available vulns in the exploit pool, which
            # made PATCH nearly useless (web remained exploitable at 0.8 after
            # a successful patch — observed on seed=42).
            node.vulnerability_rate = min(node.vulnerability_rate, 0.05)
            node.vulnerabilities = []
            # If we had attacker-discovered vulns, clear knowledge
            node.attacker_knows_vulnerabilities = False
            node.service_status = ServiceStatus.PATCHED
            context.result = "patch_success"
            self._emit_event(context, action, "patch_success", f"patched {node.id}")
        else:
            context.result = "patch_failed"
            self._emit_event(context, action, "patch_failed", f"patch failed on {node.id}")
        return context

    def _blue_block(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Block connections to/from a node via firewall rule (conceptual)."""
        if node.service_status == ServiceStatus.DOWN:
            context.unnecessary_block = True
            context.result = "block_failed"
            self._emit_event(context, action, "block_failed", f"{node.id} already down")
            return context

        # Blocking is modeled as an alert + partial restriction for now
        node.alert_level = AlertLevel.HIGH
        self.network.active_alerts[node.id] = AlertLevel.HIGH
        self.network.firewall_rules.append({"type": "block", "target": node.id})
        context.result = "block_applied"
        self._emit_event(context, action, "block_applied", f"blocked {node.id}")
        return context

    def _blue_isolate(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Isolate a node from the network (removes connectivity)."""
        if node.isolated == IsolationStatus.ISOLATED:
            context.unnecessary_isolation = True
            context.result = "isolate_already"
            self._emit_event(context, action, "isolate_already", f"{node.id} already isolated")
            return context

        success = _die_roll(0.9, rng)
        if success:
            node.isolated = IsolationStatus.ISOLATED
            context.result = "isolate_success"
            context.attack_contained = True
            node.alert_level = AlertLevel.HIGH
            self.network.active_alerts[node.id] = AlertLevel.HIGH
            self._emit_event(context, action, "isolate_success", f"isolated {node.id}")
            if node.service_status == ServiceStatus.ACTIVE:
                context.service_downtime = True
            # Check whether this isolation protects a critical asset
            if node.is_critical_asset and not node.is_compromised():
                context.critical_asset_protected = True
            # If the node was compromised, isolation contains it
            if node.is_compromised():
                context.attack_contained = True
        else:
            context.result = "isolate_failed"
            self._emit_event(context, action, "isolate_failed", f"isolate failed on {node.id}")
        return context

    def _blue_reset(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Reset a compromised node to a clean state."""
        if not node.is_compromised():
            context.unnecessary_reset = True
            context.result = "reset_already"
            self._emit_event(context, action, "reset_already", f"{node.id} not compromised")
            return context

        success = _die_roll(0.95, rng)
        if success:
            node.compromise_status = CompromiseStatus.NONE
            node.isolated = IsolationStatus.CONNECTED
            node.service_status = ServiceStatus.ACTIVE
            node.alert_level = AlertLevel.NONE
            node.attacker_knows_vulnerabilities = False
            node.attacker_knows_credentials = False
            node.defender_aware_of_compromise = False  # fresh node: awareness stale
            # Remove active alerts for this node
            self.network.active_alerts.pop(node.id, None)
            context.host_recovered = True
            context.result = "reset_success"
            self._emit_event(context, action, "reset_success", f"reset {node.id}")
            if node.is_critical_asset:
                context.critical_asset_protected = True
        else:
            context.result = "reset_failed"
            self._emit_event(context, action, "reset_failed", f"reset failed on {node.id}")
        return context

    def _blue_deploy_honeypot(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Deploy a honeypot decoy on a node (marks node as honeypot)."""
        if node.is_honeypot:
            context.result = "honeypot_already"
            self._emit_event(context, action, "honeypot_already", f"{node.id} already a honeypot")
            return context

        success = _die_roll(0.6, rng)
        if success:
            node.is_honeypot = True
            node.vulnerability_rate = 0.9
            node.attacker_knows_vulnerabilities = False
            context.result = "honeypot_deployed"
            self._emit_event(context, action, "honeypot_deployed", f"deployed honeypot on {node.id}")
        else:
            context.result = "honeypot_failed"
            self._emit_event(context, action, "honeypot_failed", f"honeypot deployment failed on {node.id}")
        return context

    def _blue_restore(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        """Restore service uptime on a node."""
        if node.service_status == ServiceStatus.ACTIVE:
            context.unnecessary_restore = True
            context.result = "restore_already"
            self._emit_event(context, action, "restore_already", f"{node.id} already active")
            return context

        success = _die_roll(0.85, rng)
        if success:
            node.service_status = ServiceStatus.ACTIVE
            context.service_downtime = False
            context.result = "restore_success"
            self._emit_event(context, action, "restore_success", f"restored {node.id}")
        else:
            context.result = "restore_failed"
            self._emit_event(context, action, "restore_failed", f"restore failed on {node.id}")
        return context

    def _blue_wait(
        self,
        action: Action,
        node: Any,
        context: TransitionContext,
        rng: random.Random,
    ) -> TransitionContext:
        context.result = "wait"
        self._emit_event(context, action, "wait", "defender waits")
        return context

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------
    def _emit_event(
        self,
        context: TransitionContext,
        action: Action,
        result: str,
        detail: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        ev = {
            "step": context.step,
            "timestamp": context.timestamp,
            "agent": action.agent,
            "action": action.action_type.value,
            "target": action.target.node_id if action.target else None,
            "result": result,
            "details": {**details, "detail": detail} if details else {"detail": detail},
        }
        context.events.append(ev)

    # ------------------------------------------------------------------
    # Termination check
    # ------------------------------------------------------------------
    def should_terminate(self) -> bool:
        """Episode terminates when a critical asset is compromised (Red wins) or the
        attacker is removed/reset from all compromised nodes and no active compromise
        remains. Phase 1 starts simple: critical-asset compromise = terminal."""
        # Terminal condition A: critical asset compromised
        if self.network.critical_assets_compromised() > 0:
            return True
        # Terminal condition B: Red has no foothold (all compromises cleared) — optional
        # for simplicity we also allow max step truncation externally.
        return False
