# violet/rewards.py — configurable reward engine

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from violet.types import ATTACKER_AGENT, DEFENDER_AGENT


# ---------------------------------------------------------------------------
# Reward component weights (configurable — these are initial experimental values)
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """Weights applied by the reward engine.

    These are EXPERIMENTAL starting values. The spec explicitly says we will test
    alternative configurations (sparse, dense, availability-penalty, etc.) and not
    claim optimality.
    """

    # ---- Red (attacker) ----
    red_recon_success: float = 2.0
    red_exploit_success: float = 10.0
    red_priv_esc_success: float = 15.0
    red_lateral_success: float = 10.0
    red_critical_asset_reached: float = 50.0
    red_exfiltration_success: float = 100.0
    red_exploit_failed: float = -3.0
    red_detected: float = -5.0
    red_honeypot_triggered: float = -15.0
    red_agent_terminated: float = -100.0

    # ---- Blue (defender) ----
    # Note: attack_detected pays only on the FIRST detection of a given node's
    # compromise (see TransitionEngine._blue_monitor). Repeat detections are free
    # events, not rewardable — this prevents detection-farming loops.
    blue_attack_detected: float = 5.0
    blue_attack_contained: float = 20.0
    blue_host_recovered: float = 15.0
    blue_critical_asset_protected: float = 50.0
    blue_attacker_removed: float = 100.0
    blue_false_positive: float = -1.0
    blue_unnecessary_isolation: float = -8.0
    blue_unnecessary_patch: float = -4.0
    blue_unnecessary_reset: float = -2.0
    blue_unnecessary_block: float = -2.0
    blue_unnecessary_restore: float = -2.0
    blue_service_downtime: float = -10.0
    blue_critical_asset_compromised: float = -100.0

    # ---- Joint / time ----
    step_penalty: float = -0.1  # small time cost for both

    def to_dict(self) -> dict[str, float]:
        return {
            k: v for k, v in self.__dict__.items() if isinstance(v, float)
        }


# ---------------------------------------------------------------------------
# Reward engine — computes rewards from transition + event context
# ---------------------------------------------------------------------------
@dataclass
class RewardEngine:
    config: RewardConfig = field(default_factory=RewardConfig)

    def red_reward(self, context: dict[str, Any]) -> float:
        """Compute Red's reward for a step given transition context."""
        reward = 0.0

        if context.get("result") == "recon_success":
            reward += self.config.red_recon_success
        if context.get("result") == "exploit_success":
            reward += self.config.red_exploit_success
        if context.get("result") == "priv_esc_success":
            reward += self.config.red_priv_esc_success
        if context.get("result") == "lateral_success":
            reward += self.config.red_lateral_success
        if context.get("new_state_changed") == "critical_asset_compromised":
            reward += self.config.red_critical_asset_reached
        if context.get("result") == "exfiltration_success":
            reward += self.config.red_exfiltration_success

        if context.get("result") == "exploit_failed":
            reward += self.config.red_exploit_failed
        if context.get("detected"):
            reward += self.config.red_detected
        if context.get("honeypot_triggered"):
            reward += self.config.red_honeypot_triggered
        if context.get("terminated"):
            reward += self.config.red_agent_terminated

        reward += self.config.step_penalty
        return reward

    def blue_reward(self, context: dict[str, Any]) -> float:
        """Compute Blue's reward for a step given transition context."""
        reward = 0.0

        if context.get("attack_detected"):
            reward += self.config.blue_attack_detected
        if context.get("attack_contained"):
            reward += self.config.blue_attack_contained
        if context.get("host_recovered"):
            reward += self.config.blue_host_recovered
        if context.get("critical_asset_protected"):
            reward += self.config.blue_critical_asset_protected
        if context.get("attacker_removed"):
            reward += self.config.blue_attacker_removed

        if context.get("unnecessary_isolation"):
            reward += self.config.blue_unnecessary_isolation
        if context.get("unnecessary_patch"):
            reward += self.config.blue_unnecessary_patch
        if context.get("unnecessary_reset"):
            reward += self.config.blue_unnecessary_reset
        if context.get("unnecessary_block"):
            reward += self.config.blue_unnecessary_block
        if context.get("unnecessary_restore"):
            reward += self.config.blue_unnecessary_restore
        if context.get("false_positive"):
            reward += self.config.blue_false_positive
        if context.get("service_downtime"):
            reward += self.config.blue_service_downtime
        if context.get("critical_asset_compromised"):
            reward += self.config.blue_critical_asset_compromised

        reward += self.config.step_penalty
        return reward

    def reward(
        self,
        agent: str,
        context: dict[str, Any],
    ) -> float:
        if agent == ATTACKER_AGENT:
            return self.red_reward(context)
        if agent == DEFENDER_AGENT:
            return self.blue_reward(context)
        raise ValueError(f"Unknown agent: {agent!r}")

    def pretty(self, agent: str, context: dict[str, Any]) -> dict[str, Any]:
        """Human-readable decomposition of the reward for debugging traces."""
        if agent == ATTACKER_AGENT:
            total = self.red_reward(context)
            components = self._red_components(context)
        elif agent == DEFENDER_AGENT:
            total = self.blue_reward(context)
            components = self._blue_components(context)
        else:
            raise ValueError(f"Unknown agent: {agent!r}")
        return {"total": total, "components": components}

    def _red_components(self, context: dict[str, Any]) -> dict[str, float]:
        c = self.config
        return {
            "recon_success": c.red_recon_success if context.get("result") == "recon_success" else 0.0,
            "exploit_success": c.red_exploit_success if context.get("result") == "exploit_success" else 0.0,
            "priv_esc_success": c.red_priv_esc_success if context.get("result") == "priv_esc_success" else 0.0,
            "lateral_success": c.red_lateral_success if context.get("result") == "lateral_success" else 0.0,
            "critical_asset_reached": c.red_critical_asset_reached if context.get("new_state_changed") == "critical_asset_compromised" else 0.0,
            "exfiltration_success": c.red_exfiltration_success if context.get("result") == "exfiltration_success" else 0.0,
            "exploit_failed": c.red_exploit_failed if context.get("result") == "exploit_failed" else 0.0,
            "detected": c.red_detected if context.get("detected") else 0.0,
            "honeypot_triggered": c.red_honeypot_triggered if context.get("honeypot_triggered") else 0.0,
            "terminated": c.red_agent_terminated if context.get("terminated") else 0.0,
            "step_penalty": c.step_penalty,
        }

    def _blue_components(self, context: dict[str, Any]) -> dict[str, float]:
        c = self.config
        return {
            "attack_detected": c.blue_attack_detected if context.get("attack_detected") else 0.0,
            "attack_contained": c.blue_attack_contained if context.get("attack_contained") else 0.0,
            "host_recovered": c.blue_host_recovered if context.get("host_recovered") else 0.0,
            "critical_asset_protected": c.blue_critical_asset_protected if context.get("critical_asset_protected") else 0.0,
            "attacker_removed": c.blue_attacker_removed if context.get("attacker_removed") else 0.0,
            "unnecessary_isolation": c.blue_unnecessary_isolation if context.get("unnecessary_isolation") else 0.0,
            "unnecessary_patch": c.blue_unnecessary_patch if context.get("unnecessary_patch") else 0.0,
            "unnecessary_reset": c.blue_unnecessary_reset if context.get("unnecessary_reset") else 0.0,
            "unnecessary_block": c.blue_unnecessary_block if context.get("unnecessary_block") else 0.0,
            "unnecessary_restore": c.blue_unnecessary_restore if context.get("unnecessary_restore") else 0.0,
            "false_positive": c.blue_false_positive if context.get("false_positive") else 0.0,
            "service_downtime": c.blue_service_downtime if context.get("service_downtime") else 0.0,
            "critical_asset_compromised": c.blue_critical_asset_compromised if context.get("critical_asset_compromised") else 0.0,
            "step_penalty": c.step_penalty,
        }


# ---------------------------------------------------------------------------
# Reward configuration variants (for ablation studies)
# ---------------------------------------------------------------------------
def reward_config_sparse() -> RewardConfig:
    """Sparse: only terminal / major milestones rewarded."""
    c = RewardConfig()
    c.red_recon_success = 0.0
    c.red_exploit_success = 0.0
    c.red_lateral_success = 0.0
    c.red_priv_esc_success = 0.0
    c.step_penalty = 0.0
    return c


def reward_config_dense() -> RewardConfig:
    """Dense: reward every successful action."""
    c = RewardConfig()
    # keep defaults
    return c


def reward_config_availability_penalty() -> RewardConfig:
    """Security + availability penalty: defender punished more for unnecessary downtime."""
    c = RewardConfig()
    c.blue_unnecessary_isolation = -12.0
    c.blue_unnecessary_patch = -8.0
    c.blue_service_downtime = -15.0
    return c
