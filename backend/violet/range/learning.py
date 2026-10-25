"""Independent PPO policies and fixed-shape observations for the local HTTP range.

These policies choose only catalogued actions. They cannot provide a URL,
network destination, request body, or shell command to the executor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from violet.algorithms.ppo import ActorCritic, compute_gae

Role = Literal["red", "blue"]
OBSERVATION_SCHEMA = "violet.range.observation.v2"

# Red action IDs map in the runner to the existing strict RangeAction contracts.
RED_ACTIONS = (
    "discover_home",
    "discover_login",
    "discover_document",
    "login_synthetic_alice",
    "login_synthetic_bob",
    "read_own_document",
    "read_other_document",
    "check_health",
)
# ID zero is a no-op. The remaining IDs map to existing ControlRequest actions.
BLUE_ACTIONS = ("no_op", "harden_document_ownership", "rate_limit_alice", "block_alice")
OBSERVATION_FEATURES: dict[Role, tuple[str, ...]] = {
    "red": (
        "step_fraction",
        "remaining_budget_fraction",
        "own_auth_alice",
        "own_auth_bob",
        "last_request_success",
        "last_http_status_normalized",
        "own_verified_objective_fraction",
    ),
    "blue": (
        "step_fraction",
        "remaining_budget_fraction",
        "observed_request_count_fraction",
        "observed_failed_request_fraction",
        "public_service_health",
        "verified_foreign_read_finding",
        "verified_foreign_read_prevention",
        "applied_control_fraction",
    ),
}


def clipped_surrogate_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return PPO's clipped policy loss, probability ratios, and clip fraction."""
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("PPO clip epsilon must be in (0,1)")
    log_ratio = new_log_probabilities - old_log_probabilities
    if not torch.isfinite(log_ratio).all() or not torch.isfinite(advantages).all():
        raise FloatingPointError("PPO log probabilities and advantages must be finite")
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("PPO probability ratio overflowed")
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
    return loss, ratio, clip_fraction


def _unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class RangeObservation:
    """Validated role-scoped observation; values are normalized to [0, 1]."""

    role: Role
    features: tuple[float, ...]
    schema_id: str = OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        expected = OBSERVATION_FEATURES[self.role]
        if len(self.features) != len(expected) or any(
            not np.isfinite(x) or not 0.0 <= x <= 1.0 for x in self.features
        ):
            raise ValueError(
                f"{self.role} observation must contain {len(expected)} values in [0,1]"
            )

    def array(self) -> np.ndarray:
        return np.asarray(self.features, dtype=np.float32)

    @classmethod
    def red(
        cls,
        *,
        step: int,
        budget: int,
        authenticated_alice: bool,
        authenticated_bob: bool,
        last_success: bool,
        last_status: int | None,
        verified_objectives: int,
    ) -> RangeObservation:
        return cls(
            "red",
            (
                _unit(step / max(1, budget)),
                _unit((budget - step) / max(1, budget)),
                float(authenticated_alice),
                float(authenticated_bob),
                float(last_success),
                _unit((last_status or 0) / 599),
                _unit(verified_objectives),
            ),
        )

    @classmethod
    def blue(
        cls,
        *,
        step: int,
        budget: int,
        requests: int,
        failed_requests: int,
        service_healthy: bool,
        verified_attack: bool,
        verified_prevention: bool,
        controls_applied: int,
    ) -> RangeObservation:
        return cls(
            "blue",
            (
                _unit(step / max(1, budget)),
                _unit((budget - step) / max(1, budget)),
                _unit(requests / max(1, budget)),
                _unit(failed_requests / max(1, requests)),
                float(service_healthy),
                float(verified_attack),
                float(verified_prevention),
                _unit(controls_applied / max(1, step)),
            ),
        )


@dataclass(frozen=True)
class PolicyDecision:
    action: int
    log_probability: float
    value: float
    policy_version: int
    action_probability: float = 1.0
    entropy: float = 0.0


@dataclass(frozen=True)
class RangeTransition:
    observation: np.ndarray
    action_mask: np.ndarray
    decision: PolicyDecision
    reward: float
    next_value: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class RangePPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 32
    hidden_size: int = 64
    seed: int = 42


class RangePPOAgent:
    """One role's policy, critic, rollout and optimizer; Red and Blue never share state."""

    def __init__(self, role: Role, config: RangePPOConfig | None = None):
        self.role = role
        self.config = config or RangePPOConfig()
        self.actions = RED_ACTIONS if role == "red" else BLUE_ACTIONS
        self.device = torch.device("cpu")
        role_seed = self.config.seed + (0 if role == "red" else 1)
        self._torch_rng = torch.Generator(device=self.device).manual_seed(role_seed)
        self._numpy_rng = np.random.default_rng(role_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(role_seed)
            self.model = ActorCritic(
                len(OBSERVATION_FEATURES[role]), len(self.actions), self.config.hidden_size
            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.transitions: list[RangeTransition] = []
        self.policy_version = 0
        self.optimizer_step_count = 0

    def act(
        self,
        observation: RangeObservation,
        *,
        action_mask: np.ndarray | None = None,
        training: bool,
    ) -> PolicyDecision:
        if observation.role != self.role or observation.schema_id != OBSERVATION_SCHEMA:
            raise ValueError("observation role or schema does not match this policy")
        mask = (
            np.ones(len(self.actions), dtype=bool)
            if action_mask is None
            else np.asarray(action_mask, dtype=bool)
        )
        if mask.shape != (len(self.actions),) or not mask.any():
            raise ValueError("action mask must permit at least one catalogued action")
        with torch.no_grad():
            obs = torch.as_tensor(observation.array(), device=self.device).unsqueeze(0)
            logits, value = self.model(obs)
            logits = logits.masked_fill(
                ~torch.as_tensor(mask, device=self.device).unsqueeze(0), -1e9
            )
            dist = Categorical(logits=logits)
            if training:
                action = torch.multinomial(dist.probs, 1, generator=self._torch_rng).squeeze(-1)
            else:
                action = torch.argmax(logits, dim=-1)
            log_prob = dist.log_prob(action)  # type: ignore[no-untyped-call]
        return PolicyDecision(
            action=int(action.item()),
            log_probability=float(log_prob.item()),
            value=float(value.item()),
            policy_version=self.policy_version,
            action_probability=float(dist.probs[0, action].item()),
            entropy=float(dist.entropy().item()),  # type: ignore[no-untyped-call]
        )

    def value(self, observation: RangeObservation) -> float:
        """Read the critic without sampling or mutating policy state."""
        if observation.role != self.role:
            raise ValueError("observation role does not match this policy")
        with torch.no_grad():
            _, value = self.model(
                torch.as_tensor(observation.array(), device=self.device).unsqueeze(0)
            )
        return float(value.item())

    def record(self, transition: RangeTransition) -> None:
        if transition.observation.shape != (len(OBSERVATION_FEATURES[self.role]),):
            raise ValueError("transition observation has the wrong role-specific shape")
        if not np.isfinite(transition.observation).all() or np.any(
            (transition.observation < 0.0) | (transition.observation > 1.0)
        ):
            raise ValueError("transition observation must be normalized to [0,1]")
        if transition.action_mask.shape != (len(self.actions),):
            raise ValueError("transition action mask has the wrong shape")
        if not 0 <= transition.decision.action < len(self.actions):
            raise ValueError("transition action is outside this role's catalog")
        if not transition.action_mask[transition.decision.action]:
            raise ValueError("cannot learn from a rejected or masked action")
        if not all(
            np.isfinite(value)
            for value in (
                transition.reward,
                transition.next_value,
                transition.decision.value,
                transition.decision.log_probability,
            )
        ):
            raise ValueError("transition values and log probability must be finite")
        if transition.terminated and transition.truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        self.transitions.append(transition)

    def break_trace(self) -> None:
        """Stop GAE from crossing an action that was not executed."""
        if self.transitions:
            previous = self.transitions[-1]
            if not previous.terminated and not previous.truncated:
                self.transitions[-1] = replace(previous, truncated=True)

    def update(self) -> dict[str, float | int]:
        """Run clipped PPO on accepted, actually executed transitions only."""
        if not self.transitions:
            return {
                "updates": 0,
                "transitions": 0,
                "optimizer_steps": 0,
                "optimizer_steps_total": self.optimizer_step_count,
            }
        transitions = self.transitions
        rewards = [t.reward for t in transitions]
        values = [t.decision.value for t in transitions]
        terminated = [t.terminated for t in transitions]
        truncated = [t.truncated for t in transitions]
        advantages, returns = compute_gae(
            rewards,
            values,
            terminated,
            transitions[-1].next_value,
            self.config.gamma,
            self.config.gae_lambda,
            truncations=truncated,
            next_values=[t.next_value for t in transitions],
        )
        raw_advantages = advantages.copy()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        obs = torch.as_tensor(np.stack([t.observation for t in transitions]), device=self.device)
        masks = torch.as_tensor(
            np.stack([t.action_mask for t in transitions]), dtype=torch.bool, device=self.device
        )
        actions = torch.as_tensor(
            [t.decision.action for t in transitions], dtype=torch.long, device=self.device
        )
        old_logp = torch.as_tensor(
            [t.decision.log_probability for t in transitions], device=self.device
        )
        adv = torch.as_tensor(advantages.copy(), device=self.device)
        ret = torch.as_tensor(returns.copy(), device=self.device)
        stats: dict[str, list[float]] = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "clip_fraction": [],
            "gradient_norm": [],
        }
        optimizer_steps_this_update = 0
        n = len(transitions)
        for _ in range(self.config.epochs):
            indices = self._numpy_rng.permutation(n)
            for start in range(0, n, self.config.minibatch_size):
                mb = torch.as_tensor(
                    indices[start : start + self.config.minibatch_size], device=self.device
                )
                logits, predicted_values = self.model(obs[mb])
                dist = Categorical(logits=logits.masked_fill(~masks[mb], -1e9))
                new_logp = dist.log_prob(actions[mb])  # type: ignore[no-untyped-call]
                policy_loss, _ratio, clip_fraction = clipped_surrogate_loss(
                    new_logp, old_logp[mb], adv[mb], self.config.clip_epsilon
                )
                value_loss = nn.functional.mse_loss(predicted_values, ret[mb])
                entropy = dist.entropy().mean()  # type: ignore[no-untyped-call]
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO loss; rollout was not updated")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
                if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("non-finite PPO gradient; rollout was not updated")
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_gradient_norm
                )
                if not torch.isfinite(grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("PPO gradient norm is non-finite")
                self.optimizer.step()
                self.optimizer_step_count += 1
                optimizer_steps_this_update += 1
                stats["loss"].append(float(loss.detach().item()))
                stats["policy_loss"].append(float(policy_loss.detach().item()))
                stats["value_loss"].append(float(value_loss.detach().item()))
                stats["entropy"].append(float(entropy.detach().item()))
                stats["clip_fraction"].append(float(clip_fraction.detach().item()))
                stats["gradient_norm"].append(float(grad_norm.detach().item()))
        self.transitions = []
        self.policy_version += 1
        return {
            "updates": self.policy_version,
            "transitions": n,
            "optimizer_steps": optimizer_steps_this_update,
            "optimizer_steps_total": self.optimizer_step_count,
            "advantage_mean": float(np.mean(raw_advantages)),
            "advantage_std": float(np.std(raw_advantages)),
            "advantage_min": float(np.min(raw_advantages)),
            "advantage_max": float(np.max(raw_advantages)),
            **{key: float(np.mean(values)) for key, values in stats.items()},
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_id": OBSERVATION_SCHEMA,
                "role": self.role,
                "actions": self.actions,
                "config": self.config.__dict__,
                "policy_version": self.policy_version,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "torch_rng_state": self._torch_rng.get_state(),
                "numpy_rng_state": self._numpy_rng.bit_generator.state,
            },
            destination,
        )

    def load(self, path: str | Path, *, inference_only: bool = False) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)  # nosec B614
        if (
            payload.get("schema_id") != OBSERVATION_SCHEMA
            or payload.get("role") != self.role
            or tuple(payload.get("actions", ())) != self.actions
        ):
            raise ValueError("checkpoint role, observation schema, or action catalog mismatch")
        self.model.load_state_dict(payload["model"])
        if not inference_only:
            self.optimizer.load_state_dict(payload["optimizer"])
            if "torch_rng_state" in payload:
                self._torch_rng.set_state(payload["torch_rng_state"])
            if "numpy_rng_state" in payload:
                self._numpy_rng.bit_generator.state = payload["numpy_rng_state"]
        self.policy_version = int(payload["policy_version"])
