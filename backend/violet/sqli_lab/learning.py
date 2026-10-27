"""Separate factorized Red and categorical Blue PPO policies for SQLi episodes."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from violet.sqli_lab.contracts import (
    BLUE_ACTIONS,
    RED_COMPONENTS,
    BaseTerm,
    Connector,
    FieldName,
    Predicate,
    SqliRequestAction,
)

Role = Literal["red", "blue"]
RED_FEATURE_NAMES = (
    "step_fraction",
    "remaining_fraction",
    "last_status_fraction",
    "last_rows_fraction",
    "last_unauthorized_fraction",
    "last_query_error",
    "recent_attack_signal",
    "last_action_was_injection",
)
BLUE_FEATURE_NAMES = (
    "step_fraction",
    "remaining_fraction",
    "guest_requests_fraction",
    "http_failures_fraction",
    "injection_signals_fraction",
    "query_endpoint_health",
    "parameterized_control_active",
    "rate_limit_control_active",
)
FEATURES: dict[Role, tuple[str, ...]] = {"red": RED_FEATURE_NAMES, "blue": BLUE_FEATURE_NAMES}
HEAD_SIZES: dict[Role, tuple[int, ...]] = {
    "red": tuple(len(values) for values in RED_COMPONENTS.values()),
    "blue": (len(BLUE_ACTIONS),),
}

# Keep the values used by the optimizer beside their implementation so every
# run can record the exact PPO recipe behind its update and checkpoint.
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_VALUE_LOSS_COEFFICIENT = 0.5
PPO_ENTROPY_COEFFICIENT = 0.01
PPO_MAX_GRAD_NORM = 0.5
PPO_OPTIMIZATION_PASSES = 4


def ppo_settings(
    learning_rate: float, *, optimization_passes: int = PPO_OPTIMIZATION_PASSES
) -> dict[str, Any]:
    """Serializable provenance for this PPO implementation."""
    return {
        "algorithm": "PPO_clipped_surrogate_GAE",
        "actor_critic": "factorized_gru_actor_critic",
        "gamma": PPO_GAMMA,
        "gae_lambda": PPO_GAE_LAMBDA,
        "clip_range": PPO_CLIP_RANGE,
        "policy_loss_coefficient": 1.0,
        "value_loss_coefficient": PPO_VALUE_LOSS_COEFFICIENT,
        "entropy_coefficient": PPO_ENTROPY_COEFFICIENT,
        "optimizer": "Adam",
        "learning_rate": float(learning_rate),
        "gradient_clip_max_norm": PPO_MAX_GRAD_NORM,
        "optimization_passes": int(optimization_passes),
        "advantage_normalization": "mean_std_when_batch_has_multiple_transitions",
    }


@dataclass(frozen=True)
class SqliSample:
    action_indices: tuple[int, ...]
    component_probabilities: tuple[tuple[float, ...], ...]
    selected_probabilities: tuple[float, ...]
    old_log_probability: float
    value_estimate: float
    entropy: float
    policy_version: int
    inference_ms: float
    recurrent_state: tuple[float, ...] = ()
    recurrent_state_after: tuple[float, ...] = ()


class FactorizedActorCritic(nn.Module):
    """Role-local GRU actor-critic with a categorical head per action component."""

    def __init__(self, observation_size: int, head_sizes: tuple[int, ...], width: int = 96):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_size, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
        )
        self.memory = nn.GRUCell(width, width)
        self.actor_heads = nn.ModuleList(nn.Linear(width, size) for size in head_sizes)
        self.critic = nn.Linear(width, 1)

    def forward(
        self, observation: torch.Tensor, recurrent_state: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        encoded = self.trunk(observation)
        hidden = self.memory(encoded, recurrent_state)
        return [head(hidden) for head in self.actor_heads], self.critic(hidden).squeeze(-1), hidden


class SqliPPOPolicy:
    """A role-local actor, critic, optimizer, RNG, rollout, and policy version."""

    def __init__(self, role: Role, *, seed: int, learning_rate: float = 3e-4):
        self.role = role
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.device = torch.device("cpu")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = FactorizedActorCritic(len(FEATURES[role]), HEAD_SIZES[role]).to(
                self.device
            )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.policy_version = 0
        self.optimizer_step_count = 0
        self.hidden_size = 96
        self.recurrent_state = np.zeros(self.hidden_size, dtype=np.float32)
        self.last_update_diagnostics: list[dict[str, Any]] = []

    def reset_hidden_state(self) -> None:
        self.recurrent_state = np.zeros(self.hidden_size, dtype=np.float32)

    def _distributions(
        self, observation: np.ndarray, recurrent_state: np.ndarray | None = None
    ) -> tuple[list[Categorical], torch.Tensor, torch.Tensor]:
        encoded = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = self.recurrent_state if recurrent_state is None else recurrent_state
        hidden = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        logits, value, next_state = self.model(encoded, hidden)
        return [Categorical(logits=head) for head in logits], value.squeeze(0), next_state

    def probabilities(
        self, observation: np.ndarray, recurrent_state: np.ndarray | None = None
    ) -> tuple[tuple[float, ...], ...]:
        with torch.no_grad():
            distributions, _value, _next_state = self._distributions(observation, recurrent_state)
        return tuple(
            tuple(float(value) for value in distribution.probs[0].cpu().tolist())
            for distribution in distributions
        )

    def value(self, observation: np.ndarray, recurrent_state: np.ndarray | None = None) -> float:
        with torch.no_grad():
            _distributions, value, _next_state = self._distributions(observation, recurrent_state)
        return float(value.item())

    def act(
        self, observation: np.ndarray, *, stochastic: bool
    ) -> tuple[dict[str, Any], SqliSample]:
        started = perf_counter()
        state_before = self.recurrent_state.copy()
        with torch.no_grad():
            distributions, value, next_state = self._distributions(observation, state_before)
            indices: list[int] = []
            selected_probs: list[float] = []
            log_prob = 0.0
            entropy = 0.0
            all_probs: list[tuple[float, ...]] = []
            for distribution in distributions:
                probabilities = distribution.probs[0].cpu().numpy().astype(np.float64)
                probabilities /= probabilities.sum()
                index = (
                    int(self.rng.choice(len(probabilities), p=probabilities))
                    if stochastic
                    else int(np.argmax(probabilities))
                )
                probability = float(probabilities[index])
                indices.append(index)
                selected_probs.append(probability)
                all_probs.append(tuple(float(item) for item in probabilities))
                log_prob += float(
                    distribution.log_prob(  # type: ignore[no-untyped-call]
                        torch.tensor([index], device=self.device)
                    ).item()
                )
                entropy += float(distribution.entropy().item())  # type: ignore[no-untyped-call]
        sample = SqliSample(
            action_indices=tuple(indices),
            component_probabilities=tuple(all_probs),
            selected_probabilities=tuple(selected_probs),
            old_log_probability=log_prob,
            value_estimate=float(value.item()),
            entropy=entropy,
                policy_version=self.policy_version,
                inference_ms=(perf_counter() - started) * 1000,
                recurrent_state=tuple(float(value) for value in state_before),
                recurrent_state_after=tuple(float(value) for value in next_state[0].cpu().tolist()),
            )
        self.recurrent_state = np.asarray(sample.recurrent_state_after, dtype=np.float32)
        if self.role == "red":
            component_names = tuple(RED_COMPONENTS)
            component_values = [
                RED_COMPONENTS[name][index]
                for name, index in zip(component_names, indices, strict=True)
            ]
            action = SqliRequestAction(
                field=cast("FieldName", component_values[0]),
                base_term=cast("BaseTerm", component_values[1]),
                close_quote=component_values[2] == "yes",
                connector=cast("Connector", component_values[3]),
                predicate=cast("Predicate", component_values[4]),
                comment=component_values[5] == "yes",
            )
            return action.model_dump(mode="json"), sample
        return {"operation": BLUE_ACTIONS[indices[0]]}, sample

    def update(self, transitions: list[dict[str, Any]], passes: int = 4) -> dict[str, float]:
        if not transitions:
            self.last_update_diagnostics = []
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "post_update_approx_kl": 0.0,
                "post_update_clip_fraction": 0.0,
                "advantage_mean": 0.0,
                "advantage_std": 0.0,
                "sample_count": 0.0,
                "optimizer_steps": 0.0,
                "policy_version_before": float(self.policy_version),
                "policy_version_after": float(self.policy_version),
            }
        if passes < 1:
            raise ValueError("PPO requires at least one optimization pass")
        advantages = np.zeros(len(transitions), dtype=np.float32)
        next_advantage = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            item = transitions[index]
            terminal = bool(item.get("terminated", False))
            boundary = bool(item.get("truncated", False))
            continuation = 0.0 if terminal or boundary else 1.0
            delta = (
                float(item["reward"])
                + PPO_GAMMA * float(item["next_value"]) * (0.0 if terminal else 1.0)
                - float(item["value"])
            )
            next_advantage = delta + PPO_GAMMA * PPO_GAE_LAMBDA * continuation * next_advantage
            advantages[index] = next_advantage
        returns = advantages + np.asarray(
            [float(item["value"]) for item in transitions], dtype=np.float32
        )
        normalized = advantages.copy()
        if len(normalized) > 1:
            normalized = (normalized - normalized.mean()) / (normalized.std() + 1e-8)
        observations = torch.as_tensor(
            np.stack([item["observation"] for item in transitions]), dtype=torch.float32
        )
        recurrent_states = torch.as_tensor(
            np.stack(
                [
                    item.get("recurrent_state")
                    or tuple(0.0 for _ in range(self.hidden_size))
                    for item in transitions
                ]
            ),
            dtype=torch.float32,
        )
        action_indices = torch.as_tensor(
            np.asarray([item["action_indices"] for item in transitions]), dtype=torch.int64
        )
        old_log_probs = torch.as_tensor(
            [float(item["old_log_probability"]) for item in transitions], dtype=torch.float32
        )
        advantages_tensor = torch.as_tensor(normalized, dtype=torch.float32)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
        before = self.policy_version
        gradient_norm_value = 0.0
        for _ in range(passes):
            logits, values, _next_states = self.model(observations, recurrent_states)
            distributions = [Categorical(logits=head) for head in logits]
            new_log_probs = torch.stack(
                [
                    distribution.log_prob(action_indices[:, column])  # type: ignore[no-untyped-call]
                    for column, distribution in enumerate(distributions)
                ],
                dim=0,
            ).sum(dim=0)
            entropy = (
                torch.stack(
                    [distribution.entropy() for distribution in distributions], dim=0  # type: ignore[no-untyped-call]
                ).sum(dim=0).mean()
            )
            ratios = torch.exp(new_log_probs - old_log_probs)
            clipped = torch.clamp(ratios, 1.0 - PPO_CLIP_RANGE, 1.0 + PPO_CLIP_RANGE)
            policy_loss = -torch.minimum(ratios * advantages_tensor, clipped * advantages_tensor).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns_tensor)
            loss = (
                policy_loss
                + PPO_VALUE_LOSS_COEFFICIENT * value_loss
                - PPO_ENTROPY_COEFFICIENT * entropy
            )
            self.optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), PPO_MAX_GRAD_NORM
            )
            gradient_norm_value = float(gradient_norm.item())
            self.optimizer.step()
            self.optimizer_step_count += 1
        self.policy_version += 1

        # Re-evaluate the final policy after all optimizer passes. These KL and
        # clipping values describe the post-update policy against the sampled
        # behavior policy, rather than the parameters before the final step.
        with torch.no_grad():
            logits, values, _next_states = self.model(observations, recurrent_states)
            distributions = [Categorical(logits=head) for head in logits]
            new_log_probs = torch.stack(
                [
                    distribution.log_prob(action_indices[:, column])  # type: ignore[no-untyped-call]
                    for column, distribution in enumerate(distributions)
                ],
                dim=0,
            ).sum(dim=0)
            entropy = torch.stack(
                [distribution.entropy() for distribution in distributions], dim=0  # type: ignore[no-untyped-call]
            ).sum(dim=0).mean()
            ratios = torch.exp(new_log_probs - old_log_probs)
            clipped = torch.clamp(ratios, 1.0 - PPO_CLIP_RANGE, 1.0 + PPO_CLIP_RANGE)
            policy_loss = -torch.minimum(
                ratios * advantages_tensor, clipped * advantages_tensor
            ).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns_tensor)
            loss = (
                policy_loss
                + PPO_VALUE_LOSS_COEFFICIENT * value_loss
                - PPO_ENTROPY_COEFFICIENT * entropy
            )
            log_ratio = new_log_probs - old_log_probs
            post_update_kl = float(((ratios - 1.0) - log_ratio).mean().item())
            post_update_clip_fraction = float(
                ((ratios - 1.0).abs() > PPO_CLIP_RANGE).float().mean().item()
            )
            self.last_update_diagnostics = []
            for index, transition in enumerate(transitions):
                new_component_probabilities = [
                    tuple(float(value) for value in distribution.probs[index].cpu().tolist())
                    for distribution in distributions
                ]
                selected_indices = [int(value) for value in transition["action_indices"]]
                selected_new = [
                    probabilities[action_index]
                    for probabilities, action_index in zip(
                        new_component_probabilities, selected_indices, strict=True
                    )
                ]
                new_log_probability = float(new_log_probs[index].item())
                self.last_update_diagnostics.append(
                    {
                        "decision_id": transition.get("decision_id"),
                        "step_id": transition.get("step_id"),
                        "policy_version": transition.get("policy_version"),
                        "action_indices": selected_indices,
                        "old_component_probabilities": transition.get("component_probabilities", []),
                        "new_component_probabilities": [list(values) for values in new_component_probabilities],
                        "old_selected_component_probabilities": transition.get("selected_probabilities", []),
                        "new_selected_component_probabilities": selected_new,
                        "old_joint_log_probability": float(transition["old_log_probability"]),
                        "new_joint_log_probability": new_log_probability,
                        "old_joint_probability": float(np.exp(float(transition["old_log_probability"]))),
                        "new_joint_probability": float(np.exp(new_log_probability)),
                        "critic_value": float(transition["value"]),
                        "next_value": float(transition["next_value"]),
                        "reward": float(transition["reward"]),
                        "advantage": float(advantages[index]),
                        "normalized_advantage": float(normalized[index]),
                        "return_target": float(returns[index]),
                        "terminated": bool(transition.get("terminated", False)),
                        "truncated": bool(transition.get("truncated", False)),
                    }
                )
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": post_update_kl,
            "clip_fraction": post_update_clip_fraction,
            "post_update_approx_kl": post_update_kl,
            "post_update_clip_fraction": post_update_clip_fraction,
            "gradient_norm": gradient_norm_value,
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
            "sample_count": float(len(transitions)),
            "optimizer_steps": float(passes),
            "policy_version_before": float(before),
            "policy_version_after": float(self.policy_version),
        }

    def load_model_state(self, state: dict[str, Any], checkpoint_format: str) -> None:
        """Load current GRU checkpoints and adapt pre-GRU state dicts explicitly."""
        if checkpoint_format == "sqli.joint_ppo.v4":
            self.model.load_state_dict(state, strict=True)
            return
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        expected_memory = {
            "memory.weight_ih",
            "memory.weight_hh",
            "memory.bias_ih",
            "memory.bias_hh",
        }
        if set(missing) != expected_memory or unexpected:
            raise ValueError("legacy SQLi policy weights cannot be adapted to the recurrent model")

    def load_optimizer_state(self, state: dict[str, Any], checkpoint_format: str) -> None:
        if checkpoint_format == "sqli.joint_ppo.v4":
            self.optimizer.load_state_dict(state)
            return

        old_group = state.get("param_groups", [{}])[0]
        old_ids = list(old_group.get("params", []))
        current = self.optimizer.state_dict()
        current_ids = list(current["param_groups"][0]["params"])
        current_names = [name for name, _parameter in self.model.named_parameters()]
        legacy_names = [name for name in current_names if not name.startswith("memory.")]
        if len(old_ids) != len(legacy_names):
            raise ValueError("legacy SQLi optimizer parameter mapping is inconsistent")
        old_by_name = dict(zip(legacy_names, old_ids, strict=True))
        new_by_name = dict(zip(current_names, current_ids, strict=True))
        adapted_state = {
            new_by_name[name]: state.get("state", {})[old_by_name[name]]
            for name in legacy_names
            if old_by_name[name] in state.get("state", {})
        }
        adapted_group = dict(old_group)
        adapted_group["params"] = current_ids
        self.optimizer.load_state_dict({"state": adapted_state, "param_groups": [adapted_group]})
