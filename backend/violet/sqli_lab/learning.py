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


class FactorizedActorCritic(nn.Module):
    """Shared state representation with one categorical head per action component."""

    def __init__(self, observation_size: int, head_sizes: tuple[int, ...], width: int = 96):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_size, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
        )
        self.actor_heads = nn.ModuleList(nn.Linear(width, size) for size in head_sizes)
        self.critic = nn.Linear(width, 1)

    def forward(self, observation: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        hidden = self.trunk(observation)
        return [head(hidden) for head in self.actor_heads], self.critic(hidden).squeeze(-1)


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

    def _distributions(self, observation: np.ndarray) -> tuple[list[Categorical], torch.Tensor]:
        encoded = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(encoded)
        return [Categorical(logits=head) for head in logits], value.squeeze(0)

    def probabilities(self, observation: np.ndarray) -> tuple[tuple[float, ...], ...]:
        with torch.no_grad():
            distributions, _value = self._distributions(observation)
        return tuple(
            tuple(float(value) for value in distribution.probs[0].cpu().tolist())
            for distribution in distributions
        )

    def value(self, observation: np.ndarray) -> float:
        with torch.no_grad():
            _distributions, value = self._distributions(observation)
        return float(value.item())

    def act(
        self, observation: np.ndarray, *, stochastic: bool
    ) -> tuple[dict[str, Any], SqliSample]:
        started = perf_counter()
        with torch.no_grad():
            distributions, value = self._distributions(observation)
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
        )
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

    def update(self, transitions: list[dict[str, Any]]) -> dict[str, float]:
        if not transitions:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "advantage_mean": 0.0,
                "advantage_std": 0.0,
                "sample_count": 0.0,
                "optimizer_steps": 0.0,
                "policy_version_before": float(self.policy_version),
                "policy_version_after": float(self.policy_version),
            }
        advantages = np.zeros(len(transitions), dtype=np.float32)
        next_advantage = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            item = transitions[index]
            terminal = bool(item.get("terminated", False))
            boundary = bool(item.get("truncated", False))
            continuation = 0.0 if terminal or boundary else 1.0
            delta = (
                float(item["reward"])
                + 0.99 * float(item["next_value"]) * (0.0 if terminal else 1.0)
                - float(item["value"])
            )
            next_advantage = delta + 0.99 * 0.95 * continuation * next_advantage
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
        action_indices = torch.as_tensor(
            np.asarray([item["action_indices"] for item in transitions]), dtype=torch.int64
        )
        old_log_probs = torch.as_tensor(
            [float(item["old_log_probability"]) for item in transitions], dtype=torch.float32
        )
        advantages_tensor = torch.as_tensor(normalized, dtype=torch.float32)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
        logits, values = self.model(observations)
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
                [
                    distribution.entropy()  # type: ignore[no-untyped-call]
                    for distribution in distributions
                ],
                dim=0,
            )
            .sum(dim=0)
            .mean()
        )
        ratios = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratios, 0.8, 1.2)
        policy_loss = -torch.minimum(ratios * advantages_tensor, clipped * advantages_tensor).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns_tensor)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        before = self.policy_version
        self.policy_version += 1
        self.optimizer_step_count += 1
        log_ratio = new_log_probs.detach() - old_log_probs
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(((ratios.detach() - 1.0) - log_ratio).mean().item()),
            "clip_fraction": float(((ratios.detach() - 1.0).abs() > 0.2).float().mean().item()),
            "gradient_norm": float(gradient_norm.item()),
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
            "sample_count": float(len(transitions)),
            "optimizer_steps": 1.0,
            "policy_version_before": float(before),
            "policy_version_after": float(self.policy_version),
        }
