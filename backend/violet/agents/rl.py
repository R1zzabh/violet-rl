# violet/agents/rl.py — RL agent wrapper + checkpoint stubs

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from violet.agents.base import Agent


@dataclass
class RLCheckpoint:
    """Metadata stored alongside a model checkpoint so we can reproduce a run later."""

    run_id: str
    algorithm: str
    agent_role: str  # "red" or "blue"
    seed: int
    environment_version: str = "violet_v1"
    reward_config_version: str = "v1"
    observation_version: str = "v1"
    action_space_version: str = "v1"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    training_steps: int = 0
    timestamp: str = ""
    git_commit: str = ""
    checkpoint_path: str = ""
    metrics_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "environment_version": self.environment_version,
            "agent_role": self.agent_role,
            "seed": self.seed,
            "reward_config_version": self.reward_config_version,
            "observation_version": self.observation_version,
            "action_space_version": self.action_space_version,
            "hyperparameters": self.hyperparameters,
            "training_steps": self.training_steps,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "checkpoint_path": self.checkpoint_path,
            "metrics_summary": self.metrics_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLCheckpoint:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RLAgent(Agent):
    """RL agent wrapper — provides the stable Agent interface for learned policies.

    Phase 5: the policy is a `violet.algorithms.ppo.ActorCritic` model. The wrapper
    keeps the environment and experiment runner independent of the RL implementation
    (an RLlib policy object can later be substituted behind the same interface).
    """

    agent_name: str
    role: str
    checkpoint: RLCheckpoint | None = None
    _policy: Any | None = None  # torch.nn.Module set at construction/load
    deterministic: bool = True  # greedy evaluation; set False for stochastic play

    def __init__(self, agent_name: str, role: str = "blue", policy: Any | None = None):
        self.agent_name = agent_name
        self.role = role
        if policy is not None:
            self._set_policy(policy)

    def _set_policy(self, policy: Any) -> None:
        policy.eval()
        self._policy = policy

    def select_action(self, env: Any, info: dict[str, Any]) -> int:
        """Return a masked action index from the loaded policy.

        Falls back to random action selection until a policy is loaded (useful
        for smoke tests, and a deliberate degradation — never a silent crash).
        """
        if self._policy is None:
            action_space = env.action_space(self.agent_name)
            return int(env.rng.randint(0, action_space.n - 1)) if env.rng is not None else 0
        import numpy as np

        obs = np.asarray(env.observations[self.agent_name], dtype=np.float32)
        mask = np.asarray(env.action_mask(self.agent_name), dtype=bool)
        return int(self._policy.act(obs, mask, deterministic=self.deterministic))

    def load_checkpoint(self, checkpoint_dir: str, env: Any = None) -> bool:
        """Load a checkpoint and its metadata.

        Returns True when a checkpoint was found and loaded. When `env` is given,
        a PPO ActorCritic policy is built (dims inferred from the env spaces) and
        the model weights are loaded into it.
        """
        meta_path = Path(checkpoint_dir) / "checkpoint_meta.json"
        if not meta_path.exists():
            return False

        with open(meta_path) as f:
            data = json.load(f)
        self.checkpoint = RLCheckpoint.from_dict(data)

        model_path = Path(checkpoint_dir) / "model.pt"
        if model_path.exists() and env is not None:
            import torch

            from violet.algorithms.ppo import ActorCritic

            obs_space = env.observation_space(self.agent_name)
            act_space = env.action_space(self.agent_name)
            obs_dim = int(np.prod(obs_space.shape))
            hidden = int(self.checkpoint.hyperparameters.get("hidden", 64))
            model = ActorCritic(obs_dim, act_space.n, hidden=hidden)
            # Loading VIOLET's own training artifacts (local, not user-supplied).
            # weights_only=True cannot restore the optimizer state (pickled
            # objects), so full unpickling is required here.
            payload = torch.load(model_path, map_location="cpu", weights_only=False)  # nosec B614
            model.load_state_dict(payload["model_state_dict"])
            self._set_policy(model)
        return True

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        metadata: RLCheckpoint,
        model_bytes: bytes | None = None,
    ) -> str:
        """Save a checkpoint + metadata. Returns the checkpoint directory."""
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Write metadata
        meta_path = path / "checkpoint_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Write model bytes if provided (algorithm-specific in real implementation)
        if model_bytes is not None:
            model_path = path / "model.pt"
            with open(model_path, "wb") as f:
                f.write(model_bytes)

        self.checkpoint = metadata
        return str(path)


@dataclass
class TrainingRunMetadata:
    """Metadata for a training run, persisted to runs/<run_id>/config.yaml + metadata.json."""

    run_id: str
    algorithm: str
    red_policy: str  # "scripted" | "trainable"
    blue_policy: str  # "scripted" | "trainable"
    environment_version: str
    seed: int
    reward_config_version: str
    network_config: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    max_cycles: int = 100
    created_at: str = ""
    git_commit: str = ""
    status: str = "running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "red_policy": self.red_policy,
            "blue_policy": self.blue_policy,
            "environment_version": self.environment_version,
            "seed": self.seed,
            "reward_config_version": self.reward_config_version,
            "network_config": self.network_config,
            "hyperparameters": self.hyperparameters,
            "max_cycles": self.max_cycles,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "status": self.status,
        }
