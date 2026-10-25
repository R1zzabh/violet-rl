# violet/algorithms/ppo.py — self-contained PPO for the VIOLET environment
#
# Phase 5 implementation. Design decisions (documented, see docs/development_log.md):
#   * PyTorch PPO instead of RLlib: the V1 env is tiny (48 red actions, 64-dim obs),
#     so a single-file PPO is simpler, debuggable, and keeps Ray out of the dev
#     dependency tree. The RLAgent wrapper keeps an RLlib swap open.
#   * Phase 5 trains ONE agent (red) against a FROZEN opponent. Adversarial
#     self-play is Phase 6+ and must not leak into this module.
#   * Invalid actions are MASKED (hard constraints from env.action_mask) and
#     every sampled action is legal by construction.
#   * Everything is seeded: torch/np/random. CPU-only is fully supported and
#     is the default dev path; CUDA is used only if available.

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from violet.agents.rl import RLCheckpoint
from violet.types import ATTACKER_AGENT, DEFENDER_AGENT

if TYPE_CHECKING:
    from violet.env import VIOLETEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Two-head MLP: shared trunk -> policy logits (masked) + state value.

    Kept deliberately small; V1 obs are 64-dim and the action space is < 128.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)
        # Small init so initial policy logits are near-uniform (PPO good practice).
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def masked_logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply -inf to disallowed actions so they get zero probability."""
        logits, _ = self.forward(obs)
        return logits.masked_fill(~mask, float("-inf"))

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """Select a legal action for a single observation.

        Deterministic mode (argmax) is used for evaluation; stochastic mode
        (sample) is used for rollout collection. If every action were masked
        (should never happen — the env guarantees WAIT is always legal), the
        Categorical distribution over all -inf logits is uniform-NaN; we
        deliberately raise instead of returning a garbage action.
        """
        with torch.no_grad():
            logits = self.masked_logits(
                torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0),
                torch.as_tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0),
            )
            if bool(torch.isfinite(logits).sum().item()) == 0:
                raise ValueError("action_mask disallowed every action — env mask bug")
            if deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            return int(Categorical(logits=logits).sample().item())  # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    terminated: bool  # True only when the task reaches an absorbing terminal state
    truncated: bool = False  # Time limit ends the episode but permits value bootstrap
    next_value: float = 0.0


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    next_value: float,
    gamma: float,
    gae_lambda: float,
    truncations: list[bool] | None = None,
    next_values: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over one trajectory.

    `dones` marks true terminal states. Time-limit truncations should be passed
    as False so the one-step TD target bootstraps from the final observation.
    """
    size = len(rewards)
    if len(values) != size or len(dones) != size:
        raise ValueError("rewards, values, and terminal flags must have equal length")
    if truncations is not None and len(truncations) != size:
        raise ValueError("truncation flags must match the trajectory length")
    if next_values is not None and len(next_values) != size:
        raise ValueError("per-transition bootstrap values must match the trajectory length")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and GAE lambda must be in [0,1]")
    if size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    if any(not np.isfinite(value) for value in (*rewards, *values, next_value)):
        raise ValueError("trajectory rewards and values must be finite")
    if next_values is not None and any(not np.isfinite(value) for value in next_values):
        raise ValueError("per-transition bootstrap values must be finite")
    advantages = np.zeros(size, dtype=np.float32)
    truncations = truncations or [False] * len(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        bootstrap = 0.0 if dones[t] else 1.0
        continuation = 0.0 if dones[t] or truncations[t] else 1.0
        if next_values is not None:
            next_v = next_values[t]
        else:
            next_v = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * bootstrap - values[t]
        last_gae = delta + gamma * gae_lambda * continuation * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# Rollout buffer (on-policy; one PPO epoch's worth of data)
# ---------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    def __len__(self) -> int:
        return len(self.transitions)

    def finalize(self, next_value: float, gamma: float, gae_lambda: float) -> None:
        rewards = [t.reward for t in self.transitions]
        values = [t.value for t in self.transitions]
        terminated = [t.terminated for t in self.transitions]
        truncated = [t.truncated for t in self.transitions]
        adv, ret = compute_gae(
            rewards,
            values,
            terminated,
            next_value,
            gamma,
            gae_lambda,
            truncations=truncated,
            next_values=[t.next_value for t in self.transitions],
        )
        # Normalize advantages (standard PPO trick; harmless for single traj too)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.advantages = adv
        self.returns = ret

    def clear(self) -> None:
        self.transitions.clear()
        self.advantages = None
        self.returns = None

    def tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        assert self.advantages is not None and self.returns is not None
        return {
            "obs": torch.as_tensor(
                np.stack([t.obs for t in self.transitions]), dtype=torch.float32, device=device
            ),
            "mask": torch.as_tensor(
                np.stack([t.mask for t in self.transitions]), dtype=torch.bool, device=device
            ),
            "actions": torch.as_tensor(
                [t.action for t in self.transitions], dtype=torch.long, device=device
            ),
            "old_log_probs": torch.as_tensor(
                [t.log_prob for t in self.transitions], dtype=torch.float32, device=device
            ),
            "old_values": torch.as_tensor(
                [t.value for t in self.transitions], dtype=torch.float32, device=device
            ),
            "advantages": torch.as_tensor(self.advantages, dtype=torch.float32, device=device),
            "returns": torch.as_tensor(self.returns, dtype=torch.float32, device=device),
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    # rollout
    rollout_steps: int = 512  # env steps collected per PPO update
    # optimization
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs_per_update: int = 4
    minibatch_size: int = 64
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    # net
    hidden: int = 64
    # bookkeeping
    seed: int = 42
    max_updates: int = 50
    checkpoint_interval: int = 10  # updates between periodic checkpoints
    run_dir: str = "runs/ppo_red_v1"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class PPOTrainer:
    """Trains the RED attacker with PPO against a frozen BLUE opponent.

    The blue agent object is only ever used for select_action() — its
    parameters are never updated by this trainer.
    """

    def __init__(
        self,
        env: VIOLETEnv,
        blue_agent: Any,  # frozen opponent; must expose select_action(env, info)
        config: PPOConfig,
    ):
        self.env = env
        self.blue_agent = blue_agent
        self.config = config

        self._set_seeds(config.seed)

        obs_space = env.observation_space(ATTACKER_AGENT)
        act_space = env.action_space(ATTACKER_AGENT)
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = act_space.n

        self.model = ActorCritic(self.obs_dim, self.act_dim, hidden=config.hidden).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.buffer = RolloutBuffer()
        self.global_step = 0  # env steps since the very first start (survives resume)
        self.update_idx = 0  # PPO updates completed

        # metrics history: list of dicts, one per update
        self.update_log: list[dict[str, Any]] = []

        run_dir = Path(config.run_dir)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # -- seeding ------------------------------------------------------------
    def _set_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _action_seed(self) -> int:
        """Deterministic per-step seed for stochastic rollouts."""
        return (self.config.seed * 1_000_003 + self.global_step) % (2**31)

    # -- rollout ------------------------------------------------------------
    def collect_rollout(self) -> tuple[list[dict[str, Any]], float]:
        """Run the env until rollout_steps transitions are collected.

        The env's step counter persists across reset() calls inside a rollout;
        termination restarts episodes automatically. Returns episode summaries
        and the mean episode reward observed.
        """
        episodes: list[dict[str, Any]] = []
        ep_reward = 0.0
        ep_len = 0
        step_rng_state = None

        obs_dict, infos = self.env.reset(seed=self._action_seed())
        self.blue_agent.reset()
        obs = np.asarray(obs_dict[ATTACKER_AGENT], dtype=np.float32)
        mask = np.asarray(self.env.action_mask(ATTACKER_AGENT), dtype=bool)

        while len(self.buffer) < self.config.rollout_steps:
            with torch.no_grad():
                logits, value = self.model.forward(
                    torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                )
                logits = logits.masked_fill(
                    ~torch.as_tensor(mask, device=DEVICE).unsqueeze(0), float("-inf")
                )
                dist = Categorical(logits=logits)
                action = dist.sample()  # type: ignore[no-untyped-call]
                log_prob = dist.log_prob(action)  # type: ignore[no-untyped-call]

            # Frozen opponent acts from ITS OWN view of the env (no policy gradient
            # flows through blue; its outputs are treated as environment dynamics).
            blue_action = self.blue_agent.select_action(self.env, infos.get(DEFENDER_AGENT, {}))
            step_rng_state = random.getstate()

            obs_next, rewards, terminations, truncations, infos_next = self.env.step(
                {ATTACKER_AGENT: int(action.item()), DEFENDER_AGENT: blue_action}
            )
            reward = float(rewards[ATTACKER_AGENT])
            done = bool(terminations[ATTACKER_AGENT] or truncations[ATTACKER_AGENT])

            with torch.no_grad():
                _, next_value = self.model.forward(
                    torch.as_tensor(
                        obs_next[ATTACKER_AGENT], dtype=torch.float32, device=DEVICE
                    ).unsqueeze(0)
                )
            self.buffer.add(
                Transition(
                    obs=obs,
                    mask=mask,
                    action=int(action.item()),
                    log_prob=float(log_prob.item()),
                    value=float(value.item()),
                    reward=reward,
                    terminated=bool(terminations[ATTACKER_AGENT]),
                    truncated=bool(truncations[ATTACKER_AGENT]),
                    next_value=float(next_value.item()),
                )
            )
            self.global_step += 1
            ep_reward += reward
            ep_len += 1
            obs = np.asarray(obs_next[ATTACKER_AGENT], dtype=np.float32)
            mask = np.asarray(self.env.action_mask(ATTACKER_AGENT), dtype=bool)
            infos = infos_next

            if done:
                # Episode outcome metrics (evidence for learning curves —
                # spec §26: attack success must be measured, not assumed)
                episodes.append(
                    {
                        "reward": ep_reward,
                        "length": ep_len,
                        "attack_success": bool(
                            self.env.network is not None
                            and self.env.network.critical_assets_compromised() > 0
                        ),
                    }
                )
                ep_reward, ep_len = 0.0, 0
                obs_dict, infos = self.env.reset(seed=self._action_seed())
                self.blue_agent.reset()
                obs = np.asarray(obs_dict[ATTACKER_AGENT], dtype=np.float32)
                mask = np.asarray(self.env.action_mask(ATTACKER_AGENT), dtype=bool)

        if step_rng_state is not None:
            random.setstate(step_rng_state)

        # Bootstrap value for GAE: V(last obs). If the last transition was done,
        # GAE already cuts bootstrapping at the episode boundary (dones[i]=True).
        with torch.no_grad():
            _, last_value = self.model.forward(
                torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            )
        self.buffer.finalize(float(last_value.item()), self.config.gamma, self.config.gae_lambda)

        mean_ep_reward = float(np.mean([e["reward"] for e in episodes])) if episodes else 0.0
        return episodes, mean_ep_reward

    # -- PPO update ----------------------------------------------------------
    def update(self, extra_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        batch = self.buffer.tensors(DEVICE)
        n = batch["obs"].shape[0]

        # Re-evaluate old actions under the CURRENT policy to get new log-probs.
        with torch.no_grad():
            logits, _ = self.model.forward(batch["obs"])
            logits = logits.masked_fill(~batch["mask"], float("-inf"))
            old_dist = Categorical(logits=logits)
            old_log_probs = old_dist.log_prob(batch["actions"])  # type: ignore[no-untyped-call]

        ratio0 = torch.exp(old_log_probs - batch["old_log_probs"])

        stats = {
            "clip_frac_mean": float(((ratio0 - 1.0).abs() > self.config.clip_eps).float().mean()),
        }

        idx = np.arange(n)
        epoch_stats: list[dict[str, float]] = []
        for _ in range(self.config.epochs_per_update):
            np.random.shuffle(idx)
            for start in range(0, n, self.config.minibatch_size):
                mb = idx[start : start + self.config.minibatch_size]
                mb_t = {k: v[mb] for k, v in batch.items()}

                logits, values = self.model.forward(mb_t["obs"])
                logits = logits.masked_fill(~mb_t["mask"], float("-inf"))
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_t["actions"])  # type: ignore[no-untyped-call]
                entropy = dist.entropy().mean()  # type: ignore[no-untyped-call]

                ratio = torch.exp(new_log_probs - mb_t["old_log_probs"])
                advantages = mb_t["advantages"]
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, mb_t["returns"])

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_t["old_log_probs"] - new_log_probs).mean()
                epoch_stats.append(
                    {
                        "loss": float(loss.item()),
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy.item()),
                        "approx_kl": float(approx_kl.item()),
                    }
                )

        def mean(k: str) -> float:
            return float(np.mean([s[k] for s in epoch_stats]))

        self.update_idx += 1
        # On-policy: the batch is consumed by this update and must not leak
        # into the next rollout (a stale buffer would make collect_rollout a
        # no-op and corrupt GAE bookkeeping).
        self.buffer.clear()
        result = {
            "update": self.update_idx,
            "global_step": self.global_step,
            "clip_frac": stats["clip_frac_mean"],
            "loss": mean("loss"),
            "policy_loss": mean("policy_loss"),
            "value_loss": mean("value_loss"),
            "entropy": mean("entropy"),
            "approx_kl": mean("approx_kl"),
        }
        # Persist rollout outcome metrics alongside optimizer stats so learning
        # curves can be generated from REAL data (reward vs update, attack
        # success rate vs update). No fabricated curves downstream.
        if extra_metrics:
            result.update(extra_metrics)
        self.update_log.append(result)
        self._append_metrics_jsonl(result)
        return result

    def _append_metrics_jsonl(self, record: dict[str, Any]) -> None:
        path = Path(self.config.run_dir) / "metrics" / "training.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # -- checkpointing ---------------------------------------------------------
    def save_checkpoint(self, tag: str) -> str:
        ckpt_dir = Path(self.config.run_dir) / "checkpoints" / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_idx": self.update_idx,
            "config": self.config.to_dict(),
        }
        torch.save(payload, ckpt_dir / "model.pt")

        meta = RLCheckpoint(
            run_id=Path(self.config.run_dir).name,
            algorithm="ppo",
            agent_role=ATTACKER_AGENT,
            seed=self.config.seed,
            hyperparameters=self.config.to_dict(),
            training_steps=self.global_step,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        with open(ckpt_dir / "checkpoint_meta.json", "w") as f:
            json.dump(meta.to_dict(), f, indent=2)
        return str(ckpt_dir)

    def load_checkpoint(self, ckpt_dir: str) -> None:
        # Loading VIOLET's own training artifacts (local, not user-supplied).
        payload = torch.load(Path(ckpt_dir) / "model.pt", map_location=DEVICE, weights_only=False)  # nosec B614
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = int(payload["global_step"])
        self.update_idx = int(payload["update_idx"])
        cfg_dict = payload.get("config", {})
        for k, v in cfg_dict.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

    # -- main loop ---------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        """Run the full training loop; returns a summary dict."""
        best_reward = -float("inf")
        for _ in range(self.config.max_updates):
            episodes, mean_ep_reward = self.collect_rollout()
            attack_success_rate = (
                float(np.mean([e["attack_success"] for e in episodes])) if episodes else 0.0
            )
            self.update(
                {
                    "mean_episode_reward": mean_ep_reward,
                    "episodes_completed": len(episodes),
                    "attack_success_rate": attack_success_rate,
                }
            )

            if mean_ep_reward > best_reward:
                best_reward = mean_ep_reward
                self.save_checkpoint("best")

            if self.update_idx % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"update_{self.update_idx:05d}")

        self.save_checkpoint("final")
        return {
            "updates": self.update_idx,
            "global_step": self.global_step,
            "best_mean_episode_reward": best_reward,
            "device": str(DEVICE),
        }
