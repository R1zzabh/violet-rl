# violet/training.py — training loop skeleton

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from violet.agents import Agent
from violet.env import VIOLETEnv
from violet.rewards import RewardConfig
from violet.types import ATTACKER_AGENT, DEFENDER_AGENT


@dataclass
class EpisodeMetrics:
    episode_id: str
    seed: int
    length: int
    red_reward: float
    blue_reward: float
    red_steps: int
    blue_steps: int
    winner: str | None
    red_stats: Dict[str, Any] = field(default_factory=dict)
    blue_stats: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TrainingConfig:
    run_id: str
    algorithm: str = "ppo"
    max_episodes: int = 2000
    max_cycles_per_episode: int = 100
    checkpoint_interval: int = 500
    best_checkpoint_metric: str = "blue_reward"  # or "red_reward" depending on which agent we train
    seed: int = 42
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    environment_seed: int = 42


def _make_run_dir(base_dir: str, run_id: str) -> Path:
    path = Path(base_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics").mkdir(exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    (path / "evaluation").mkdir(exist_ok=True)
    return path


def _write_metrics_jsonl(run_dir: Path, metrics: List[EpisodeMetrics]) -> None:
    metrics_path = run_dir / "metrics" / "training.jsonl"
    with open(metrics_path, "a") as f:
        for m in metrics:
            f.write(json.dumps(m.__dict__) + "\n")


def train_red(
    env: VIOLETEnv,
    red_agent: Agent,
    blue_agent: Agent,
    config: TrainingConfig,
    base_dir: str = "runs",
    on_episode: Optional[Callable[[EpisodeMetrics], None]] = None,
) -> List[EpisodeMetrics]:
    """Train RED against a fixed BLUE for one phase.

    Skeleton: extends to full RLlib integration later. For now it runs the
    environment with the given agents and records metrics.
    """
    run_dir = _make_run_dir(base_dir, config.run_id)
    metrics: List[EpisodeMetrics] = []
    best_metric: float = -float("inf")

    for episode in range(1, config.max_episodes + 1):
        obs, infos = env.reset(seed=config.environment_seed + episode)
        red_agent.reset()
        blue_agent.reset()

        episode_reward_red = 0.0
        episode_reward_blue = 0.0
        episode_events: List[Dict[str, Any]] = []
        terminated = False

        for step in range(config.max_cycles_per_episode):
            if terminated:
                break

            red_action = red_agent.select_action(env, infos.get(ATTACKER_AGENT, {}))
            blue_action = blue_agent.select_action(env, infos.get(DEFENDER_AGENT, {}))

            actions = {
                ATTACKER_AGENT: red_action,
                DEFENDER_AGENT: blue_action,
            }
            obs, rewards, terminations, truncations, infos = env.step(actions)

            episode_reward_red += rewards.get(ATTACKER_AGENT, 0.0)
            episode_reward_blue += rewards.get(DEFENDER_AGENT, 0.0)
            terminated = any(terminations.values()) or any(truncations.values())

            for ev in infos.get(ATTACKER_AGENT, {}).get("events", []):
                episode_events.append(ev)
            for ev in infos.get(DEFENDER_AGENT, {}).get("events", []):
                episode_events.append(ev)

        # Determine winner
        winner = None
        if env.network is not None and env.network.critical_assets_compromised() > 0:
            winner = "red"
        elif episode_reward_blue > episode_reward_red + 10:
            winner = "blue"
        elif episode_reward_red > episode_reward_blue + 10:
            winner = "red"

        metrics_obj = EpisodeMetrics(
            episode_id=f"{config.run_id}-episode-{episode}",
            seed=config.environment_seed + episode,
            length=step + 1,
            red_reward=episode_reward_red,
            blue_reward=episode_reward_blue,
            red_steps=step + 1,
            blue_steps=step + 1,
            winner=winner,
            red_stats={},
            blue_stats={},
            events=episode_events[-50:],  # keep a bounded number of events per episode
        )
        metrics.append(metrics_obj)

        if on_episode is not None:
            on_episode(metrics_obj)

        # Checkpointing
        if episode % config.checkpoint_interval == 0:
            ckpt_dir = run_dir / "checkpoints" / f"checkpoint_{episode:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # In V1 we store metadata + a placeholder; the real model is saved by RLlib.
            ckpt_meta = {
                "run_id": config.run_id,
                "algorithm": config.algorithm,
                "episode": episode,
                "red_reward": episode_reward_red,
                "blue_reward": episode_reward_blue,
                "seed": config.seed,
            }
            with open(ckpt_dir / "checkpoint_meta.json", "w") as f:
                json.dump(ckpt_meta, f, indent=2)

            if config.best_checkpoint_metric == "blue_reward":
                if episode_reward_blue > best_metric:
                    best_metric = episode_reward_blue
                    best_dir = run_dir / "checkpoints" / "checkpoint_best"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    with open(best_dir / "checkpoint_meta.json", "w") as f:
                        json.dump(ckpt_meta, f, indent=2)

        # Write metrics incrementally (so a crash doesn't lose everything)
        _write_metrics_jsonl(run_dir, metrics[-50:])  # keep recent in the file

    # Final metrics file
    _write_metrics_jsonl(run_dir, metrics)
    return metrics


def train_blue(
    env: VIOLETEnv,
    red_agent: Agent,
    blue_agent: Agent,
    config: TrainingConfig,
    base_dir: str = "runs",
    on_episode: Optional[Callable[[EpisodeMetrics], None]] = None,
) -> List[EpisodeMetrics]:
    """Train BLUE against a fixed RED for phase 1.

    Same skeleton as train_red but conceptually the defender is learning.
    """
    return train_red(env, blue_agent, red_agent, config, base_dir, on_episode)


def evaluate_episode(
    env: VIOLETEnv,
    red_agent: Agent,
    blue_agent: Agent,
    seed: int,
    max_cycles: int = 100,
) -> EpisodeMetrics:
    """Run a single evaluation episode and return metrics.

    Used for independent evaluation, separate from training.
    """
    obs, infos = env.reset(seed=seed)
    red_agent.reset()
    blue_agent.reset()

    episode_reward_red = 0.0
    episode_reward_blue = 0.0
    episode_events: List[Dict[str, Any]] = []
    terminated = False

    for step in range(max_cycles):
        if terminated:
            break
        red_action = red_agent.select_action(env, infos.get(ATTACKER_AGENT, {}))
        blue_action = blue_agent.select_action(env, infos.get(DEFENDER_AGENT, {}))
        actions = {
            ATTACKER_AGENT: red_action,
            DEFENDER_AGENT: blue_action,
        }
        obs, rewards, terminations, truncations, infos = env.step(actions)
        episode_reward_red += rewards.get(ATTACKER_AGENT, 0.0)
        episode_reward_blue += rewards.get(DEFENDER_AGENT, 0.0)
        terminated = any(terminations.values()) or any(truncations.values())
        for ev in infos.get(ATTACKER_AGENT, {}).get("events", []):
            episode_events.append(ev)
        for ev in infos.get(DEFENDER_AGENT, {}).get("events", []):
            episode_events.append(ev)

    winner = None
    if env.network is not None and env.network.critical_assets_compromised() > 0:
        winner = "red"

    return EpisodeMetrics(
        episode_id=f"eval-seed-{seed}",
        seed=seed,
        length=step + 1,
        red_reward=episode_reward_red,
        blue_reward=episode_reward_blue,
        red_steps=step + 1,
        blue_steps=step + 1,
        winner=winner,
        red_stats={},
        blue_stats={},
        events=episode_events[-50:],
    )
