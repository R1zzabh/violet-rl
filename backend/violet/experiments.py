# violet/experiments.py — experiment runner stub

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from violet.agents import (
    Agent,
    ProactiveBlueAgent,
    RandomBlueAgent,
    RandomRedAgent,
    ReactiveBlueAgent,
    RuleBasedBlueAgent,
    ScriptedAggressiveRedAgent,
    ScriptedLateralRedAgent,
    ScriptedObjectiveRedAgent,
    ScriptedRandomizedRedAgent,
    ScriptedStealthRedAgent,
)
from violet.analytics import behavior_discovery_report
from violet.env import VIOLETEnv
from violet.rewards import RewardConfig
from violet.scenarios import v1_enterprise_network
from violet.training import EpisodeMetrics, TrainingConfig, evaluate_episode, train_blue
from violet.types import EpisodeResult


@dataclass
class ExperimentConfig:
    run_id: str
    scenario: str = "v1_enterprise"
    red_policy: str = "scripted_aggressive"
    blue_policy: str = "random"
    algorithm: str = "ppo"
    seed: int = 42
    max_episodes: int = 500
    max_cycles_per_episode: int = 100
    reward_config_version: str = "v1"
    environment_version: str = "violet_v1"
    evaluation_episodes: int = 100
    evaluation_seed: int = 123


class ExperimentRunner:
    """Runs a full experiment: training + evaluation + behavior analysis + artifact storage."""

    def __init__(self, config: ExperimentConfig, base_dir: str = "runs"):
        self.config = config
        self.base_dir = base_dir
        self.env: VIOLETEnv | None = None
        self.red_agent: Agent | None = None
        self.blue_agent: Agent | None = None
        self.training_metrics: list[EpisodeMetrics] = []
        self.evaluation_metrics: list[EpisodeMetrics] = []

    def setup(self) -> None:
        """Create environment + agents from config."""
        self.env = VIOLETEnv(
            scenario_fn=v1_enterprise_network,
            max_cycles=self.config.max_cycles_per_episode,
            seed=self.config.seed,
        )
        self._make_agents()

    def _make_agents(self) -> None:
        red_map = {
            "random": RandomRedAgent(),
            "scripted_aggressive": ScriptedAggressiveRedAgent(target_node_id="database"),
            "scripted_stealth": ScriptedStealthRedAgent(target_node_id="database"),
            "scripted_lateral": ScriptedLateralRedAgent(target_node_id="database"),
            "scripted_randomized": ScriptedRandomizedRedAgent(),
            "scripted_objective": ScriptedObjectiveRedAgent(objective_node_id="database"),
        }
        blue_map = {
            "random": RandomBlueAgent(),
            "rule_based": RuleBasedBlueAgent(),
            "reactive": ReactiveBlueAgent(),
            "proactive": ProactiveBlueAgent(),
        }

        self.red_agent = red_map.get(self.config.red_policy, RandomRedAgent())
        # Blue: in phase 1 we may use random/rule-based; in later phases we'd instantiate an RLAgent.
        self.blue_agent = blue_map.get(self.config.blue_policy, RandomBlueAgent())

    def run_training(self) -> list[EpisodeMetrics]:
        """Run training episodes."""
        if self.env is None or self.red_agent is None or self.blue_agent is None:
            self.setup()
        assert self.env is not None and self.red_agent is not None and self.blue_agent is not None
        self.training_metrics = train_blue(
            self.env,
            self.red_agent,
            self.blue_agent,
            TrainingConfig(
                run_id=self.config.run_id,
                algorithm=self.config.algorithm,
                max_episodes=self.config.max_episodes,
                max_cycles_per_episode=self.config.max_cycles_per_episode,
                seed=self.config.seed,
                reward_config=RewardConfig(),
                environment_seed=self.config.seed,
                checkpoint_interval=max(1, self.config.max_episodes // 5),
                best_checkpoint_metric="blue_reward",
            ),
            base_dir=self.base_dir,
        )
        return self.training_metrics

    def run_evaluation(self, seeds: list[int] | None = None) -> list[EpisodeMetrics]:
        """Run independent evaluation episodes (different seeds from training)."""
        if self.env is None or self.red_agent is None or self.blue_agent is None:
            self.setup()
        assert self.env is not None and self.red_agent is not None and self.blue_agent is not None
        if seeds is None:
            seeds = list(
                range(
                    self.config.evaluation_seed,
                    self.config.evaluation_seed + self.config.evaluation_episodes,
                )
            )
        metrics: list[EpisodeMetrics] = []
        for s in seeds:
            m = evaluate_episode(
                self.env,
                self.red_agent,
                self.blue_agent,
                seed=s,
                max_cycles=self.config.max_cycles_per_episode,
            )
            metrics.append(m)
        self.evaluation_metrics = metrics
        return metrics

    def run_baseline_evaluation(self) -> list[EpisodeMetrics]:
        """Run the same evaluation with a baseline policy for comparison."""
        if self.env is None:
            self.setup()
        assert self.env is not None
        baseline_red = RandomRedAgent()
        baseline_blue = RandomBlueAgent()
        seeds = list(
            range(
                self.config.evaluation_seed,
                self.config.evaluation_seed + self.config.evaluation_episodes,
            )
        )
        metrics: list[EpisodeMetrics] = []
        for s in seeds:
            m = evaluate_episode(
                self.env,
                baseline_red,
                baseline_blue,
                seed=s,
                max_cycles=self.config.max_cycles_per_episode,
            )
            metrics.append(m)
        return metrics

    def persist_artifacts(self) -> Path:
        """Persist run metadata, metrics, and evaluation summary to runs/<run_id>/."""
        run_dir = Path(self.base_dir) / self.config.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "metrics").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "evaluation").mkdir(exist_ok=True)

        # config.yaml
        import yaml

        config_dict = {
            "run_id": self.config.run_id,
            "scenario": self.config.scenario,
            "red_policy": self.config.red_policy,
            "blue_policy": self.config.blue_policy,
            "algorithm": self.config.algorithm,
            "seed": self.config.seed,
            "max_episodes": self.config.max_episodes,
            "max_cycles_per_episode": self.config.max_cycles_per_episode,
            "reward_config_version": self.config.reward_config_version,
            "environment_version": self.config.environment_version,
            "evaluation_episodes": self.config.evaluation_episodes,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": self._git_commit(),
        }
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)

        # metadata.json
        metadata = {
            "run_id": self.config.run_id,
            "algorithm": self.config.algorithm,
            "environment_version": self.config.environment_version,
            "agent_role": "blue",
            "seed": self.config.seed,
            "reward_config_version": self.config.reward_config_version,
            "observation_version": "v1",
            "action_space_version": "v1",
            "hyperparameters": {},
            "training_steps": len(self.training_metrics),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": self._git_commit(),
            "checkpoint_path": str(run_dir / "checkpoints"),
            "metrics_summary": {
                "episodes": len(self.training_metrics),
                "final_red_reward": self.training_metrics[-1].red_reward
                if self.training_metrics
                else None,
                "final_blue_reward": self.training_metrics[-1].blue_reward
                if self.training_metrics
                else None,
            },
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # metrics.jsonl
        metrics_path = run_dir / "metrics" / "training.jsonl"
        with open(metrics_path, "w") as f:
            for m in self.training_metrics:
                f.write(json.dumps(m.__dict__) + "\n")

        # evaluation summary
        if self.evaluation_metrics:
            eval_path = run_dir / "evaluation" / "eval_summary.json"
            summary = {
                "run_id": self.config.run_id,
                "evaluation_episodes": len(self.evaluation_metrics),
                "seeds": [m.seed for m in self.evaluation_metrics],
                "mean_red_reward": sum(m.red_reward for m in self.evaluation_metrics)
                / len(self.evaluation_metrics),
                "mean_blue_reward": sum(m.blue_reward for m in self.evaluation_metrics)
                / len(self.evaluation_metrics),
                "attack_success_rate": sum(1 for m in self.evaluation_metrics if m.winner == "red")
                / len(self.evaluation_metrics),
                "events": [
                    m.events for m in self.evaluation_metrics[:10]
                ],  # keep a few sample episodes
            }
            with open(eval_path, "w") as f:
                json.dump(summary, f, indent=2)

        # behavior discovery report (if we have evaluation data)
        if self.evaluation_metrics:
            baseline_metrics = self.run_baseline_evaluation()

            def to_episode_result(m: EpisodeMetrics, config_version: str) -> EpisodeResult:
                return EpisodeResult(
                    episode_id=m.episode_id,
                    seed=m.seed,
                    config_version=config_version,
                    length=m.length,
                    red_reward=m.red_reward,
                    blue_reward=m.blue_reward,
                    winner=m.winner,
                    red_stats=m.red_stats,
                    blue_stats=m.blue_stats,
                    events=m.events,
                )

            baseline = [
                to_episode_result(m, self.config.environment_version) for m in baseline_metrics
            ]
            evaluation_results = [
                to_episode_result(m, self.config.environment_version)
                for m in self.evaluation_metrics
            ]
            behavior_discovery_report(
                run_id=self.config.run_id,
                red_results=evaluation_results,
                blue_results=evaluation_results,
                red_baseline=baseline,
                blue_baseline=baseline,
                output_dir=str(run_dir / "evaluation"),
            )

        return run_dir

    def _git_commit(self) -> str:
        """Best-effort git commit lookup."""
        import subprocess

        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            return out
        except Exception:
            return "unknown"
