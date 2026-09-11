# violet/evaluation.py — independent evaluation of policies on unseen seeds
#
# Phase 5: evaluation is deliberately separate from training. It never reuses
# training seeds (unless explicitly asked to), runs deterministic greedy
# policies by default, and produces a JSON-serializable report.

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from violet.agents.base import Agent
from violet.env import VIOLETEnv
from violet.types import ATTACKER_AGENT, DEFENDER_AGENT


@dataclass
class EvaluationConfig:
    n_episodes: int = 20
    seed_start: int = 10_000       # unseen-seed range, disjoint from training seeds
    max_cycles: int = 100
    deterministic: bool = True     # greedy actions for both RL agents

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeRecord:
    seed: int
    length: int
    red_reward: float
    blue_reward: float
    winner: Optional[str]
    database_compromised: bool
    compromised_nodes: List[str]
    terminated: bool  # terminal state (vs truncation at the step limit)


@dataclass
class EvaluationReport:
    config: Dict[str, Any]
    episodes: List[EpisodeRecord] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "episodes": [asdict(e) for e in self.episodes],
            "summary": self.summary,
        }

    def save(self, path: str) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(out)


def _finalize(
    env: VIOLETEnv,
    seed: int,
    red_reward: float,
    blue_reward: float,
    length: int,
    terminated: bool,
) -> EpisodeRecord:
    network = env.network
    assert network is not None
    compromised = [
        nid for nid, node in network.nodes.items() if node.is_compromised()
    ]
    critical_compromised = [
        nid
        for nid, node in network.nodes.items()
        if node.is_critical_asset and node.is_compromised()
    ]
    winner = "red" if critical_compromised else ("blue" if terminated else None)
    return EpisodeRecord(
        seed=seed,
        length=length,
        red_reward=red_reward,
        blue_reward=blue_reward,
        winner=winner,
        database_compromised=bool(critical_compromised),
        compromised_nodes=compromised,
        terminated=terminated,
    )


def evaluate_pair(
    env: VIOLETEnv,
    red_agent: Agent,
    blue_agent: Agent,
    config: Optional[EvaluationConfig] = None,
) -> EvaluationReport:
    """Run head-to-head episodes on unseen seeds and aggregate metrics.

    Works with any Agent (scripted, rule-based, or RL via RLAgent wrapper).
    RL agents must be loaded/constructed with deterministic behavior before
    being passed in; the config flag is recorded in the report.
    """
    cfg = config or EvaluationConfig()
    report = EvaluationReport(config=cfg.to_dict())

    for i in range(cfg.n_episodes):
        seed = cfg.seed_start + i
        obs, infos = env.reset(seed=seed)
        red_agent.reset()
        blue_agent.reset()

        red_r = blue_r = 0.0
        terminated = False
        steps = 0
        for steps in range(1, cfg.max_cycles + 1):
            red_action = red_agent.select_action(env, infos.get(ATTACKER_AGENT, {}))
            blue_action = blue_agent.select_action(env, infos.get(DEFENDER_AGENT, {}))
            obs, rewards, terminations, truncations, infos = env.step(
                {ATTACKER_AGENT: red_action, DEFENDER_AGENT: blue_action}
            )
            red_r += float(rewards[ATTACKER_AGENT])
            blue_r += float(rewards[DEFENDER_AGENT])
            if terminations[ATTACKER_AGENT] or truncations[ATTACKER_AGENT]:
                terminated = bool(terminations[ATTACKER_AGENT])
                break

        report.episodes.append(
            _finalize(env, seed, red_r, blue_r, steps, terminated)
        )

    report.summary = _summarize(report.episodes)
    return report


def _summarize(episodes: List[EpisodeRecord]) -> Dict[str, Any]:
    n = len(episodes)
    if n == 0:
        return {"n_episodes": 0}
    red_wins = sum(1 for e in episodes if e.winner == "red")
    blue_wins = sum(1 for e in episodes if e.winner == "blue")
    db = [1.0 if e.database_compromised else 0.0 for e in episodes]
    return {
        "n_episodes": n,
        "attack_success_rate": sum(db) / n,
        "red_win_rate": red_wins / n,
        "blue_win_rate": blue_wins / n,
        "undecided_rate": (n - red_wins - blue_wins) / n,
        "mean_red_reward": statistics.fmean(e.red_reward for e in episodes),
        "mean_blue_reward": statistics.fmean(e.blue_reward for e in episodes),
        "std_red_reward": statistics.pstdev([e.red_reward for e in episodes]) if n > 1 else 0.0,
        "mean_episode_length": statistics.fmean(e.length for e in episodes),
        "mean_compromised_nodes": statistics.fmean(len(e.compromised_nodes) for e in episodes),
    }
