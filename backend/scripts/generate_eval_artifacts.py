"""Generate REAL evaluation + behavior-discovery artifacts from the trained PPO run.

Everything written here is computed from actual simulation episodes — no
synthetic numbers. Artifacts land in runs/ppo_red_v1/evaluation/ and are
served to the frontend by the /evaluation and /behavior API endpoints.

Run from backend/: .venv/bin/python scripts/generate_eval_artifacts.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from violet.agents import (
    ProactiveBlueAgent,
    RandomRedAgent,
    RLAgent,
    RuleBasedBlueAgent,
    ScriptedAggressiveRedAgent,
    ScriptedLateralRedAgent,
    ScriptedStealthRedAgent,
)
from violet.analytics import behavior_discovery_report
from violet.env import VIOLETEnv
from violet.types import ATTACKER_AGENT, DEFENDER_AGENT, EpisodeResult

RUN_DIR = Path("runs/ppo_red_v1")
N_EPISODES = 100
SEED_START = 10_000  # unseen seeds, disjoint from training


def make_red(kind: str, env: VIOLETEnv):
    if kind == "rl":
        agent = RLAgent("red", role="red")
        if not agent.load_checkpoint(str(RUN_DIR / "checkpoints" / "best"), env=env):
            raise SystemExit(f"checkpoint missing under {RUN_DIR}/checkpoints/best")
        return agent
    if kind == "scripted_aggressive":
        return ScriptedAggressiveRedAgent(target_node_id="database")
    if kind == "scripted_stealth":
        return ScriptedStealthRedAgent(target_node_id="database")
    if kind == "scripted_lateral":
        return ScriptedLateralRedAgent(target_node_id="database")
    if kind == "random":
        return RandomRedAgent()
    raise ValueError(kind)


def make_blue(kind: str):
    if kind == "rule_based":
        return RuleBasedBlueAgent()
    if kind == "proactive":
        return ProactiveBlueAgent()
    raise ValueError(kind)


def run_episodes(red_kind: str, blue_kind: str) -> list[EpisodeResult]:
    from violet.scenarios import v1_enterprise_network

    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=100, seed=SEED_START)
    red = make_red(red_kind, env)
    blue = make_blue(blue_kind)

    results: list[EpisodeResult] = []
    for i in range(N_EPISODES):
        seed = SEED_START + i
        _obs, infos = env.reset(seed=seed)
        red.reset()
        blue.reset()
        red_r = blue_r = 0.0
        terminated = False
        events: list[dict] = []
        steps = 0
        for steps in range(1, 101):  # noqa: B007 — steps feeds length below
            a_red = red.select_action(env, infos.get(ATTACKER_AGENT, {}))
            a_blue = blue.select_action(env, infos.get(DEFENDER_AGENT, {}))
            _obs, rewards, terms, truncs, infos = env.step(
                {ATTACKER_AGENT: a_red, DEFENDER_AGENT: a_blue}
            )
            red_r += float(rewards[ATTACKER_AGENT])
            blue_r += float(rewards[DEFENDER_AGENT])
            events.extend(infos.get(ATTACKER_AGENT, {}).get("events", []))
            events.extend(infos.get(DEFENDER_AGENT, {}).get("events", []))
            if terms[ATTACKER_AGENT] or truncs[ATTACKER_AGENT]:
                terminated = bool(terms[ATTACKER_AGENT])
                break
        assert env.network is not None
        db_comp = env.network.critical_assets_compromised() > 0
        results.append(
            EpisodeResult(
                episode_id=f"{red_kind}-vs-{blue_kind}-seed-{seed}",
                seed=seed,
                config_version="violet_v1",
                length=steps,
                red_reward=red_r,
                blue_reward=blue_r,
                winner="red" if db_comp else ("blue" if terminated else None),
                red_stats={"kind": red_kind},
                blue_stats={"kind": blue_kind},
                events=events,
            )
        )
    return results


def summarize(results: list[EpisodeResult]) -> dict:
    n = len(results)
    red_wins = sum(1 for r in results if r.winner == "red")
    blue_wins = sum(1 for r in results if r.winner == "blue")
    return {
        "n_episodes": n,
        "attack_success_rate": red_wins / n,
        "red_win_rate": red_wins / n,
        "blue_win_rate": blue_wins / n,
        "undecided_rate": (n - red_wins - blue_wins) / n,
        "mean_red_reward": sum(r.red_reward for r in results) / n,
        "mean_blue_reward": sum(r.blue_reward for r in results) / n,
        "mean_episode_length": sum(r.length for r in results) / n,
    }


def main() -> None:
    out_dir = RUN_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    matchups = [
        ("rl", "rule_based"),
        ("rl", "proactive"),
        ("scripted_aggressive", "rule_based"),
        ("scripted_stealth", "rule_based"),
        ("scripted_lateral", "rule_based"),
    ]

    all_results: dict[str, list[EpisodeResult]] = {}
    reports = []
    for red_kind, blue_kind in matchups:
        key = f"{red_kind}_vs_{blue_kind}"
        print(f"running {key} ({N_EPISODES} episodes)...", flush=True)
        results = run_episodes(red_kind, blue_kind)
        all_results[key] = results
        reports.append(
            {
                "matchup": key,
                "red_agent": red_kind,
                "blue_agent": blue_kind,
                "summary": summarize(results),
            }
        )

    eval_payload = {
        "run_id": "ppo_red_v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_episodes_per_matchup": N_EPISODES,
        "seed_range": [SEED_START, SEED_START + N_EPISODES - 1],
        "matchups": reports,
    }
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(eval_payload, f, indent=2)

    # Behavior discovery: RL red vs the scripted-aggressive baseline episodes.
    rl_results = all_results["rl_vs_rule_based"]
    baseline_results = all_results["scripted_aggressive_vs_rule_based"]
    behavior = behavior_discovery_report(
        run_id="ppo_red_v1",
        red_results=rl_results,
        blue_results=rl_results,
        red_baseline=baseline_results,
        blue_baseline=baseline_results,
        output_dir=str(out_dir),
    )
    with open(out_dir / "behavior_report.json", "w") as f:
        json.dump(behavior, f, indent=2)

    print(f"wrote {out_dir / 'eval_summary.json'}")
    print(f"wrote {out_dir / 'behavior_report.json'}")
    for m in reports:
        s = m["summary"]
        print(
            f"  {m['matchup']}: attack_success={s['attack_success_rate']:.2f} "
            f"blue_win={s['blue_win_rate']:.2f} mean_len={s['mean_episode_length']:.1f}"
        )
    print(f"done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
