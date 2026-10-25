"""Compare scripted and learned policies on identical held-out range seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from violet.range.learning import RangePPOConfig
from violet.range.train_real import RangeTrainingRunner


def wilson_interval(successes: int, trials: int) -> list[float] | None:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and the nonnegative trial count")
    if trials == 0:
        return None
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    margin = (
        z * np.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)) / denominator
    )
    return [float(max(0.0, center - margin)), float(min(1.0, center + margin))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--red-checkpoint", required=True)
    parser.add_argument("--blue-checkpoint", required=True)
    parser.add_argument(
        "--training-seeds",
        default="",
        help="comma-separated training episode seeds for checkpoint provenance",
    )
    parser.add_argument("--output", default="runs/real_range_eval/comparison.json")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    training_seeds = [int(seed.strip()) for seed in args.training_seeds.split(",") if seed.strip()]

    matchups = (
        ("scripted_vs_scripted", "scripted", "scripted"),
        # Calibration pair confirms the challenge can report a verified attack
        # when Blue deliberately takes the already-catalogued no-op action.
        ("scripted_red_vs_passive_blue", "scripted", "passive"),
        ("random_red_vs_scripted_blue", "random", "scripted"),
        ("learned_red_vs_scripted_blue", "learned", "scripted"),
        ("scripted_red_vs_learned_blue", "scripted", "learned"),
        ("learned_vs_learned", "learned", "learned"),
    )
    output: dict[str, object] = {
        "schema_id": "violet.range.comparison.v3",
        "generated_at": datetime.now(UTC).isoformat(),
        "episodes_per_matchup": args.episodes,
        "seed_range": [args.seed, args.seed + args.episodes - 1],
        "held_out": True,
        "training_seed_range": [min(training_seeds), max(training_seeds)]
        if training_seeds
        else None,
        "training_seeds": training_seeds,
        "ppo_config": RangePPOConfig().__dict__,
        "policy_seed": args.seed,
        "scenario": "default_violet_ledger_fixture",
        "matchups": {},
    }
    for name, red, blue in matchups:
        runner = RangeTrainingRunner(
            args.api,
            str(Path(args.output).parent / name),
            seed=args.seed,
            episodes=args.episodes,
            training=False,
            red_policy=red,
            blue_policy=blue,
            red_checkpoint=args.red_checkpoint if red == "learned" else None,
            blue_checkpoint=args.blue_checkpoint if blue == "learned" else None,
        )
        records = runner.run()
        output["checkpoint_provenance"] = {
            role: (
                {
                    "path": str(Path(path)),
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "policy_version": agent.policy_version,
                }
                if path is not None
                else None
            )
            for role, path, agent in (
                ("red", args.red_checkpoint if red == "learned" else None, runner.red),
                ("blue", args.blue_checkpoint if blue == "learned" else None, runner.blue),
            )
        }
        completed = len(records)
        eligible = sum(not row["verified_attack"] for row in records)
        red_success = np.asarray([bool(row["verified_attack"]) for row in records], dtype=float)
        blue_prevented = np.asarray(
            [bool(row["verified_prevention"]) for row in records if not row["verified_attack"]],
            dtype=float,
        )
        red_returns = np.asarray([float(row["red_return"]) for row in records])
        blue_returns = np.asarray([float(row["blue_return"]) for row in records])
        legitimate_attempts = sum(int(row["legitimate_request_attempts"]) for row in records)
        legitimate_successes = sum(int(row["legitimate_request_successes"]) for row in records)
        false_blocks = sum(int(row["false_blocked_legitimate_requests"]) for row in records)
        bootstrap_rng = np.random.default_rng(args.seed + 900_000)

        def bootstrap_interval(
            values: np.ndarray, rng: np.random.Generator = bootstrap_rng
        ) -> list[float] | None:
            if values.size == 0:
                return None
            draws = rng.choice(values, size=(2000, values.size), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            return [float(low), float(high)]

        stats = {
            "completed_episodes": completed,
            "verified_red_success_rate": float(red_success.mean()) if completed else None,
            "verified_red_success_rate_wilson_95ci": wilson_interval(
                int(red_success.sum()), completed
            ),
            "verified_blue_prevention_rate": float(blue_prevented.mean()) if eligible else None,
            "verified_blue_prevention_rate_wilson_95ci": wilson_interval(
                int(blue_prevented.sum()), eligible
            ),
            "blue_prevention_eligible_episodes": eligible,
            "legitimate_request_attempts": legitimate_attempts,
            "legitimate_request_successes": legitimate_successes,
            "legitimate_request_success_rate": (
                legitimate_successes / legitimate_attempts if legitimate_attempts else None
            ),
            "false_blocked_legitimate_requests": false_blocks,
            "false_block_rate": false_blocks / legitimate_attempts if legitimate_attempts else None,
            "mean_red_return": float(red_returns.mean()) if completed else None,
            "mean_red_return_bootstrap_95ci": bootstrap_interval(red_returns),
            "mean_blue_return": float(blue_returns.mean()) if completed else None,
            "mean_blue_return_bootstrap_95ci": bootstrap_interval(blue_returns),
            "mean_episode_length": sum(int(row["episode_length"]) for row in records)
            / max(1, completed),
            "mean_selected_action_probability": {
                role: float(
                    np.mean([row["mean_selected_action_probability"][role] for row in records])
                )
                if completed
                else None
                for role in ("red", "blue")
            },
            "mean_policy_entropy": {
                role: float(np.mean([row["mean_policy_entropy"][role] for row in records]))
                if completed
                else None
                for role in ("red", "blue")
            },
            "action_frequencies": {
                role: {
                    action: float(
                        np.mean([row["action_frequencies"][role][action] for row in records])
                    )
                    if completed
                    else None
                    for action in (records[0]["action_frequencies"][role] if completed else {})
                }
                for role in ("red", "blue")
            },
            "invalid_action_rate": sum(int(row["invalid_action_count"]) for row in records)
            / max(1, sum(int(row["request_count"]) for row in records)),
            "mean_http_latency_ms": sum(float(row["mean_http_latency_ms"]) for row in records)
            / max(1, completed),
            "episodes": records,
        }
        output["matchups"][name] = stats  # type: ignore[index]
        runner.client.close()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "matchups": output["matchups"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
