# violet/cli.py — command-line interface for VIOLET

from __future__ import annotations

import argparse

from violet.env import VIOLETEnv
from violet.scenarios import v1_enterprise_network, v1_scenario_config


def _cmd_scenario_list(_: argparse.Namespace) -> int:
    cfg = v1_scenario_config()
    print(f"Name:    {cfg['name']}")
    print(f"Nodes:   {cfg['node_count']}")
    print(f"Critical assets: {cfg['critical_assets']}")
    print("Topology:")
    for node, edges in cfg["topology"].items():
        print(f"  {node:12} -> {edges['outgoing']}")
    return 0


def _cmd_env_smoke(_: argparse.Namespace) -> int:
    """Run a short smoke test: reset + a few random steps + render."""
    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=10, seed=42)
    env.reset(seed=42)
    print(f"Reset OK. Agents: {list(env.agents)}")
    for step in range(3):
        assert env.rng is not None, "environment RNG not initialized"
        actions = {
            "red": env.rng.randint(0, env.action_space("red").n - 1),
            "blue": env.rng.randint(0, env.action_space("blue").n - 1),
        }
        _obs, rewards, terminations, _truncations, _infos = env.step(actions)
        print(
            f"Step {step + 1}: red_reward={rewards['red']:.2f}, blue_reward={rewards['blue']:.2f}"
        )
        if any(terminations.values()):
            print("Terminated.")
            break
    print("Network snapshot:")
    print(env.render("ansi"))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Run PPO training (verified pipeline: train -> checkpoints -> metrics)."""
    from violet.agents import RuleBasedBlueAgent
    from violet.algorithms.ppo import PPOConfig, PPOTrainer

    updates = getattr(args, "updates", 50)
    seed = getattr(args, "seed", 42)
    run_dir = getattr(args, "run_dir", None) or f"runs/ppo_red_seed{seed}"

    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=50, seed=seed)
    blue = RuleBasedBlueAgent()
    cfg = PPOConfig(
        rollout_steps=512,
        max_updates=updates,
        checkpoint_interval=max(1, updates // 2),
        seed=seed,
        run_dir=run_dir,
    )
    trainer = PPOTrainer(env, blue, cfg)
    summary = trainer.train()
    print("Training complete.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  metrics: {run_dir}/metrics/training.jsonl")
    print(f"  checkpoints: {run_dir}/checkpoints/")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    """Run a short manual simulation and print events — useful for debugging."""
    steps = getattr(args, "steps", 5)
    seed = getattr(args, "seed", 42)

    from violet.agents import RuleBasedBlueAgent, ScriptedAggressiveRedAgent

    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=steps, seed=seed)
    red = ScriptedAggressiveRedAgent(target_node_id="database")
    blue = RuleBasedBlueAgent()

    env.reset(seed=seed)
    red.reset()
    blue.reset()
    print(f"Episode seed={seed}, steps={steps}")
    for step in range(steps):
        red_action = red.select_action(env, env.infos.get("red", {}))
        blue_action = blue.select_action(env, env.infos.get("blue", {}))
        actions = {"red": red_action, "blue": blue_action}
        _obs, _rewards, terminations, _truncations, infos = env.step(actions)

        for agent_name in ("red", "blue"):
            for ev in infos.get(agent_name, {}).get("events", []):
                print(
                    f"  [{ev['step']}] {ev['agent']} -> {ev['action']} {ev['target'] or ''} : {ev['result']}"
                )

        if any(terminations.values()):
            print(f"Terminated at step {step + 1}")
            break

    print(env.render("ansi"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="violet", description="VIOLET: Multi-Agent RL Cyber Attack & Defense Simulator"
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("scenario", help="list available scenarios")
    p_list.add_argument("list", nargs="?", default=None, help="list scenarios (alias for 'list')")

    p_env = sub.add_parser("env", help="environment commands")
    p_env.add_argument("test", nargs="?", default=None, help="smoke test the environment")

    p_train = sub.add_parser("train", help="run PPO training (red vs rule-based blue)")
    p_train.add_argument("--updates", type=int, default=50, help="number of PPO updates")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument(
        "--run-dir", type=str, default=None, help="output directory (default: runs/ppo_red_seed<N>)"
    )

    p_replay = sub.add_parser("replay", help="run a short manual episode for debugging")
    p_replay.add_argument("--steps", type=int, default=10)
    p_replay.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.command == "scenario":
        return _cmd_scenario_list(args)
    if args.command == "env":
        return _cmd_env_smoke(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "replay":
        return _cmd_replay(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
