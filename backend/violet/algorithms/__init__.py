# violet/algorithms — swappable RL algorithm implementations
#
# Phase 5: PPO (self-contained PyTorch, module `violet.algorithms.ppo`).
# PyTorch is an optional dependency (`pip install "violet[rl]"`), so this
# package deliberately does NOT import ppo.py at package-import time.
# Import the symbols you need directly:
#
#     from violet.algorithms.ppo import ActorCritic, PPOConfig, PPOTrainer
#
# Design rationale (why not RLlib for Phase 5): the V1 environment is tiny
# (48 red actions, 64-dim observations), so a single-file PPO is simpler,
# fully debuggable, and keeps Ray out of the dev dependency tree. The
# RLAgent wrapper in violet.agents.rl keeps the door open for an RLlib swap.
