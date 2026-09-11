# violet/env.py — PettingZoo ParallelEnv for VIOLET

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

from pettingzoo import ParallelEnv

from violet.network import Network
from violet.scenarios import v1_enterprise_network
from violet.transition import TransitionContext, TransitionEngine
from violet.types import (
    ATTACKER_AGENT,
    BlueActionType,
    DEFENDER_AGENT,
    RedActionType,
)
from violet.rewards import RewardEngine


# ---------------------------------------------------------------------------
# Optional gymnasium support — the environment works without it in smoke/dev mode.
# Real RLlib integration will import gymnasium properly.
# ---------------------------------------------------------------------------
try:
    import gymnasium as _gym  # noqa: F401
    from gymnasium import spaces
except Exception:  # pragma: no cover — degraded dev mode
    _gym = None

    class _DegradedBox:
        def __init__(self, low=-1.0, high=1.0, shape=(0,), dtype=float):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _DegradedDiscrete:
        def __init__(self, n: int):
            self.n = n
            self.dtype = "int64"

    class spaces:
        Box = _DegradedBox
        Discrete = _DegradedDiscrete


# ---------------------------------------------------------------------------
# Observation feature dimension (placeholder vector encoding for v1)
# ---------------------------------------------------------------------------
_OBS_FEATURE_DIM = 64


class VIOLETEnv(ParallelEnv):
    """PettingZoo ParallelEnv for the VIOLET multi-agent cyber environment.

    Agents:
        "red"  — attacker
        "blue" — defender

    Uses a fixed network scenario (v1_enterprise) for phase 1, with partial
    observability: each agent receives a subset of the true state.
    """

    metadata = {
        "name": "violet_v1",
        "render_modes": ["human", "ansi"],
        "max_cycles": 100,
    }

    def __init__(
        self,
        scenario_fn=v1_enterprise_network,
        max_cycles: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.scenario_fn = scenario_fn
        self.max_cycles = max_cycles
        self.seed_value = seed
        self.rng: Optional[Any] = None

        # Episode state
        self.network: Optional[Network] = None
        self.transition_engine: Optional[TransitionEngine] = None
        self.reward_engine: RewardEngine = RewardEngine()
        self.current_step: int = 0
        self.agents: List[str] = [ATTACKER_AGENT, DEFENDER_AGENT]
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.rewards: Dict[str, float] = {}
        self.observations: Dict[str, Any] = {}
        self.infos: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.last_result: Dict[str, str] = {}

        # Build the env once per instance (used for shape inference)
        self._build_initial()

        # Spaces
        n_nodes = len(self.network.nodes)
        self.red_action_space = spaces.Discrete(len(RedActionType) * n_nodes)
        self.blue_action_space = spaces.Discrete(len(BlueActionType) * n_nodes)

        self._observation_spaces = {
            ATTACKER_AGENT: spaces.Box(low=-1.0, high=1.0, shape=(_OBS_FEATURE_DIM,), dtype=float),
            DEFENDER_AGENT: spaces.Box(low=-1.0, high=1.0, shape=(_OBS_FEATURE_DIM,), dtype=float),
        }

    def _build_initial(self) -> None:
        self.network = self.scenario_fn()
        self.transition_engine = TransitionEngine(self.network, self.reward_engine, self.seed_value)
        self.current_step = 0
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.rewards = {a: 0.0 for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.events = []
        self.last_result = {a: "" for a in self.agents}

    # ------------------------------------------------------------------
    # ParallelEnv interface
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.seed_value = seed
        self._build_initial()
        self.rng = self._make_rng(self.seed_value)
        obs = self._observations()
        self.observations = obs
        self.rewards = {a: 0.0 for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.events = []
        self.last_result = {a: "" for a in self.agents}
        return obs, self.infos

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        if self.current_step >= self.max_cycles:
            for a in self.agents:
                self.truncations[a] = True
            return self.observations, self.rewards, self.terminations, self.truncations, self.infos  # type: ignore[return-value]

        self.current_step += 1
        context = TransitionContext(step=self.current_step, timestamp=time.time())
        self.events.append({"step": self.current_step, "type": "step_start"})

        red_action = self._decode_red_action(actions.get(ATTACKER_AGENT, 0))
        blue_action = self._decode_blue_action(actions.get(DEFENDER_AGENT, 0))

        self.transition_engine.execute_red_action(red_action, context, self.rng)
        self.transition_engine.execute_blue_action(blue_action, context, self.rng)
        self.network.episode_step = self.current_step

        self.rewards[ATTACKER_AGENT] = self.reward_engine.red_reward(self._reward_context(context))
        self.rewards[DEFENDER_AGENT] = self.reward_engine.blue_reward(self._reward_context(context))

        terminated = self._terminations(context)
        # PettingZoo convention: truncation fires on the first step BEYOND the
        # limit, so episode length == max_cycles exactly. Termination (terminal
        # state) still fires on the step that reaches it.
        for a in self.agents:
            self.terminations[a] = terminated
            self.truncations[a] = self.current_step > self.max_cycles or terminated

        self.observations = self._observations()

        # Red and blue share one TransitionContext, so context.events holds BOTH
        # agents' events. Deliver each agent only its own events — otherwise every
        # event is duplicated in per-agent infos (observed: 400 stream events for
        # a 100-step episode) and the dashboard event log double-counts actions.
        for a in self.agents:
            self.infos[a] = {
                "step": self.current_step,
                "events": [ev for ev in context.events if ev.get("agent") == a],
                "reward_components": self.reward_engine.pretty(a, self._reward_context(context)),
                "terminated": terminated,
            }

        self.last_result = {
            ATTACKER_AGENT: context.result or "",
            DEFENDER_AGENT: "ok",
        }
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def render(self, mode: str = "human") -> str | None:
        if self.network is None:
            return None
        if mode == "ansi":
            return self._ansi_render()
        return str(self.network.snapshot())

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        if agent == ATTACKER_AGENT:
            return self.red_action_space
        if agent == DEFENDER_AGENT:
            return self.blue_action_space
        raise KeyError(f"Unknown agent: {agent}")

    def action_mask(self, agent: str) -> List[bool]:
        """Validity mask over the flat action space, for masked-policy RL.

        Encodes only HARD constraints (the flat encoding targets every node,
        but gateway nodes accept no operations). Finer-grained validity
        (e.g. "EXPLOIT requires the vuln to exist") is intentionally left to
        the transition engine, which fails invalid actions gracefully.

        Layout matches the flat encoding: index = node_idx * n_types + type_idx.
        """
        if self.network is None:
            raise RuntimeError("Environment not initialized")
        node_ids = list(self.network.nodes.keys())
        if agent == ATTACKER_AGENT:
            types = list(RedActionType)
        elif agent == DEFENDER_AGENT:
            types = list(BlueActionType)
        else:
            raise KeyError(f"Unknown agent: {agent}")
        wait_idx = types.index([t for t in types if t.value == "WAIT"][0])
        n_types = len(types)
        mask = [False] * (len(node_ids) * n_types)
        for i, nid in enumerate(node_ids):
            node = self.network.nodes[nid]
            base = i * n_types
            if node.node_type == "gateway":
                mask[base + wait_idx] = True
            else:
                for t in range(n_types):
                    mask[base + t] = True
        return mask

    def agent_iter(self, max_iter: int = 2**32):
        for _ in range(max_iter):
            yield self.agents[:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_rng(self, seed: Optional[int]):
        import random
        return random.Random(seed) if seed is not None else random.Random()

    def _decode_red_action(self, action_index: int):
        if self.network is None:
            raise RuntimeError("Environment not initialized")
        n_nodes = max(1, len(self.network.nodes))
        node_idx = action_index // len(RedActionType)
        action_type_idx = action_index % len(RedActionType)
        if node_idx >= n_nodes:
            node_idx = n_nodes - 1
        node_ids = list(self.network.nodes.keys())
        target = node_ids[node_idx] if node_ids else "internet"
        action_type = list(RedActionType)[action_type_idx]
        return self._make_action(ATTACKER_AGENT, action_type, target)

    def _decode_blue_action(self, action_index: int):
        if self.network is None:
            raise RuntimeError("Environment not initialized")
        n_nodes = max(1, len(self.network.nodes))
        node_idx = action_index // len(BlueActionType)
        action_type_idx = action_index % len(BlueActionType)
        if node_idx >= n_nodes:
            node_idx = n_nodes - 1
        node_ids = list(self.network.nodes.keys())
        target = node_ids[node_idx] if node_ids else "internet"
        action_type = list(BlueActionType)[action_type_idx]
        return self._make_action(DEFENDER_AGENT, action_type, target)

    def _make_action(self, agent: str, action_type: Any, target: str):
        from violet.types import Action, ActionTarget
        return Action(
            agent=agent,
            action_type=action_type,
            target=ActionTarget(node_id=target),
        )

    def _reward_context(self, context: TransitionContext) -> Dict[str, Any]:
        return {
            "result": context.result,
            "detected": context.detected,
            "honeypot_triggered": context.honeypot_triggered,
            "terminated": context.terminated,
            "attack_detected": context.attack_detected,
            "attack_contained": context.attack_contained,
            "host_recovered": context.host_recovered,
            "critical_asset_protected": context.critical_asset_protected,
            "attacker_removed": context.attacker_removed,
            "unnecessary_isolation": context.unnecessary_isolation,
            "unnecessary_patch": context.unnecessary_patch,
            "unnecessary_reset": context.unnecessary_reset,
            "unnecessary_block": context.unnecessary_block,
            "unnecessary_restore": context.unnecessary_restore,
            "service_downtime": context.service_downtime,
            "critical_asset_compromised": context.critical_asset_compromised,
            "new_state_changed": context.new_state_changed,
        }

    def _terminations(self, context: TransitionContext) -> bool:
        if self.transition_engine is None:
            return False
        return self.transition_engine.should_terminate()

    def _observations(self) -> Dict[str, Any]:
        if self.network is None:
            return {a: None for a in self.agents}
        return {
            ATTACKER_AGENT: self._red_observation(),
            DEFENDER_AGENT: self._blue_observation(),
        }

    def _red_observation(self) -> List[float]:
        if self.network is None:
            return [0.0] * _OBS_FEATURE_DIM
        vector = [0.0] * _OBS_FEATURE_DIM
        idx = 0
        for nid, node in self.network.nodes.items():
            if idx >= _OBS_FEATURE_DIM:
                break
            vector[idx] = 1.0 if node.attacker_discovered else 0.0
            idx += 1
            if idx >= _OBS_FEATURE_DIM:
                break
            vector[idx] = 1.0 if node.is_compromised() else 0.0
            idx += 1
            if idx >= _OBS_FEATURE_DIM:
                break
            vector[idx] = node.value / 100.0
            idx += 1
        return vector

    def _blue_observation(self) -> List[float]:
        if self.network is None:
            return [0.0] * _OBS_FEATURE_DIM
        vector = [0.0] * _OBS_FEATURE_DIM
        idx = 0
        for nid, node in self.network.nodes.items():
            if idx >= _OBS_FEATURE_DIM:
                break
            vector[idx] = 1.0 if node.alert_level.value != "none" else 0.0
            idx += 1
            if idx >= _OBS_FEATURE_DIM:
                break
            vector[idx] = 1.0 if node.is_compromised() else 0.0
            idx += 1
            if idx >= _OBS_FEATURE_DIM:
                break
            vector[idx] = 1.0 if node.isolated.value == "isolated" else 0.0
            idx += 1
            if idx >= _OBS_FEATURE_DIM:
                break
            vector[idx] = 1.0 if node.service_status.value == "down" else 0.0
            idx += 1
            if idx >= _OBS_FEATURE_DIM:
                break
            vector[idx] = node.value / 100.0
            idx += 1
        return vector

    def _ansi_render(self) -> str:
        if self.network is None:
            return ""
        lines = []
        lines.append(f"Step {self.current_step} | Active: {self.agents}")
        lines.append(f"red last-step reward: {self.rewards.get(ATTACKER_AGENT, 0):.2f}")
        lines.append(f"blue last-step reward: {self.rewards.get(DEFENDER_AGENT, 0):.2f}")
        lines.append(f"Terminated: {any(self.terminations.values())}")
        lines.append("Network state:")
        for nid, node in self.network.nodes.items():
            lines.append(
                f"  {nid:12} {node.compromise_status.value:24} "
                f"isolated={node.isolated.value:9} aler={node.alert_level.value:5}"
            )
        lines.append(f"Events: {len(self.events)}")
        return "\n".join(lines)
