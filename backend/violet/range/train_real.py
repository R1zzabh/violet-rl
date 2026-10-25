"""Small, reproducible PPO trainer that communicates only through the range API."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import numpy as np

from violet.range.contracts import (
    AuthTestAction,
    CheckDocumentAccessAction,
    CheckTargetHealthAction,
    ControlRequest,
    DiscoverRouteAction,
    ExecutorJobRequest,
)
from violet.range.learning import (
    BLUE_ACTIONS,
    RED_ACTIONS,
    PolicyDecision,
    RangeObservation,
    RangePPOAgent,
    RangePPOConfig,
    RangeTransition,
    Role,
)


class ScriptedRangePolicy:
    """Deterministic in-range baseline using the same bounded action catalog."""

    def __init__(self, role: str) -> None:
        self.role = role
        self.actions = RED_ACTIONS if role == "red" else BLUE_ACTIONS
        self.policy_version = 0
        self.transitions: list[RangeTransition] = []

    def act(
        self,
        observation: RangeObservation,
        *,
        action_mask: np.ndarray | None = None,
        training: bool,
    ) -> PolicyDecision:
        del training
        if self.role == "red":
            if observation.features[2] < 0.5:
                proposed = 3  # Alice synthetic authentication
            elif observation.features[6] < 0.5:
                proposed = 6  # Alice attempts Bob's synthetic document
            else:
                proposed = 7  # Objective already verified: stop requesting documents
        else:
            proposed = 0 if observation.features[7] > 0.0 else 1
        mask = np.ones(len(self.actions), dtype=bool) if action_mask is None else action_mask
        if not mask[proposed]:
            proposed = int(np.flatnonzero(mask)[0])
        return PolicyDecision(proposed, 0.0, 0.0, self.policy_version)

    @staticmethod
    def value(_observation: RangeObservation) -> float:
        return 0.0

    @staticmethod
    def record(_transition: RangeTransition) -> None:
        return None

    def update(self) -> dict[str, float | int]:
        return {"updates": 0, "transitions": 0}


class RandomRangePolicy(ScriptedRangePolicy):
    """Seeded uniform baseline over only the currently valid catalog actions."""

    def __init__(self, role: str, seed: int) -> None:
        super().__init__(role)
        self.rng = np.random.default_rng(seed)

    def act(
        self,
        observation: RangeObservation,
        *,
        action_mask: np.ndarray | None = None,
        training: bool,
    ) -> PolicyDecision:
        del observation, training
        mask = np.ones(len(self.actions), dtype=bool) if action_mask is None else action_mask
        choices = np.flatnonzero(mask)
        if not len(choices):
            raise ValueError("action mask must permit at least one catalogued action")
        action = int(self.rng.choice(choices))
        probability = 1.0 / len(choices)
        return PolicyDecision(
            action,
            float(np.log(probability)),
            0.0,
            self.policy_version,
            probability,
            float(np.log(len(choices))),
        )


class PassiveBluePolicy(ScriptedRangePolicy):
    """No-op calibration baseline used to verify the target detects an attack."""

    def __init__(self) -> None:
        super().__init__("blue")

    def act(
        self,
        observation: RangeObservation,
        *,
        action_mask: np.ndarray | None = None,
        training: bool,
    ) -> PolicyDecision:
        del observation, training
        mask = np.ones(len(self.actions), dtype=bool) if action_mask is None else action_mask
        if not mask[0]:
            raise ValueError("the passive Blue calibration requires no-op to be valid")
        return PolicyDecision(0, 0.0, 0.0, self.policy_version)


def red_reward_components(
    *,
    verified_foreign_read: bool,
    action_id: int,
    successful_http_response: bool,
    principal_ref: str | None,
) -> dict[str, float]:
    """Reward verified objective progress and Alice setup, not irrelevant HTTP success."""
    return {
        "verified_foreign_read": 1.0 if verified_foreign_read else 0.0,
        "successful_alice_authentication": (
            0.05
            if action_id == 3 and successful_http_response and principal_ref == "alice"
            else 0.0
        ),
        "request_failure_cost": -0.1 if not successful_http_response else 0.0,
    }


class RangeTrainingRunner:
    """Alternating Red actions and a Blue ownership-control decision per episode."""

    def __init__(
        self,
        api_url: str,
        run_dir: str,
        *,
        seed: int = 42,
        episodes: int = 10,
        red_steps: int = 6,
        training: bool = True,
        red_checkpoint: str | None = None,
        blue_checkpoint: str | None = None,
        red_policy: str = "learned",
        blue_policy: str = "learned",
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.run_dir = Path(run_dir)
        self.seed = seed
        self.episodes = episodes
        self.red_steps = red_steps
        self.training = training
        self.run_id = str(uuid4())
        self.current_episode_id: str | None = None
        self.client = httpx.Client(base_url=self.api_url, timeout=10.0)
        self.red = self._make_policy("red", red_policy, seed)
        self.blue = self._make_policy("blue", blue_policy, seed + 1)
        if red_checkpoint:
            self.red.load(red_checkpoint, inference_only=not training)
        if blue_checkpoint:
            self.blue.load(blue_checkpoint, inference_only=not training)

    @staticmethod
    def _make_policy(role: Role, policy: str, seed: int) -> Any:
        if policy == "scripted":
            return ScriptedRangePolicy(role)
        if policy == "random":
            return RandomRangePolicy(role, seed)
        if policy == "passive" and role == "blue":
            return PassiveBluePolicy()
        if policy == "learned":
            return RangePPOAgent(role, RangePPOConfig(seed=seed))
        raise ValueError(f"unknown {role} policy: {policy}")

    def _wait(self, path: str, *, timeout: float = 15.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.client.get(path)
            response.raise_for_status()
            payload = cast("dict[str, Any]", response.json())
            if payload.get("status") not in {"queued", "queued_or_running", "running"}:
                return payload
            time.sleep(0.1)
        raise TimeoutError(f"range operation timed out: {path}")

    def _reset(self, run_id: str, episode_id: str, seed: int) -> None:
        response = self.client.post(
            "/range/reset", json={"seed": seed, "run_id": run_id, "episode_id": episode_id}
        )
        response.raise_for_status()
        ack = self._wait(f"/range/controls/{response.json()['command_id']}")
        if ack.get("status") != "applied":
            raise RuntimeError(f"target reset failed: {ack}")

    def _job(self, action: Any, run_id: str, episode_id: str, step: int) -> dict[str, Any]:
        request = ExecutorJobRequest(
            action=action,
            run_id=UUID(run_id),
            episode_id=UUID(episode_id),
            step_id=step,
        )
        response = self.client.post("/range/jobs", json=request.model_dump(mode="json"))
        response.raise_for_status()
        return self._wait(f"/range/jobs/{response.json()['job_id']}")

    @staticmethod
    def _red_action(action_id: int, sessions: dict[str, str]) -> Any:
        if action_id < 3:
            return DiscoverRouteAction(
                operation="DISCOVER_ROUTE",
                route_candidate_id=("home", "login", "document")[action_id],
            )
        if action_id in (3, 4):
            return AuthTestAction(
                operation="AUTH_TEST",
                credential_ref="synthetic_alice" if action_id == 3 else "synthetic_bob",
            )
        if action_id in (5, 6):
            return CheckDocumentAccessAction(
                operation="CHECK_DOCUMENT_ACCESS",
                document_ref="doc_alice" if action_id == 5 else "doc_bob",
                # This exercise's objective is an authenticated Alice-to-Bob IDOR.
                session_handle=sessions.get("alice"),
            )
        return CheckTargetHealthAction(operation="CHECK_TARGET_HEALTH")

    def _persist_transition(
        self,
        *,
        run_id: str,
        episode_id: str,
        seed: int,
        role: str,
        observation: RangeObservation,
        decision: Any,
        mask: np.ndarray,
        accepted: bool,
        executed_action: int | None,
        reward: float,
        components: dict[str, float],
        next_observation: RangeObservation,
        next_value: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        payload = {
            "run_id": run_id,
            "episode_id": episode_id,
            "seed": seed,
            "role": role,
            "schema_id": observation.schema_id,
            "observation": observation.features,
            "proposed_action": decision.action,
            "action_mask": mask.tolist(),
            "accepted": accepted,
            "executed_action": executed_action,
            "reward": reward,
            "reward_components": components,
            "next_observation": next_observation.features,
            "next_value": next_value,
            "terminated": terminated,
            "truncated": truncated,
            "policy_version": decision.policy_version,
            "selected_action_probability": decision.action_probability,
            "policy_entropy": decision.entropy,
            "mode": "training" if self.training else "evaluation",
        }
        response = self.client.post("/range/training/transitions", json=payload)
        response.raise_for_status()

    def run_episode(self, episode_number: int) -> dict[str, Any]:
        run_id, episode_id = self.run_id, str(uuid4())
        self.current_episode_id = episode_id
        episode_seed = self.seed + episode_number
        self._reset(run_id, episode_id, episode_seed)
        health_response = self.client.get("/range/health")
        health_response.raise_for_status()
        if health_response.json().get("status") != "ok":
            raise RuntimeError(
                f"range target is unhealthy before episode: {health_response.json()}"
            )
        service_healthy = True
        sessions: dict[str, str] = {}
        request_count = failed_count = 0
        last_status: int | None = None
        last_success = False
        verified_objective = 0
        confirmed_attack = False
        prevention_verified = False
        ownership_hardened = False
        red_return = 0.0
        blue_return = 0.0
        elapsed_ms: list[float] = []
        invalid_actions = 0
        legitimate_request_attempts = 0
        legitimate_request_successes = 0
        false_blocked_legitimate_requests = 0
        red_actions: list[int] = []
        blue_actions: list[int] = []
        selected_probabilities: dict[str, list[float]] = {"red": [], "blue": []}
        policy_entropies: dict[str, list[float]] = {"red": [], "blue": []}

        for step in range(self.red_steps):
            observation = RangeObservation.red(
                step=step,
                budget=self.red_steps,
                authenticated_alice="alice" in sessions,
                authenticated_bob="bob" in sessions,
                last_success=last_success,
                last_status=last_status,
                verified_objectives=verified_objective,
            )
            mask = np.ones(len(RED_ACTIONS), dtype=bool)
            mask[5] = "alice" in sessions
            mask[6] = "alice" in sessions
            decision = self.red.act(observation, action_mask=mask, training=self.training)
            action = self._red_action(decision.action, sessions)
            result_payload = self._job(action, run_id, episode_id, step)
            result = result_payload["result"]
            accepted = result.get("status") == "completed"
            executed = decision.action if accepted else None
            invalid_actions += int(not accepted)
            request_count += 1
            elapsed_ms.append(float(result.get("duration_ms", 0.0)))
            last_status = result.get("http_status")
            last_success = result.get("status") == "completed" and last_status == 200
            failed_count += int(not last_success)
            principal = result.get("principal_ref")
            if decision.action == 5 and principal == "alice":
                legitimate_request_attempts += 1
                legitimate_request_successes += int(last_status == 200 and accepted)
                false_blocked_legitimate_requests += int(last_status in {403, 423, 429})
            session_handle = result.get("session_handle")
            if principal in {"alice", "bob"} and session_handle:
                sessions[str(principal)] = str(session_handle)
            finding = result_payload.get("finding") or {}
            newly_confirmed_attack = not confirmed_attack and finding.get("status") == "confirmed"
            confirmed_attack = confirmed_attack or newly_confirmed_attack
            if newly_confirmed_attack:
                verified_objective += 1
            components = red_reward_components(
                verified_foreign_read=newly_confirmed_attack,
                action_id=decision.action,
                successful_http_response=last_success,
                principal_ref=str(principal) if principal is not None else None,
            )
            reward = sum(components.values())
            next_observation = RangeObservation.red(
                step=step + 1,
                budget=self.red_steps,
                authenticated_alice="alice" in sessions,
                authenticated_bob="bob" in sessions,
                last_success=last_success,
                last_status=last_status,
                verified_objectives=verified_objective,
            )
            next_value = self.red.value(next_observation)
            is_last = step == self.red_steps - 1
            red_return += reward
            if accepted and self.training:
                self.red.record(
                    RangeTransition(
                        observation.array(),
                        mask,
                        decision,
                        reward,
                        next_value,
                        False,
                        is_last,
                    )
                )
            self._persist_transition(
                run_id=run_id,
                episode_id=episode_id,
                seed=episode_seed,
                role="red",
                observation=observation,
                decision=decision,
                mask=mask,
                accepted=accepted,
                executed_action=executed,
                reward=reward,
                components=components,
                next_observation=next_observation,
                next_value=next_value,
                terminated=False,
                truncated=is_last,
            )
            red_actions.append(decision.action)
            selected_probabilities["red"].append(decision.action_probability)
            policy_entropies["red"].append(decision.entropy)

            # Blue observes only published counters, service health, and the
            # independent finding produced by the just-completed Red request.
            blue_obs = RangeObservation.blue(
                step=step + 1,
                budget=self.red_steps,
                requests=request_count,
                failed_requests=failed_count,
                service_healthy=service_healthy,
                verified_attack=confirmed_attack,
                verified_prevention=prevention_verified,
                controls_applied=int(ownership_hardened),
            )
            blue_mask = np.array([True, not ownership_hardened, False, False], dtype=bool)
            blue_decision = self.blue.act(blue_obs, action_mask=blue_mask, training=self.training)
            blue_actions.append(blue_decision.action)
            selected_probabilities["blue"].append(blue_decision.action_probability)
            policy_entropies["blue"].append(blue_decision.entropy)
            control_applied = blue_decision.action == 0
            control_id: str | None = None
            if blue_decision.action == 1:
                response = self.client.post(
                    "/range/controls",
                    json=ControlRequest(
                        action="HARDEN_DOCUMENT_OWNERSHIP",
                        run_id=UUID(run_id),
                        episode_id=UUID(episode_id),
                    ).model_dump(mode="json"),
                )
                response.raise_for_status()
                control_id = response.json()["command_id"]
                ack = self._wait(f"/range/controls/{control_id}")
                control_applied = ack.get("status") == "applied"
                ownership_hardened = ownership_hardened or control_applied

            # Verify the control only after its acknowledgement. A finding that
            # Red already confirmed earns Blue no prevention credit, even if
            # subsequent requests are denied successfully.
            verification_status = "not_run"
            if control_applied and blue_decision.action == 1 and control_id:
                alice = self._job(
                    AuthTestAction(operation="AUTH_TEST", credential_ref="synthetic_alice"),
                    run_id,
                    episode_id,
                    50 + step * 4,
                )
                bob = self._job(
                    AuthTestAction(operation="AUTH_TEST", credential_ref="synthetic_bob"),
                    run_id,
                    episode_id,
                    51 + step * 4,
                )
                alice_handle = alice["result"].get("session_handle")
                bob_handle = bob["result"].get("session_handle")
                if alice_handle and bob_handle:
                    unauthorized = self._job(
                        CheckDocumentAccessAction(
                            operation="CHECK_DOCUMENT_ACCESS",
                            document_ref="doc_bob",
                            session_handle=alice_handle,
                        ),
                        run_id,
                        episode_id,
                        52 + step * 4,
                    )
                    owner = self._job(
                        CheckDocumentAccessAction(
                            operation="CHECK_DOCUMENT_ACCESS",
                            document_ref="doc_bob",
                            session_handle=bob_handle,
                        ),
                        run_id,
                        episode_id,
                        53 + step * 4,
                    )
                    verified = self.client.post(
                        f"/range/controls/{control_id}/verify",
                        json={
                            "unauthorized_job_id": unauthorized["job"]["job_id"],
                            "owner_job_id": owner["job"]["job_id"],
                        },
                    )
                    if verified.status_code == 200:
                        verification_status = verified.json().get("status", "inconclusive")
                    else:
                        verification_status = "inconclusive"
            eligible_prevention = not confirmed_attack
            step_prevention = verification_status == "verified" and eligible_prevention
            prevention_verified = prevention_verified or step_prevention
            if step_prevention:
                blue_components = {"independently_verified_prevention": 1.0}
            elif newly_confirmed_attack:
                blue_components = {"verified_attack_left_unprevented": -1.0}
            elif blue_decision.action == 1 and not control_applied:
                blue_components = {"control_rejected_or_failed": -0.1}
            else:
                blue_components = {"no_verified_outcome": 0.0}
            blue_reward = sum(blue_components.values())
            blue_return += blue_reward
            accepted_blue = blue_decision.action == 0 or control_applied
            executed_blue = blue_decision.action if accepted_blue else None
            blue_next = RangeObservation.blue(
                step=step + 1,
                budget=self.red_steps,
                requests=request_count,
                failed_requests=failed_count,
                service_healthy=service_healthy,
                verified_attack=confirmed_attack,
                verified_prevention=prevention_verified,
                controls_applied=int(ownership_hardened),
            )
            blue_next_value = self.blue.value(blue_next)
            if accepted_blue and self.training:
                self.blue.record(
                    RangeTransition(
                        blue_obs.array(),
                        blue_mask,
                        blue_decision,
                        blue_reward,
                        blue_next_value,
                        False,
                        is_last,
                    )
                )
            self._persist_transition(
                run_id=run_id,
                episode_id=episode_id,
                seed=episode_seed,
                role="blue",
                observation=blue_obs,
                decision=blue_decision,
                mask=blue_mask,
                accepted=accepted_blue,
                executed_action=executed_blue,
                reward=blue_reward,
                components=blue_components,
                next_observation=blue_next,
                next_value=blue_next_value,
                terminated=False,
                truncated=is_last,
            )
        red_update = (
            self.red.update()
            if self.training
            else {"updates": 0, "transitions": len(self.red.transitions)}
        )
        blue_update = (
            self.blue.update()
            if self.training
            else {"updates": 0, "transitions": len(self.blue.transitions)}
        )
        if self.training:
            for role, update in (("red", red_update), ("blue", blue_update)):
                if update.get("updates", 0):
                    persisted = self.client.post(
                        "/range/training/updates",
                        json={
                            "run_id": run_id,
                            "episode_id": episode_id,
                            "seed": episode_seed,
                            "role": role,
                            "policy_version": update["updates"],
                            "transition_count": update["transitions"],
                            "metrics": {
                                key: update[key]
                                for key in (
                                    "loss",
                                    "policy_loss",
                                    "value_loss",
                                    "entropy",
                                    "clip_fraction",
                                    "gradient_norm",
                                    "optimizer_steps",
                                    "advantage_mean",
                                    "advantage_std",
                                    "advantage_min",
                                    "advantage_max",
                                )
                            },
                        },
                    )
                    persisted.raise_for_status()
        return {
            "episode": episode_number,
            "episode_id": episode_id,
            "seed": episode_seed,
            "run_id": run_id,
            "red_actions": red_actions,
            "blue_actions": blue_actions,
            "action_frequencies": {
                role: {
                    name: actions.count(index) / max(1, len(actions))
                    for index, name in enumerate(RED_ACTIONS if role == "red" else BLUE_ACTIONS)
                }
                for role, actions in (("red", red_actions), ("blue", blue_actions))
            },
            "mean_selected_action_probability": {
                role: float(np.mean(values)) if values else None
                for role, values in selected_probabilities.items()
            },
            "mean_policy_entropy": {
                role: float(np.mean(values)) if values else None
                for role, values in policy_entropies.items()
            },
            "verified_attack": confirmed_attack,
            "verified_prevention": prevention_verified,
            "red_return": red_return,
            "blue_return": blue_return,
            "episode_length": self.red_steps,
            "request_count": request_count,
            "legitimate_request_attempts": legitimate_request_attempts,
            "legitimate_request_successes": legitimate_request_successes,
            "false_blocked_legitimate_requests": false_blocked_legitimate_requests,
            "invalid_action_count": invalid_actions,
            "mean_http_latency_ms": float(np.mean(elapsed_ms)) if elapsed_ms else 0.0,
            "red_update": red_update,
            "blue_update": blue_update,
        }

    def run(self) -> list[dict[str, Any]]:
        health = self.client.get("/range/health")
        health.raise_for_status()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for episode in range(self.episodes):
            try:
                record = self.run_episode(episode)
                record["status"] = "completed"
                results.append(record)
                with (self.run_dir / "training.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                if self.training:
                    self.red.save(self.run_dir / "red.pt")
                    self.blue.save(self.run_dir / "blue.pt")
                summary = {
                    key: record[key]
                    for key in (
                        "verified_attack",
                        "verified_prevention",
                        "red_return",
                        "blue_return",
                        "episode_length",
                        "request_count",
                        "invalid_action_count",
                        "mean_http_latency_ms",
                    )
                }
                self.client.post(
                    "/range/training/episodes",
                    json={
                        "run_id": self.run_id,
                        "episode_id": record["episode_id"],
                        "seed": record["seed"],
                        "episode_status": "completed",
                        "run_status": "completed" if episode == self.episodes - 1 else "running",
                        "summary": summary,
                    },
                ).raise_for_status()
            except Exception as exc:
                failure = {
                    "episode": episode,
                    "seed": self.seed + episode,
                    "run_id": self.run_id,
                    "episode_id": self.current_episode_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                with (self.run_dir / "training.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(failure) + "\n")
                print(json.dumps(failure), flush=True)
                if self.current_episode_id:
                    with contextlib.suppress(Exception):
                        self.client.post(
                            "/range/training/episodes",
                            json={
                                "run_id": self.run_id,
                                "episode_id": self.current_episode_id,
                                "seed": self.seed + episode,
                                "episode_status": "failed",
                                "run_status": "failed",
                                "summary": {},
                            },
                        ).raise_for_status()
                raise
            print(json.dumps(record))
        return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--red-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default="runs/real_range_ppo")
    parser.add_argument(
        "--evaluate", action="store_true", help="run deterministic inference without PPO updates"
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue training from both role checkpoints"
    )
    parser.add_argument("--red-checkpoint")
    parser.add_argument("--blue-checkpoint")
    args = parser.parse_args()
    if args.episodes < 1 or not 1 <= args.red_steps <= 20:
        parser.error("episodes must be positive and red-steps must be between 1 and 20")
    if args.evaluate and not (args.red_checkpoint and args.blue_checkpoint):
        parser.error("--evaluate requires both --red-checkpoint and --blue-checkpoint")
    if args.resume and not (args.red_checkpoint and args.blue_checkpoint):
        parser.error("--resume requires both --red-checkpoint and --blue-checkpoint")
    if args.evaluate and args.resume:
        parser.error("--evaluate and --resume are mutually exclusive")
    RangeTrainingRunner(
        args.api,
        args.run_dir,
        seed=args.seed,
        episodes=args.episodes,
        red_steps=args.red_steps,
        training=not args.evaluate,
        red_checkpoint=args.red_checkpoint,
        blue_checkpoint=args.blue_checkpoint,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
