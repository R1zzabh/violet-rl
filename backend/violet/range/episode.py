"""Bounded two-stage Red/Blue coordinator for the instrumented Violet Ledger target."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast
from uuid import UUID, uuid4, uuid5

import httpx
from pydantic import TypeAdapter

from violet.range.contracts import (
    AgentDecision,
    AuthTestAction,
    BlueAction,
    CheckDocumentAccessAction,
    PolicySample,
    RangeAction,
    RangeObservation,
    WaitAction,
)

Policy = Callable[[RangeObservation], dict[str, Any] | Awaitable[dict[str, Any]]]


class RangeEpisodeRunner:
    """Execute sequential Red request then Blue telemetry/control rounds via the API."""

    def __init__(
        self,
        api_base_url: str,
        run_id: UUID,
        episode_id: UUID,
        *,
        red_policy: Policy,
        blue_policy: Policy,
        red_policy_id: str = "red-baseline-v1",
        blue_policy_id: str = "blue-baseline-v1",
        poll_interval: float = 0.1,
        request_timeout: float = 5.0,
        stop_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.run_id = run_id
        self.episode_id = episode_id
        self.red_policy = red_policy
        self.blue_policy = blue_policy
        self.red_policy_id = red_policy_id
        self.blue_policy_id = blue_policy_id
        self.poll_interval = min(max(poll_interval, 0.01), 1.0)
        self.request_timeout = min(max(request_timeout, 0.1), 5.0)
        self.stop_check = stop_check
        self._deadline_at = 0.0

    async def _choose(self, policy: Policy, observation: RangeObservation) -> dict[str, Any]:
        selected = policy(observation)
        if inspect.isawaitable(selected):
            selected = await selected
        if not isinstance(selected, dict):
            raise TypeError("range policy must return a typed action dictionary")
        return selected

    @staticmethod
    def _take_policy_sample(policy: Policy) -> dict[str, Any] | None:
        consume = getattr(policy, "consume_sample", None)
        return consume() if callable(consume) else None

    @staticmethod
    def _policy_sample(sample: dict[str, Any] | None) -> PolicySample | None:
        if sample is None:
            return None
        return PolicySample(
            encoding_schema_id=sample["encoding_schema_id"],
            encoded_observation=tuple(sample["observation"]),
            action_distribution=tuple(sample["action_distribution"]),
            selected_action_index=int(sample["action"]),
            selected_action_probability=float(sample["selected_action_probability"]),
            old_log_probability=float(sample["log_prob"]),
            value_estimate=float(sample["value"]),
            entropy=float(sample["entropy"]),
            policy_version=int(sample["policy_version"]),
            mode=sample["mode"],
        )

    async def _persist_transition(
        self,
        client: httpx.AsyncClient,
        trace_entry: dict[str, Any],
        role: str,
        next_observation: RangeObservation,
        *,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        sample = trace_entry.get(f"{role}_sample")
        if not isinstance(sample, dict):
            return
        policy = self.red_policy if role == "red" else self.blue_policy
        value_fn = getattr(policy, "estimate_value", None)
        next_value = float(value_fn(next_observation)) if callable(value_fn) else 0.0
        response = await client.post(
            f"/range/runs/{self.run_id}/episodes/{self.episode_id}/transitions",
            json={
                "decision_id": trace_entry[f"{role}_decision_id"],
                "next_observation": next_observation.model_dump(mode="json"),
                "next_value": next_value,
                "terminated": terminated,
                "truncated": truncated,
                "mode": sample["mode"],
            },
        )
        response.raise_for_status()
        trace_entry[f"{role}_transition"] = response.json()

    async def _wait_for_job(self, client: httpx.AsyncClient, job_id: str) -> dict[str, Any] | None:
        while time.monotonic() < self._deadline_at:
            response = await client.get(f"/range/jobs/{job_id}")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("range job response is not an object")
            if "result" in payload:
                return cast("dict[str, Any]", payload)
            await asyncio.sleep(
                min(self.poll_interval, max(0.0, self._deadline_at - time.monotonic()))
            )
        return None

    async def _wait_for_control(
        self, client: httpx.AsyncClient, command_id: str
    ) -> dict[str, Any] | None:
        while time.monotonic() < self._deadline_at:
            response = await client.get(f"/range/controls/{command_id}")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("range control response is not an object")
            if payload.get("status") in {"applied", "failed", "rejected"}:
                return cast("dict[str, Any]", payload)
            await asyncio.sleep(
                min(self.poll_interval, max(0.0, self._deadline_at - time.monotonic()))
            )
        return None

    async def _finish_episode(
        self, client: httpx.AsyncClient, status: str, reason: str | None = None
    ) -> None:
        response = await client.post(
            f"/range/runs/{self.run_id}/episodes/{self.episode_id}/finish",
            json={"status": status, "reason": reason},
        )
        response.raise_for_status()

    async def run(self, *, seed: int, max_steps: int = 12) -> dict[str, Any]:
        if not 1 <= max_steps <= 40:
            raise ValueError("max_steps must be between 1 and the 40 request episode budget")
        self._deadline_at = time.monotonic() + 300.0
        async with httpx.AsyncClient(
            base_url=self.api_base_url, timeout=self.request_timeout
        ) as client:
            reset = await client.post(
                "/range/reset",
                json={
                    "seed": seed,
                    "run_id": str(self.run_id),
                    "episode_id": str(self.episode_id),
                    "command_id": str(uuid5(self.episode_id, "violet-range-reset")),
                },
            )
            reset.raise_for_status()
            reset_id = reset.json()["command_id"]
            reset_ack = await self._wait_for_control(client, reset_id)
            if reset_ack is None:
                await self._finish_episode(client, "incomplete", "reset_ack_timeout")
                return {"status": "incomplete", "reason": "reset_ack_timeout", "trace": []}
            if reset_ack.get("status") != "applied":
                raise RuntimeError("target reset was not applied")

            trace: list[dict[str, Any]] = []
            session_aliases: dict[str, str] = {}
            for step_id in range(max_steps):
                if self.stop_check is not None and await self.stop_check():
                    await self._finish_episode(client, "incomplete", "run_stop_requested")
                    return {"status": "stopped", "reason": "run_stop_requested", "trace": trace}
                red_obs_response = await client.get(
                    f"/range/runs/{self.run_id}/observations/red",
                    params={"episode_id": str(self.episode_id), "step_id": step_id},
                )
                red_obs_response.raise_for_status()
                red_observation = RangeObservation.model_validate(red_obs_response.json())
                if step_id > 0:
                    await self._persist_transition(client, trace[-1], "red", red_observation)
                if red_observation.target_status == "unavailable":
                    await self._finish_episode(client, "incomplete", "target_unavailable")
                    return {
                        "status": "incomplete",
                        "reason": "target_unavailable",
                        "trace": trace,
                    }
                red_action: RangeAction = TypeAdapter(RangeAction).validate_python(
                    await self._choose(self.red_policy, red_observation)
                )
                red_sample = self._take_policy_sample(self.red_policy)
                red_decision_id = uuid4()
                red_decision = AgentDecision(
                    decision_id=red_decision_id,
                    run_id=self.run_id,
                    episode_id=self.episode_id,
                    step_id=step_id,
                    role="red",
                    policy_id=self.red_policy_id,
                    observation_schema_id=red_observation.schema_id,
                    legal_action_mask=red_observation.legal_action_mask or (True,),
                    selected_action=red_action.model_dump(mode="json"),
                    inference_ms=float(red_sample.get("inference_ms", 0.0)) if red_sample else 0.0,
                    policy_sample=self._policy_sample(red_sample),
                )
                saved = await client.post(
                    "/range/decisions",
                    json={
                        "decision": red_decision.model_dump(mode="json"),
                        "observation": red_observation.model_dump(mode="json"),
                    },
                )
                saved.raise_for_status()
                job = None
                if not isinstance(red_action, WaitAction):
                    submitted = await client.post(
                        "/range/jobs",
                        json={
                            "action": red_action.model_dump(mode="json"),
                            "run_id": str(self.run_id),
                            "episode_id": str(self.episode_id),
                            "step_id": step_id,
                            "decision_id": str(red_decision_id),
                            "idempotency_key": str(uuid5(red_decision_id, "violet-range-job")),
                        },
                    )
                    submitted.raise_for_status()
                    job = await self._wait_for_job(client, submitted.json()["job_id"])
                    if job is None:
                        await self._finish_episode(
                            client, "incomplete", "episode_deadline_exceeded"
                        )
                        return {
                            "status": "incomplete",
                            "reason": "episode_deadline_exceeded",
                            "trace": trace,
                        }
                    result = job["result"]
                    if result.get("principal_ref") and result.get("session_alias"):
                        session_aliases[result["principal_ref"]] = result["session_alias"]
                else:
                    result = {"status": "skipped", "target_event_id": None}
                trace.append(
                    {
                        "step_id": step_id,
                        "red": job
                        or {
                            "status": "decision_recorded",
                            "decision": red_decision.model_dump(mode="json"),
                        },
                        "red_observation": red_observation.model_dump(mode="json"),
                        "red_sample": red_sample,
                    }
                )
                reward = await client.post(
                    f"/range/runs/{self.run_id}/episodes/{self.episode_id}/rewards",
                    json={
                        "role": "red",
                        "step_id": step_id,
                        "job_id": job["job"]["job_id"] if job else None,
                        "decision_id": str(red_decision_id) if job is None else None,
                    },
                )
                reward.raise_for_status()
                trace[-1]["red_reward"] = reward.json()
                if result.get("status") == "outcome_unknown":
                    await self._finish_episode(client, "incomplete", "outcome_unknown")
                    return {"status": "incomplete", "reason": "outcome_unknown", "trace": trace}

                evaluated = await client.post(
                    f"/range/runs/{self.run_id}/episodes/{self.episode_id}/detections"
                )
                evaluated.raise_for_status()
                blue_obs_response = await client.get(
                    f"/range/runs/{self.run_id}/observations/blue",
                    params={"episode_id": str(self.episode_id), "step_id": step_id},
                )
                blue_obs_response.raise_for_status()
                blue_observation = RangeObservation.model_validate(blue_obs_response.json())
                if step_id > 0:
                    await self._persist_transition(client, trace[-1], "blue", blue_observation)
                blue_action = BlueAction.model_validate(
                    await self._choose(self.blue_policy, blue_observation)
                )
                blue_sample = self._take_policy_sample(self.blue_policy)
                blue_decision_id = uuid4()
                blue_decision = AgentDecision(
                    decision_id=blue_decision_id,
                    run_id=self.run_id,
                    episode_id=self.episode_id,
                    step_id=step_id,
                    role="blue",
                    policy_id=self.blue_policy_id,
                    observation_schema_id=blue_observation.schema_id,
                    legal_action_mask=blue_observation.legal_action_mask or (True,),
                    selected_action=blue_action.model_dump(mode="json"),
                    inference_ms=float(blue_sample.get("inference_ms", 0.0))
                    if blue_sample
                    else 0.0,
                    policy_sample=self._policy_sample(blue_sample),
                )
                saved = await client.post(
                    "/range/decisions",
                    json={
                        "decision": blue_decision.model_dump(mode="json"),
                        "observation": blue_observation.model_dump(mode="json"),
                    },
                )
                saved.raise_for_status()
                command_id = None
                ack = None
                control_verification = None
                if blue_action.operation != "MONITOR":
                    control = await client.post(
                        "/range/controls",
                        json={
                            "action": blue_action.operation,
                            "principal_ref": blue_action.principal_ref,
                            "run_id": str(self.run_id),
                            "episode_id": str(self.episode_id),
                            "decision_id": str(blue_decision_id),
                            "command_id": str(uuid5(blue_decision_id, "violet-range-control")),
                        },
                    )
                    control.raise_for_status()
                    command_id = control.json()["command_id"]
                    ack = await self._wait_for_control(client, command_id)
                    checked = None
                    if ack is None:
                        await self._finish_episode(client, "incomplete", "control_ack_timeout")
                        return {
                            "status": "incomplete",
                            "reason": "control_ack_timeout",
                            "trace": trace,
                        }
                    if (
                        blue_action.operation == "HARDEN_DOCUMENT_OWNERSHIP"
                        and ack.get("status") == "applied"
                        and session_aliases.get("alice")
                    ):
                        bob_login = await client.post(
                            "/range/jobs",
                            json={
                                "action": AuthTestAction(
                                    operation="AUTH_TEST", credential_ref="synthetic_bob"
                                ).model_dump(mode="json"),
                                "purpose": "evaluator",
                                "run_id": str(self.run_id),
                                "episode_id": str(self.episode_id),
                                "step_id": step_id,
                            },
                        )
                        bob_login.raise_for_status()
                        bob_login_job = await self._wait_for_job(client, bob_login.json()["job_id"])
                        if bob_login_job is None:
                            await self._finish_episode(
                                client, "incomplete", "control_verifier_probe_timeout"
                            )
                            return {
                                "status": "incomplete",
                                "reason": "control_verifier_probe_timeout",
                                "trace": trace,
                            }
                        bob_alias = (
                            bob_login_job["result"].get("session_alias")
                            if bob_login_job["result"].get("status") == "completed"
                            else None
                        )
                        if bob_login_job["result"].get("status") != "completed" or not bob_alias:
                            await self._finish_episode(
                                client, "incomplete", "control_verifier_probe_failed"
                            )
                            return {
                                "status": "incomplete",
                                "reason": "control_verifier_probe_failed",
                                "trace": trace,
                            }
                        if bob_alias:
                            alice_probe = await client.post(
                                "/range/jobs",
                                json={
                                    "action": CheckDocumentAccessAction(
                                        operation="CHECK_DOCUMENT_ACCESS",
                                        document_ref="doc_bob",
                                        session_alias=session_aliases["alice"],
                                    ).model_dump(mode="json"),
                                    "purpose": "evaluator",
                                    "run_id": str(self.run_id),
                                    "episode_id": str(self.episode_id),
                                    "step_id": step_id,
                                },
                            )
                            alice_probe.raise_for_status()
                            alice_job = await self._wait_for_job(
                                client, alice_probe.json()["job_id"]
                            )
                            bob_probe = await client.post(
                                "/range/jobs",
                                json={
                                    "action": CheckDocumentAccessAction(
                                        operation="CHECK_DOCUMENT_ACCESS",
                                        document_ref="doc_bob",
                                        session_alias=bob_alias,
                                    ).model_dump(mode="json"),
                                    "purpose": "evaluator",
                                    "run_id": str(self.run_id),
                                    "episode_id": str(self.episode_id),
                                    "step_id": step_id,
                                },
                            )
                            bob_probe.raise_for_status()
                            bob_job = await self._wait_for_job(client, bob_probe.json()["job_id"])
                            alice_owner_probe = await client.post(
                                "/range/jobs",
                                json={
                                    "action": CheckDocumentAccessAction(
                                        operation="CHECK_DOCUMENT_ACCESS",
                                        document_ref="doc_alice",
                                        session_alias=session_aliases["alice"],
                                    ).model_dump(mode="json"),
                                    "purpose": "evaluator",
                                    "run_id": str(self.run_id),
                                    "episode_id": str(self.episode_id),
                                    "step_id": step_id,
                                },
                            )
                            alice_owner_probe.raise_for_status()
                            alice_owner_job = await self._wait_for_job(
                                client, alice_owner_probe.json()["job_id"]
                            )
                            if alice_job is None or bob_job is None or alice_owner_job is None:
                                await self._finish_episode(
                                    client, "incomplete", "control_verifier_probe_timeout"
                                )
                                return {
                                    "status": "incomplete",
                                    "reason": "control_verifier_probe_timeout",
                                    "trace": trace,
                                }
                            checked = await client.post(
                                f"/range/controls/{command_id}/verify",
                                json={
                                    "unauthorized_job_id": alice_job["job"]["job_id"],
                                    "alice_owner_job_id": alice_owner_job["job"]["job_id"],
                                    "owner_job_id": bob_job["job"]["job_id"],
                                },
                            )
                            checked.raise_for_status()
                    control_verification = checked.json() if checked is not None else None
                detection_payload = evaluated.json()
                current_alert = next(
                    (
                        alert
                        for alert in detection_payload.get("alerts", [])
                        if alert.get("source_target_event_id") == result.get("target_event_id")
                    ),
                    None,
                )
                blue_reward_response = await client.post(
                    f"/range/runs/{self.run_id}/episodes/{self.episode_id}/rewards",
                    json={
                        "role": "blue",
                        "step_id": step_id,
                        "red_decision_id": str(red_decision_id),
                        "decision_id": str(blue_decision_id),
                        "source_target_event_id": current_alert["source_target_event_id"]
                        if current_alert
                        else None,
                        "control_command_id": command_id if control_verification else None,
                    },
                )
                blue_reward_response.raise_for_status()
                trace[-1]["blue_observation"] = blue_observation.model_dump(mode="json")
                trace[-1]["blue_decision_id"] = str(blue_decision_id)
                trace[-1]["blue_sample"] = blue_sample
                trace[-1]["blue_reward"] = blue_reward_response.json()
                trace[-1]["blue"] = {
                    "decision": blue_decision.model_dump(mode="json"),
                    "detections": detection_payload.get("alerts", []),
                    "control": ack if ack is not None else None,
                    "control_verification": control_verification,
                    "reward": blue_reward_response.json(),
                }
            terminal_observations: dict[str, RangeObservation] = {}
            for role in ("red", "blue"):
                terminal_response = await client.get(
                    f"/range/runs/{self.run_id}/observations/{role}",
                    params={"episode_id": str(self.episode_id), "step_id": len(trace)},
                )
                terminal_response.raise_for_status()
                terminal_observations[role] = RangeObservation.model_validate(
                    terminal_response.json()
                )
                if trace:
                    await self._persist_transition(
                        client,
                        trace[-1],
                        role,
                        terminal_observations[role],
                        truncated=True,
                    )
            await self._finish_episode(client, "completed")
            if trace:
                trace[-1]["terminal_observations"] = {
                    role: observation.model_dump(mode="json")
                    for role, observation in terminal_observations.items()
                }
            return {"status": "completed", "trace": trace}
