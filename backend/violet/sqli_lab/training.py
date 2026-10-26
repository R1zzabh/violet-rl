"""Episode runner, persisted replay, and PPO updates for the dedicated SQLi lab."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

import numpy as np
import torch

from violet.sqli_lab.contracts import (
    ACTION_SCHEMA,
    BLUE_ACTIONS,
    FEATURE_SCHEMA,
    OBSERVATION_SCHEMA,
    RED_COMPONENTS,
    REWARD_SCHEMA,
    SCENARIO_ID,
    TARGET_VERSION,
    SqliObservation,
    SqliRequestAction,
)
from violet.sqli_lab.learning import FEATURES, SqliPPOPolicy, SqliSample

if TYPE_CHECKING:
    import httpx

ARTIFACT_ROOT = Path(os.getenv("SQLI_ARTIFACT_ROOT", "/app/runs"))
VERIFIER_ACTION = SqliRequestAction(
    field="handle",
    base_term="rhea",
    close_quote=True,
    connector="OR",
    predicate="tautology",
    comment=True,
)
LEGITIMATE_ACTION = SqliRequestAction(
    field="handle",
    base_term="rhea",
    close_quote=False,
    connector="none",
    predicate="none",
    comment=False,
)
CHECKPOINT_CONTRACT = {
    "format": "sqli.joint_ppo.v1",
    "scenario_id": SCENARIO_ID,
    "target_version": TARGET_VERSION,
    "observation_schema_id": OBSERVATION_SCHEMA,
    "feature_schema_id": FEATURE_SCHEMA,
    "action_schema_id": ACTION_SCHEMA,
    "reward_schema_id": REWARD_SCHEMA,
    "red_component_catalog": {key: list(value) for key, value in RED_COMPONENTS.items()},
    "blue_action_catalog": list(BLUE_ACTIONS),
}


def _validate_checkpoint_contract(payload: dict[str, Any]) -> None:
    if any(payload.get(key) != value for key, value in CHECKPOINT_CONTRACT.items()):
        raise ValueError("SQLi checkpoint contract is incompatible with this scenario")


def _action_is_injection(action: SqliRequestAction) -> bool:
    return bool(action.close_quote and action.connector != "none" and action.comment)


def _row_list(response: dict[str, Any]) -> list[dict[str, Any]]:
    rows = response.get("rows", [])
    return [item for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unauthorized_count(response: dict[str, Any], actor_ref: str) -> int:
    return sum(str(row.get("owner_ref")) != actor_ref for row in _row_list(response))


def _normal(value: float, denominator: float = 1.0) -> float:
    return float(np.clip(value / max(denominator, 1e-8), 0.0, 1.0))


class SqliEpisodeRunner:
    """Run one frozen Red/Blue policy pair and record every instructional event."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        run_id: UUID,
        episode_id: UUID,
        *,
        seed: int,
        max_steps: int,
        red_policy: SqliPPOPolicy,
        blue_policy: SqliPPOPolicy | None,
        mode: str,
        comparator: bool = False,
        stop_check: Any = None,
    ) -> None:
        self.client = client
        self.run_id = run_id
        self.episode_id = episode_id
        self.seed = seed
        self.max_steps = max_steps
        self.red_policy = red_policy
        self.blue_policy = blue_policy
        self.mode = mode
        self.comparator = comparator
        self.stop_check = stop_check
        self.telemetry_history: list[dict[str, Any]] = []
        self.active_controls: dict[str, Any] = {
            "parameterized": False,
            "rate_limit_guest": False,
            "isolate_guest_remaining": 0,
        }
        self.guest_requests = 0
        self.http_failures = 0
        self.injection_signals = 0
        self.last_red_result: dict[str, Any] = {}
        self.last_red_action: SqliRequestAction | None = None

    async def _event(self, event_type: str, payload: dict[str, Any]) -> None:
        response = await self.client.post(
            "/range/sqli/events",
            json={
                "event_id": str(uuid4()),
                "run_id": str(self.run_id),
                "episode_id": str(self.episode_id),
                "seed": self.seed,
                "event_type": event_type,
                "payload": payload,
            },
        )
        response.raise_for_status()

    async def _control(self, action: str, decision_id: UUID | None) -> dict[str, Any]:
        response = await self.client.post(
            "/range/sqli/controls",
            json={
                "run_id": str(self.run_id),
                "episode_id": str(self.episode_id),
                "decision_id": str(decision_id) if decision_id else None,
                "action": action,
            },
        )
        response.raise_for_status()
        command_id = str(response.json()["command_id"])
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            poll = await self.client.get(f"/range/sqli/controls/{command_id}")
            poll.raise_for_status()
            payload = cast("dict[str, Any]", poll.json())
            if payload.get("status") in {"applied", "failed", "rejected"}:
                await self._event(
                    "sqli_control_acknowledged",
                    {"command_id": command_id, "action": action, "acknowledgement": payload},
                )
                if payload.get("status") == "applied":
                    self.active_controls = dict(payload.get("controls") or {})
                return payload
            await asyncio.sleep(0.08)
        return {"status": "outcome_unknown", "command_id": command_id}

    async def _job(
        self,
        action: SqliRequestAction,
        decision_id: UUID,
        step_id: int,
        purpose: str,
    ) -> dict[str, Any]:
        response = await self.client.post(
            "/range/sqli/jobs",
            json={
                "run_id": str(self.run_id),
                "episode_id": str(self.episode_id),
                "decision_id": str(decision_id),
                "step_id": step_id,
                "action": action.model_dump(mode="json"),
                "purpose": purpose,
            },
        )
        response.raise_for_status()
        job_id = str(response.json()["job_id"])
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            poll = await self.client.get(f"/range/sqli/jobs/{job_id}")
            poll.raise_for_status()
            payload = cast("dict[str, Any]", poll.json())
            if payload.get("status") != "queued":
                return payload
            await asyncio.sleep(0.08)
        return {
            "job_id": job_id,
            "status": "outcome_unknown",
            "purpose": purpose,
            "request": {"field": action.field, "query_value": action.compose_value()},
            "response": {},
            "telemetry": {"evidence_status": "missing"},
        }

    def _red_observation(self, step_id: int) -> SqliObservation:
        result = self.last_red_result
        response = _object(result.get("response"))
        rows = _row_list(response)
        status = int(result.get("http_status") or 0)
        telemetry = _object(result.get("telemetry"))
        unauthorized = _unauthorized_count(response, "guest")
        features = (
            _normal(step_id, self.max_steps),
            _normal(self.max_steps - step_id, self.max_steps),
            _normal(status, 599),
            _normal(len(rows), 8),
            _normal(unauthorized, 8),
            float(status == 400),
            float(bool(telemetry.get("query_syntax_signal"))),
            float(_action_is_injection(self.last_red_action)) if self.last_red_action else 0.0,
        )
        head_count = sum(len(values) for values in RED_COMPONENTS.values())
        return SqliObservation(
            schema_id="red.sqli.range_observation.v1",
            run_id=self.run_id,
            episode_id=self.episode_id,
            step_id=step_id,
            features=features,
            telemetry={
                "last_http_status": status,
                "last_row_count": len(rows),
                "last_unauthorized_count": unauthorized,
                "last_query_syntax_signal": bool(telemetry.get("query_syntax_signal")),
            },
            remaining_requests=max(0, self.max_steps - step_id),
            legal_action_mask=tuple(True for _ in range(head_count)),
        )

    def _blue_observation(self, step_id: int) -> SqliObservation:
        available = bool(self.telemetry_history)
        features = (
            _normal(step_id, self.max_steps),
            _normal(self.max_steps - step_id, self.max_steps),
            _normal(self.guest_requests, self.max_steps + 3),
            _normal(self.http_failures, max(1, len(self.telemetry_history))),
            _normal(self.injection_signals, max(1, len(self.telemetry_history))),
            float(available),
            float(bool(self.active_controls.get("parameterized"))),
            float(bool(self.active_controls.get("rate_limit_guest"))),
        )
        # This telemetry deliberately excludes query text, record values, and owners.
        recent = self.telemetry_history[-8:]
        return SqliObservation(
            schema_id="blue.sqli.range_observation.v1",
            run_id=self.run_id,
            episode_id=self.episode_id,
            step_id=step_id,
            features=features,
            telemetry={
                "recent_requests": recent,
                "active_controls": self.active_controls,
                "guest_request_count": self.guest_requests,
                "http_failure_count": self.http_failures,
                "injection_signal_count": self.injection_signals,
            },
            remaining_requests=max(0, self.max_steps - step_id),
            legal_action_mask=tuple(True for _ in BLUE_ACTIONS),
        )

    async def _request_event(self, result: dict[str, Any], *, purpose: str, step_id: int) -> None:
        response = _object(result.get("response"))
        telemetry = _object(result.get("telemetry"))
        rows = _row_list(response)
        summary = {
            "purpose": purpose,
            "step_id": step_id,
            "job_id": result.get("job_id"),
            "decision_id": result.get("decision_id"),
            "request_id": result.get("request_id"),
            "status": result.get("status"),
            "actor_ref": result.get("actor_ref"),
            "http_status": result.get("http_status"),
            "target_event_id": result.get("target_event_id"),
            "request": result.get("request", {}),
            "response": response,
            "telemetry": telemetry,
            "duration_ms": result.get("duration_ms"),
            "evidence_status": telemetry.get("evidence_status"),
            "returned_row_refs": [str(row.get("record_ref", "")) for row in rows],
        }
        await self._event("sqli_target_response", summary)
        if telemetry.get("evidence_status") == "verified":
            self.telemetry_history.append(
                {
                    "http_status": int(result.get("http_status") or 0),
                    "query_syntax_signal": bool(telemetry.get("query_syntax_signal")),
                    "query_length": int(telemetry.get("query_length") or 0),
                    "row_count": int(telemetry.get("row_count") or 0),
                    "route_id": telemetry.get("route_id"),
                }
            )
            self.guest_requests += int(result.get("actor_ref") == "guest")
            self.http_failures += int(int(result.get("http_status") or 0) >= 400)
            self.injection_signals += int(bool(telemetry.get("query_syntax_signal")))

    async def run(self) -> dict[str, Any]:
        await self._event(
            "sqli_episode_started",
            {
                "scenario_id": SCENARIO_ID,
                "target_version": TARGET_VERSION,
                "seed": self.seed,
                "max_steps": self.max_steps,
                "observation_schema_id": OBSERVATION_SCHEMA,
                "feature_schema_id": FEATURE_SCHEMA,
                "action_schema_id": ACTION_SCHEMA,
                "reward_schema_id": REWARD_SCHEMA,
                "target_fixture_set": "synthetic_records.v1",
                "comparator": "no_defense_control" if self.comparator else None,
            },
        )
        reset = await self._control("RESET", None)
        if reset.get("status") != "applied":
            return {"status": "incomplete", "reason": "target_reset_unknown", "trace": []}

        trace: list[dict[str, Any]] = []
        terminal_status = "completed"
        reason: str | None = None
        for step_id in range(self.max_steps):
            if self.stop_check is not None and await self.stop_check():
                terminal_status, reason = "incomplete", "run_stop_requested"
                break
            red_observation = self._red_observation(step_id)
            red_obs_array = np.asarray(red_observation.features, dtype=np.float32)
            red_action_dict, red_sample = self.red_policy.act(
                red_obs_array, stochastic=self.mode == "train"
            )
            red_action = SqliRequestAction.model_validate(red_action_dict)
            red_decision_id = uuid4()
            await self._event(
                "sqli_red_decision",
                {
                    "decision_id": str(red_decision_id),
                    "policy_id": f"red-ppo-{self.run_id}",
                    "policy_role": "red",
                    "observation": red_observation.model_dump(mode="json"),
                    "feature_names": list(FEATURES["red"]),
                    "components": red_action.components(),
                    "action": red_action.model_dump(mode="json"),
                    "selected_action_indices": list(red_sample.action_indices),
                    "component_distributions": [
                        list(item) for item in red_sample.component_probabilities
                    ],
                    "selected_component_probabilities": list(red_sample.selected_probabilities),
                    "old_log_probability": red_sample.old_log_probability,
                    "value_estimate": red_sample.value_estimate,
                    "entropy": red_sample.entropy,
                    "policy_version": red_sample.policy_version,
                    "inference_mode": self.mode,
                    "action_schema_id": ACTION_SCHEMA,
                    "inference_ms": red_sample.inference_ms,
                },
            )
            red_result = await self._job(red_action, red_decision_id, step_id, "policy")
            await self._request_event(red_result, purpose="policy", step_id=step_id)
            red_response = _object(red_result.get("response"))
            unauthorized_rows = _unauthorized_count(red_response, "guest")
            red_accepted = (
                red_result.get("status") == "completed"
                and _object(red_result.get("telemetry")).get("evidence_status") == "verified"
            )
            red_components = {
                "confirmed_unauthorized_read": float(unauthorized_rows > 0),
                "returned_row_progress": 0.02 * float(bool(_row_list(red_response))),
                "request_cost": -0.02,
            }
            red_reward = sum(red_components.values()) if red_accepted else 0.0

            blue_observation = self._blue_observation(step_id)
            blue_obs_array = np.asarray(blue_observation.features, dtype=np.float32)
            blue_decision_id = uuid4()
            if self.comparator:
                blue_action = "MONITOR"
                blue_sample: SqliSample | None = None
                blue_probabilities: list[list[float]] = [[1.0, 0.0, 0.0, 0.0]]
                blue_value = 0.0
                blue_entropy = 0.0
                blue_log_prob = 0.0
                blue_indices: list[int] = [0]
                blue_policy_id = "evaluation_comparator:no_defense.v1"
            else:
                assert self.blue_policy is not None
                blue_action_dict, blue_sample = self.blue_policy.act(
                    blue_obs_array, stochastic=self.mode == "train"
                )
                blue_action = str(blue_action_dict["operation"])
                blue_probabilities = [list(item) for item in blue_sample.component_probabilities]
                blue_value = blue_sample.value_estimate
                blue_entropy = blue_sample.entropy
                blue_log_prob = blue_sample.old_log_probability
                blue_indices = list(blue_sample.action_indices)
                blue_policy_id = f"blue-ppo-{self.run_id}"
            await self._event(
                "sqli_blue_decision",
                {
                    "phase": "policy_decision",
                    "decision_id": str(blue_decision_id),
                    "policy_id": blue_policy_id,
                    "policy_role": "blue",
                    "observation": blue_observation.model_dump(mode="json"),
                    "feature_names": list(FEATURES["blue"]),
                    "selected_action": {"operation": blue_action},
                    "selected_action_indices": blue_indices,
                    "component_distributions": blue_probabilities,
                    "selected_component_probabilities": list(blue_sample.selected_probabilities)
                    if blue_sample
                    else [1.0],
                    "old_log_probability": blue_log_prob,
                    "value_estimate": blue_value,
                    "entropy": blue_entropy,
                    "policy_version": blue_sample.policy_version if blue_sample else 0,
                    "inference_mode": "comparator" if self.comparator else self.mode,
                    "action_schema_id": "sqli.blue_action_catalog.v1",
                    "inference_ms": blue_sample.inference_ms if blue_sample else 0.0,
                },
            )

            control_ack: dict[str, Any] = {"status": "applied", "controls": self.active_controls}
            if not self.comparator and blue_action != "MONITOR":
                control_ack = await self._control(blue_action, blue_decision_id)
            verifier_attack = await self._job(
                VERIFIER_ACTION, blue_decision_id, step_id, "verifier_attack"
            )
            await self._request_event(verifier_attack, purpose="verifier_attack", step_id=step_id)
            verifier_legit = await self._job(
                LEGITIMATE_ACTION, blue_decision_id, step_id, "verifier_legitimate"
            )
            await self._request_event(
                verifier_legit, purpose="verifier_legitimate", step_id=step_id
            )
            attack_response = _object(verifier_attack.get("response"))
            legit_response = _object(verifier_legit.get("response"))
            attack_rows = _unauthorized_count(attack_response, "guest")
            legitimate_served = (
                verifier_legit.get("status") == "completed"
                and verifier_legit.get("http_status") == 200
                and any(
                    row.get("record_ref") == "rec-001" and row.get("owner_ref") == "researcher"
                    for row in _row_list(legit_response)
                )
            )
            attack_verified = (
                verifier_attack.get("status") == "completed"
                and _object(verifier_attack.get("telemetry")).get("evidence_status") == "verified"
            )
            legit_verified = (
                verifier_legit.get("status") == "completed"
                and _object(verifier_legit.get("telemetry")).get("evidence_status") == "verified"
            )
            prevented = attack_verified and attack_rows == 0
            false_positive = legit_verified and not legitimate_served
            verification = {
                "schema_id": "sqli.control_verification.v1",
                "attack_probe_job_id": verifier_attack.get("job_id"),
                "legitimate_probe_job_id": verifier_legit.get("job_id"),
                "attack_probe_status": verifier_attack.get("http_status"),
                "unauthorized_rows_returned": attack_rows,
                "legitimate_query_status": verifier_legit.get("http_status"),
                "legitimate_query_served": legitimate_served,
                "verified_prevention": prevented,
                "false_positive": false_positive,
                "evidence_verified": attack_verified and legit_verified,
            }
            await self._event("sqli_verification", verification)
            blue_components = {
                "verified_prevention": 1.0 if prevented else -1.0,
                "legitimate_query_availability": 0.25 if legitimate_served else -1.0,
                "false_positive_penalty": -1.0 if false_positive else 0.0,
                "unknown_verification_penalty": -0.25
                if not (attack_verified and legit_verified)
                else 0.0,
            }
            blue_accepted = (
                (self.comparator or control_ack.get("status") == "applied")
                and attack_verified
                and legit_verified
            )
            blue_reward = (
                sum(blue_components.values()) if blue_accepted and not self.comparator else 0.0
            )
            reward_payload = {
                "schema_id": REWARD_SCHEMA,
                "role": "red",
                "step_id": step_id,
                "components": red_components,
                "total": red_reward,
                "verified_unauthorized_read": unauthorized_rows > 0,
                "accepted": red_accepted,
            }
            blue_reward_payload = {
                "schema_id": REWARD_SCHEMA,
                "role": "blue",
                "step_id": step_id,
                "components": blue_components,
                "total": blue_reward,
                "accepted": blue_accepted and not self.comparator,
                "verification": verification,
            }
            await self._event("sqli_reward", {"red": reward_payload, "blue": blue_reward_payload})

            item = {
                "step_id": step_id,
                "red_observation": red_observation,
                "red_observation_array": red_obs_array,
                "red_action": red_action,
                "red_sample": red_sample,
                "red_decision_id": red_decision_id,
                "red_result": red_result,
                "red_accepted": red_accepted,
                "red_reward": red_reward,
                "red_reward_components": red_components,
                "blue_observation": blue_observation,
                "blue_observation_array": blue_obs_array,
                "blue_action": blue_action,
                "blue_sample": blue_sample,
                "blue_decision_id": blue_decision_id,
                "blue_accepted": blue_accepted and not self.comparator,
                "blue_reward": blue_reward,
                "blue_reward_components": blue_components,
                "verification": verification,
            }
            trace.append(item)
            self.last_red_result = red_result
            self.last_red_action = red_action
            if not red_accepted or not attack_verified or not legit_verified:
                terminal_status, reason = "incomplete", "unknown_or_unverified_outcome"
                break

        if trace:
            final_red_obs = self._red_observation(len(trace))
            final_blue_obs = self._blue_observation(len(trace))
            trace[-1]["red_terminal_observation_array"] = np.asarray(
                final_red_obs.features, dtype=np.float32
            )
            trace[-1]["blue_terminal_observation_array"] = np.asarray(
                final_blue_obs.features, dtype=np.float32
            )
            for index, item in enumerate(trace):
                next_red = (
                    trace[index + 1]["red_observation_array"]
                    if index + 1 < len(trace)
                    else item["red_terminal_observation_array"]
                )
                next_blue = (
                    trace[index + 1]["blue_observation_array"]
                    if index + 1 < len(trace)
                    else item["blue_terminal_observation_array"]
                )
                next_red_value = self.red_policy.value(next_red)
                next_blue_value = (
                    self.blue_policy.value(next_blue) if self.blue_policy is not None else 0.0
                )
                next_is_gap = index + 1 < len(trace) and (
                    not trace[index + 1]["red_accepted"] or not trace[index + 1]["blue_accepted"]
                )
                last_step = index + 1 == len(trace)
                red_accept = bool(item["red_accepted"])
                blue_accept = bool(item["blue_accepted"])
                await self._event(
                    "sqli_transition",
                    {
                        "role": "red",
                        "step_id": item["step_id"],
                        "decision_id": str(item["red_decision_id"]),
                        "observation": item["red_observation"].model_dump(mode="json"),
                        "action": item["red_action"].model_dump(mode="json"),
                        "action_components": item["red_action"].components(),
                        "component_distributions": [
                            list(value) for value in item["red_sample"].component_probabilities
                        ],
                        "selected_component_probabilities": list(
                            item["red_sample"].selected_probabilities
                        ),
                        "old_log_probability": item["red_sample"].old_log_probability,
                        "value_estimate": item["red_sample"].value_estimate,
                        "policy_version": item["red_sample"].policy_version,
                        "execution_status": item["red_result"].get("status"),
                        "accepted": red_accept,
                        "learning_eligible": red_accept
                        and self.mode == "train"
                        and not self.comparator,
                        "reward": item["red_reward"],
                        "reward_components": item["red_reward_components"],
                        "next_observation": (
                            trace[index + 1]["red_observation"]
                            if index + 1 < len(trace)
                            else final_red_obs
                        ).model_dump(mode="json"),
                        "next_value": next_red_value,
                        "terminated": False,
                        "truncated": last_step or next_is_gap or terminal_status != "completed",
                        "feature_schema_id": FEATURE_SCHEMA,
                    },
                )
                if not self.comparator:
                    await self._event(
                        "sqli_transition",
                        {
                            "role": "blue",
                            "step_id": item["step_id"],
                            "decision_id": str(item["blue_decision_id"]),
                            "observation": item["blue_observation"].model_dump(mode="json"),
                            "action": {"operation": item["blue_action"]},
                            "component_distributions": [
                                list(value) for value in item["blue_sample"].component_probabilities
                            ],
                            "selected_component_probabilities": list(
                                item["blue_sample"].selected_probabilities
                            ),
                            "old_log_probability": item["blue_sample"].old_log_probability,
                            "value_estimate": item["blue_sample"].value_estimate,
                            "policy_version": item["blue_sample"].policy_version,
                            "execution_status": "applied"
                            if item["blue_accepted"]
                            else "rejected_or_unknown",
                            "accepted": blue_accept,
                            "learning_eligible": blue_accept and self.mode == "train",
                            "reward": item["blue_reward"],
                            "reward_components": item["blue_reward_components"],
                            "next_observation": (
                                trace[index + 1]["blue_observation"]
                                if index + 1 < len(trace)
                                else final_blue_obs
                            ).model_dump(mode="json"),
                            "next_value": next_blue_value,
                            "terminated": False,
                            "truncated": last_step or next_is_gap or terminal_status != "completed",
                            "feature_schema_id": FEATURE_SCHEMA,
                        },
                    )

        unauthorized = sum(
            int(_unauthorized_count(_object(item["red_result"].get("response")), "guest") > 0)
            for item in trace
            if item["red_accepted"]
        )
        prevention_checks = [
            item["verification"] for item in trace if item["verification"]["evidence_verified"]
        ]
        prevention_rate = (
            sum(bool(item["verified_prevention"]) for item in prevention_checks)
            / len(prevention_checks)
            if prevention_checks
            else None
        )
        availability_rate = (
            sum(bool(item["legitimate_query_served"]) for item in prevention_checks)
            / len(prevention_checks)
            if prevention_checks
            else None
        )
        false_positive_rate = (
            sum(bool(item["false_positive"]) for item in prevention_checks) / len(prevention_checks)
            if prevention_checks
            else None
        )
        summary = {
            "episode_id": str(self.episode_id),
            "seed": self.seed,
            "status": terminal_status,
            "reason": reason,
            "kind": "evaluation_comparator" if self.comparator else "candidate",
            "steps": len(trace),
            "red_return": round(sum(float(item["red_reward"]) for item in trace), 6),
            "blue_return": round(sum(float(item["blue_reward"]) for item in trace), 6),
            "unauthorized_read_rate": _normal(
                unauthorized, max(1, sum(bool(item["red_accepted"]) for item in trace))
            ),
            "verified_prevention_rate": prevention_rate,
            "legitimate_query_availability": availability_rate,
            "false_positive_rate": false_positive_rate,
            "verification_count": len(prevention_checks),
            "target_version": TARGET_VERSION,
            "action_schema_id": ACTION_SCHEMA,
            "reward_schema_id": REWARD_SCHEMA,
        }
        await self._event("sqli_episode_summary", summary)
        finish = await self.client.post(
            f"/range/runs/{self.run_id}/episodes/{self.episode_id}/finish",
            json={"status": terminal_status, "reason": reason},
        )
        finish.raise_for_status()
        return {"status": terminal_status, "reason": reason, "trace": trace, "summary": summary}


def _transition_batch(
    trace: list[dict[str, Any]], role: str, policy: SqliPPOPolicy
) -> list[dict[str, Any]]:
    accepted_key = f"{role}_accepted"
    reward_key = f"{role}_reward"
    obs_key = f"{role}_observation_array"
    sample_key = f"{role}_sample"
    result: list[dict[str, Any]] = []
    for index, item in enumerate(trace):
        sample = item.get(sample_key)
        if not item.get(accepted_key) or not isinstance(sample, SqliSample):
            if result:
                result[-1]["truncated"] = True
            continue
        next_obs = (
            trace[index + 1][obs_key]
            if index + 1 < len(trace)
            else item.get(f"{role}_terminal_observation_array", item[obs_key])
        )
        next_gap = index + 1 < len(trace) and not trace[index + 1].get(accepted_key)
        result.append(
            {
                "observation": item[obs_key],
                "action_indices": sample.action_indices,
                "old_log_probability": sample.old_log_probability,
                "value": sample.value_estimate,
                "reward": item[reward_key],
                "next_value": policy.value(next_obs),
                "terminated": False,
                "truncated": index + 1 == len(trace) or next_gap,
            }
        )
    if result:
        result[-1]["truncated"] = True
    return result


def _save_checkpoint(
    path: Path,
    *,
    training_id: str,
    seed: int,
    episode_count: int,
    red: SqliPPOPolicy,
    blue: SqliPPOPolicy,
    red_updates: int,
    blue_updates: int,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "sqli.joint_ppo.v1",
            "scenario_id": SCENARIO_ID,
            "target_version": TARGET_VERSION,
            "observation_schema_id": OBSERVATION_SCHEMA,
            "feature_schema_id": FEATURE_SCHEMA,
            "action_schema_id": ACTION_SCHEMA,
            "reward_schema_id": REWARD_SCHEMA,
            "red_component_catalog": {key: list(value) for key, value in RED_COMPONENTS.items()},
            "blue_action_catalog": list(BLUE_ACTIONS),
            "training_run_id": training_id,
            "seed": seed,
            "episode_count": episode_count,
            "policy_versions": {"red": red.policy_version, "blue": blue.policy_version},
            "update_counts": {"red": red_updates, "blue": blue_updates},
            "model_state_dicts": {"red": red.model.state_dict(), "blue": blue.model.state_dict()},
            "optimizer_state_dicts": {
                "red": red.optimizer.state_dict(),
                "blue": blue.optimizer.state_dict(),
            },
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _record_update(
    client: httpx.AsyncClient,
    training_id: str,
    role: str,
    update_index: int,
    global_step: int,
    metrics: dict[str, Any],
    digest: str | None,
) -> None:
    response = await client.post(
        f"/range/training-runs/{training_id}/updates",
        json={
            "worker_id": client.headers.get("x-violet-worker-id", "range-trainer-1"),
            "role": role,
            "status": "running",
            "result": {
                "update_index": update_index,
                "global_step": global_step,
                "metrics": metrics,
                "checkpoint_sha256": digest,
            },
        },
    )
    response.raise_for_status()


def _bootstrap_interval(
    candidate: list[float], baseline: list[float], seed: int
) -> tuple[float, float, float] | None:
    if not candidate or len(candidate) != len(baseline):
        return None
    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(differences), size=(2000, len(differences)))
    means = differences[samples].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


async def run_sqli_job(client: httpx.AsyncClient, job: dict[str, Any]) -> dict[str, Any]:
    """Train both PPO roles at each completed episode boundary or evaluate a checkpoint."""
    training_id = str(job["training_run_id"])
    training_uuid = UUID(training_id)
    config = job["config"]
    mode = str(job["mode"])
    if config.get("scenario_id") != SCENARIO_ID or config.get("target_version") != TARGET_VERSION:
        raise ValueError("training job does not select the versioned SQLi lab scenario")
    if (
        config.get("observation_schema_id") != OBSERVATION_SCHEMA
        or config.get("action_schema_id") != ACTION_SCHEMA
        or config.get("reward_schema_id") != REWARD_SCHEMA
    ):
        raise ValueError(
            "training job has an incompatible SQLi observation, action, or reward contract"
        )
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    red = SqliPPOPolicy("red", seed=seed, learning_rate=float(config["learning_rate"]))
    blue = SqliPPOPolicy("blue", seed=seed + 1, learning_rate=float(config["learning_rate"]))
    source_id = config.get("checkpoint_run_id")
    if source_id:
        source_response = await client.get(f"/range/training-runs/{source_id}")
        source_response.raise_for_status()
        source = source_response.json()
        if source.get("status") != "completed" or source.get("role") != "joint":
            raise ValueError("SQLi checkpoint source must be a completed joint PPO run")
        source_config = source.get("config") or {}
        if (
            source_config.get("scenario_id") != SCENARIO_ID
            or source_config.get("target_version") != TARGET_VERSION
        ):
            raise ValueError("SQLi checkpoint belongs to an incompatible scenario")
        path = ARTIFACT_ROOT / str(source_id) / "checkpoints" / "joint" / "final.pt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != (source.get("result") or {}).get("checkpoint_sha256"):
            raise ValueError("SQLi checkpoint digest does not match its persisted run record")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _validate_checkpoint_contract(payload)
        red.model.load_state_dict(payload["model_state_dicts"]["red"], strict=True)
        blue.model.load_state_dict(payload["model_state_dicts"]["blue"], strict=True)
        red.policy_version = int(payload["policy_versions"]["red"])
        blue.policy_version = int(payload["policy_versions"]["blue"])
        if mode == "train":
            red.optimizer.load_state_dict(payload["optimizer_state_dicts"]["red"])
            blue.optimizer.load_state_dict(payload["optimizer_state_dicts"]["blue"])

    run_dir = ARTIFACT_ROOT / training_id
    config_dir = run_dir / "config"
    metrics_dir = run_dir / "metrics"
    checkpoint_path = run_dir / "checkpoints" / "joint" / "final.pt"
    for directory in (config_dir, metrics_dir, checkpoint_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text(
        json.dumps(
            {
                "training_run_id": training_id,
                "scenario_id": SCENARIO_ID,
                "target_version": TARGET_VERSION,
                "observation_schema_id": OBSERVATION_SCHEMA,
                "feature_schema_id": FEATURE_SCHEMA,
                "action_schema_id": ACTION_SCHEMA,
                "reward_schema_id": REWARD_SCHEMA,
                "seed": seed,
                "mode": mode,
                "config": config,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    requested = int(config["episodes"])
    max_steps = int(config["max_steps"])
    research = config.get("run_kind") == "research"
    update_count = 0
    episode_summaries: list[dict[str, Any]] = []
    candidate_scores: list[float] = []
    comparator_scores: list[float] = []
    episode_index = 0
    while episode_index < requested:
        if await _training_stop_requested(client, training_id):
            break
        candidate_episode_id = uuid4()
        candidate = SqliEpisodeRunner(
            client,
            training_uuid,
            candidate_episode_id,
            seed=seed + episode_index,
            max_steps=max_steps,
            red_policy=red,
            blue_policy=blue,
            mode=mode,
            stop_check=lambda: _training_stop_requested(client, training_id),
        )
        candidate_result = await candidate.run()
        candidate_summary = candidate_result["summary"]
        episode_entry: dict[str, Any] = dict(candidate_summary)

        if mode == "train":
            red_trace = candidate_result["trace"]
            red_transitions = _transition_batch(red_trace, "red", red)
            blue_transitions = _transition_batch(red_trace, "blue", blue)
            # Probe the unchanged policy on the episode's original observations.
            red_reference = (
                np.asarray(red_trace[0]["red_observation"].features, dtype=np.float32)
                if red_trace
                else np.zeros(len(FEATURES["red"]), dtype=np.float32)
            )
            blue_reference = (
                np.asarray(red_trace[0]["blue_observation"].features, dtype=np.float32)
                if red_trace
                else np.zeros(len(FEATURES["blue"]), dtype=np.float32)
            )
            red_before = red.probabilities(red_reference)
            blue_before = blue.probabilities(blue_reference)
            red_metrics = red.update(red_transitions)
            blue_metrics = blue.update(blue_transitions)
            if red_transitions and blue_transitions:
                update_count += 1
            digest = _save_checkpoint(
                checkpoint_path,
                training_id=training_id,
                seed=seed,
                episode_count=episode_index + 1,
                red=red,
                blue=blue,
                red_updates=red.optimizer_step_count,
                blue_updates=blue.optimizer_step_count,
            )
            red_after = red.probabilities(red_reference)
            blue_after = blue.probabilities(blue_reference)
            for role, before, after, reference, metrics, transitions in (
                ("red", red_before, red_after, red_reference, red_metrics, red_transitions),
                ("blue", blue_before, blue_after, blue_reference, blue_metrics, blue_transitions),
            ):
                await candidate._event(
                    "sqli_policy_probe",
                    {
                        "role": role,
                        "reference_observation": reference.tolist(),
                        "before_action_probabilities": [list(head) for head in before],
                        "after_action_probabilities": [list(head) for head in after],
                        "policy_version_before": int(metrics["policy_version_before"]),
                        "policy_version_after": int(metrics["policy_version_after"]),
                        "same_reference_observation": True,
                    },
                )
                metrics_with_context = {
                    **metrics,
                    "episode_index": float(episode_index),
                    "accepted_transitions": float(len(transitions)),
                    "target_version": TARGET_VERSION,
                    "observation_schema_id": OBSERVATION_SCHEMA,
                    "action_schema_id": ACTION_SCHEMA,
                    "reward_schema_id": REWARD_SCHEMA,
                }
                await candidate._event(
                    "sqli_ppo_update",
                    {
                        "role": role,
                        "update_index": episode_index + 1,
                        "global_step": sum(
                            len(_transition_batch(item_trace, role, red if role == "red" else blue))
                            for item_trace in [red_trace]
                        ),
                        "metrics": metrics_with_context,
                        "checkpoint_sha256": digest,
                        "policy_version": int(metrics["policy_version_after"]),
                        "updated": bool(transitions),
                        "update_status": "applied"
                        if transitions
                        else "skipped_no_accepted_transitions",
                        "algorithm": "PPO_clipped_surrogate_GAE",
                    },
                )
                if transitions:
                    await _record_update(
                        client,
                        training_id,
                        role,
                        episode_index + 1,
                        len(transitions),
                        metrics_with_context,
                        digest,
                    )
            episode_entry["optimizer_metrics"] = {"red": red_metrics, "blue": blue_metrics}
            episode_entry["policy_versions"] = {
                "red": red.policy_version,
                "blue": blue.policy_version,
            }
            episode_entry["checkpoint_sha256"] = digest
        else:
            comparator_id = uuid4()
            comparator_blue = SqliPPOPolicy(
                "blue",
                seed=seed + episode_index + 10000,
                learning_rate=float(config["learning_rate"]),
            )
            comparator = SqliEpisodeRunner(
                client,
                training_uuid,
                comparator_id,
                seed=seed + episode_index,
                max_steps=max_steps,
                red_policy=red,
                blue_policy=comparator_blue,
                mode="evaluate",
                comparator=True,
                stop_check=lambda: _training_stop_requested(client, training_id),
            )
            baseline_result = await comparator.run()
            baseline = baseline_result["summary"]
            episode_entry["paired_comparator_episode_id"] = str(comparator_id)
            episode_entry["paired_comparator"] = baseline
            if (
                candidate_result["status"] == "completed"
                and baseline_result["status"] == "completed"
            ):
                candidate_score = (
                    float(candidate_summary["verified_prevention_rate"] or 0.0)
                    + float(candidate_summary["legitimate_query_availability"] or 0.0)
                    - float(candidate_summary["unauthorized_read_rate"] or 0.0)
                    - float(candidate_summary["false_positive_rate"] or 0.0)
                )
                baseline_score = (
                    float(baseline["verified_prevention_rate"] or 0.0)
                    + float(baseline["legitimate_query_availability"] or 0.0)
                    - float(baseline["unauthorized_read_rate"] or 0.0)
                    - float(baseline["false_positive_rate"] or 0.0)
                )
                candidate_scores.append(candidate_score)
                comparator_scores.append(baseline_score)
            episode_entry["comparator_label"] = "No-defense control comparator (evaluation only)"
        episode_summaries.append(episode_entry)
        episode_index += 1

    result: dict[str, Any] = {
        "scenario_id": SCENARIO_ID,
        "target_version": TARGET_VERSION,
        "observation_schema_id": OBSERVATION_SCHEMA,
        "feature_schema_id": FEATURE_SCHEMA,
        "action_schema_id": ACTION_SCHEMA,
        "reward_schema_id": REWARD_SCHEMA,
        "episodes": episode_summaries,
        "completed_episodes": sum(
            item.get("status") == "completed" and item.get("kind") != "evaluation_comparator"
            for item in episode_summaries
        ),
        "requested_episodes": requested,
        "independent_training_seeds": 1 if mode == "train" else None,
        "episode_updates": update_count,
        "policy_versions": {"red": red.policy_version, "blue": blue.policy_version},
        "optimizer_metrics": {},
        "research_criteria_met": not research
        or (
            mode == "train"
            and sum(item.get("status") == "completed" for item in episode_summaries) >= requested
        ),
        "evidence_level": "real_http_target" if episode_summaries else "no_episode_completed",
    }
    if mode == "train":
        if not update_count:
            raise RuntimeError("no SQLi episode boundary reached a PPO update")
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        result["checkpoint"] = str(checkpoint_path)
        result["checkpoint_sha256"] = digest
        metrics_path = metrics_dir / "training.jsonl"
        metrics_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in episode_summaries)
        )
    else:
        interval = _bootstrap_interval(candidate_scores, comparator_scores, seed)
        source_run_response = await client.get(f"/range/training-runs/{source_id}")
        source_run_response.raise_for_status()
        source_result = source_run_response.json().get("result") or {}
        independent_seeds = int(source_result.get("independent_training_seeds", 1))
        availability: list[float] = []
        for item in episode_summaries:
            value = item.get("legitimate_query_availability")
            if item.get("kind") == "candidate" and isinstance(value, (int, float)):
                availability.append(float(value))
        availability_met = bool(availability) and all(value == 1.0 for value in availability)
        supported = (
            len(candidate_scores) >= 100
            and interval is not None
            and interval[1] > 0.0
            and independent_seeds >= 5
            and availability_met
        )
        result["evaluation"] = {
            "schema_id": "sqli.evaluation.v1",
            "target_version": TARGET_VERSION,
            "checkpoint_run_id": source_id,
            "comparator": "No-defense control comparator (evaluation only)",
            "paired_episodes": len(candidate_scores),
            "requested_pairs": requested,
            "candidate_mean_net_security_score": float(np.mean(candidate_scores))
            if candidate_scores
            else None,
            "comparator_mean_net_security_score": float(np.mean(comparator_scores))
            if comparator_scores
            else None,
            "paired_mean_difference": interval[0] if interval else None,
            "paired_bootstrap_ci_95": list(interval[1:]) if interval else None,
            "training_seed_count": independent_seeds,
            "required_training_seeds": 5,
            "required_held_out_pairs": 100,
            "availability_criterion_met": availability_met,
            "supported_improvement": supported,
            "claim": "supported improvement" if supported else "no supported improvement",
        }
        (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        (run_dir / "evaluation" / "summary.json").write_text(
            json.dumps(result["evaluation"], sort_keys=True, indent=2) + "\n"
        )
        result["research_criteria_met"] = not research or len(candidate_scores) >= requested
    return result


async def _training_stop_requested(client: httpx.AsyncClient, training_id: str) -> bool:
    heartbeat = await client.post(f"/range/training-runs/{training_id}/heartbeat")
    heartbeat.raise_for_status()
    record = await client.get(f"/range/training-runs/{training_id}")
    record.raise_for_status()
    return bool(record.json().get("stop_requested"))
