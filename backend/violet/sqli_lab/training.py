"""Episode runner, persisted replay, and PPO updates for the dedicated SQLi lab."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

import numpy as np
import torch

from violet.sqli_lab.contracts import (
    ACTION_SCHEMA,
    BLUE_ACTIONS,
    FEATURE_SCHEMA,
    INTERACTION_PROTOCOL_SCHEMA,
    LEGACY_INTERACTION_PROTOCOL_SCHEMA,
    OBSERVATION_SCHEMA,
    RED_COMPONENTS,
    REWARD_SCHEMA,
    SCENARIO_ID,
    TARGET_VERSION,
    SqliObservation,
    SqliRequestAction,
)
from violet.sqli_lab.learning import (
    FEATURES,
    SqliPPOPolicy,
    SqliSample,
    ppo_settings,
)
from violet.sqli_lab.learning import (
    PPO_OPTIMIZATION_PASSES as _PPO_OPTIMIZATION_PASSES,
)

if TYPE_CHECKING:
    import httpx

PPO_OPTIMIZATION_PASSES = _PPO_OPTIMIZATION_PASSES
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
    "format": "sqli.joint_ppo.v4",
    "scenario_id": SCENARIO_ID,
    "target_version": TARGET_VERSION,
    "observation_schema_id": OBSERVATION_SCHEMA,
    "feature_schema_id": FEATURE_SCHEMA,
    "action_schema_id": ACTION_SCHEMA,
    "reward_schema_id": REWARD_SCHEMA,
    "interaction_protocol_schema_id": INTERACTION_PROTOCOL_SCHEMA,
    "red_component_catalog": {key: list(value) for key, value in RED_COMPONENTS.items()},
    "blue_action_catalog": list(BLUE_ACTIONS),
}
SUPPORTED_REWARD_SCHEMAS = {"sqli.reward.v1", "sqli.reward.v2", REWARD_SCHEMA}


def _validate_checkpoint_contract(
    payload: dict[str, Any],
    *,
    expected_training_run_id: str | None = None,
    expected_reward_schema_id: str | None = None,
    expected_interaction_protocol_schema_id: str | None = None,
) -> None:
    expected = {
        key: value
        for key, value in CHECKPOINT_CONTRACT.items()
        if key not in {"reward_schema_id", "interaction_protocol_schema_id"}
    }
    if (
        any(
            payload.get(key) != value
            for key, value in expected.items()
            if key != "format"
        )
        or payload.get("format") not in {
            "sqli.joint_ppo.v1",
            "sqli.joint_ppo.v2",
            "sqli.joint_ppo.v3",
            "sqli.joint_ppo.v4",
        }
        or payload.get("reward_schema_id") not in SUPPORTED_REWARD_SCHEMAS
        or (
            expected_reward_schema_id is not None
            and payload.get("reward_schema_id") != expected_reward_schema_id
        )
    ):
        raise ValueError("SQLi checkpoint contract is incompatible with this scenario")
    if expected_training_run_id is not None and payload.get("training_run_id") != expected_training_run_id:
        raise ValueError("SQLi checkpoint policy metadata belongs to a different training run")

    if expected_interaction_protocol_schema_id is not None and payload.get(
        "interaction_protocol_schema_id", LEGACY_INTERACTION_PROTOCOL_SCHEMA
    ) != expected_interaction_protocol_schema_id:
        raise ValueError("SQLi checkpoint uses a different interaction protocol")

    metadata = payload.get("checkpoint_metadata")
    if payload.get("format") in {"sqli.joint_ppo.v2", "sqli.joint_ppo.v3", "sqli.joint_ppo.v4"}:
        if not isinstance(metadata, dict):
            raise ValueError("SQLi checkpoint metadata is missing")
        expected_metadata_schema = {
            "sqli.joint_ppo.v2": "sqli.checkpoint_metadata.v1",
            "sqli.joint_ppo.v3": "sqli.checkpoint_metadata.v2",
            "sqli.joint_ppo.v4": "sqli.checkpoint_metadata.v3",
        }[str(payload.get("format"))]
        completed_updates = payload.get("completed_training_updates")
        metadata_updates = metadata.get("completed_training_updates")
        versions = payload.get("policy_versions")
        if (
            metadata.get("schema_id") != expected_metadata_schema
            or metadata.get("training_run_id") != payload.get("training_run_id")
            or metadata.get("target_version") != payload.get("target_version")
            or metadata.get("reward_schema_id") != payload.get("reward_schema_id")
            or (
                payload.get("format") == "sqli.joint_ppo.v3"
                and (
                    payload.get("interaction_protocol_schema_id") != INTERACTION_PROTOCOL_SCHEMA
                    or metadata.get("interaction_protocol_schema_id") != INTERACTION_PROTOCOL_SCHEMA
                )
            )
            or (
                payload.get("format") == "sqli.joint_ppo.v4"
                and (
                    payload.get("model_architecture") != "factorized_gru_actor_critic.v1"
                    or metadata.get("model_architecture") != "factorized_gru_actor_critic.v1"
                    or payload.get("interaction_protocol_schema_id") != INTERACTION_PROTOCOL_SCHEMA
                )
            )
            or metadata.get("policy_versions") != versions
            or not isinstance(completed_updates, int)
            or completed_updates < 0
            or metadata_updates != completed_updates
            or metadata.get("campaign_revision") != payload.get("campaign_revision")
            or not isinstance(versions, dict)
            or any(
                not isinstance(versions.get(role), int)
                or versions[role] < (0 if completed_updates == 0 else 1)
                for role in ("red", "blue")
            )
            or metadata.get("optimization_passes") != PPO_OPTIMIZATION_PASSES
            or (
                metadata.get("ppo_settings") is not None
                and (
                    not isinstance(metadata.get("ppo_settings"), dict)
                    or metadata.get("ppo_settings") != payload.get("ppo_settings")
                    or metadata["ppo_settings"].get("optimization_passes")
                    != PPO_OPTIMIZATION_PASSES
                )
            )
        ):
            raise ValueError("SQLi checkpoint policy metadata is inconsistent")


def _action_is_injection(action: SqliRequestAction) -> bool:
    return bool(action.close_quote and action.connector != "none" and action.comment)


def _row_list(response: dict[str, Any]) -> list[dict[str, Any]]:
    rows = response.get("rows", [])
    return [item for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unauthorized_count(response: dict[str, Any], actor_ref: str) -> int:
    return sum(str(row.get("owner_ref")) != actor_ref for row in _row_list(response))


def _intentional_guest_isolation(
    result: dict[str, Any], expected_controls: dict[str, Any]
) -> bool:
    """Recognize the target's evidence-backed 503 from Blue's isolation control."""
    response = _object(result.get("response"))
    telemetry = _object(result.get("telemetry"))
    observed_controls = _object(telemetry.get("controls"))
    expected_version = expected_controls.get("state_version")
    observed_version = observed_controls.get("state_version")
    try:
        expected_remaining = int(expected_controls.get("isolate_guest_remaining", 0))
        observed_remaining = int(observed_controls.get("isolate_guest_remaining", -1))
    except (TypeError, ValueError):
        return False
    return (
        result.get("status") == "completed"
        and int(result.get("http_status") or 0) == 503
        and response.get("detail") == "temporary guest isolation"
        and telemetry.get("evidence_status") == "verified"
        and expected_remaining > 0
        and observed_remaining in {0, 1}
        and expected_version is not None
        and observed_version == expected_version
    )


def _normal(value: float, denominator: float = 1.0) -> float:
    return float(np.clip(value / max(denominator, 1e-8), 0.0, 1.0))


def _red_reward_components(
    *,
    accepted: bool,
    verified_unauthorized_rows: int,
    accepted_syntax_progress: bool,
    query_rejected: bool,
    returned_rows: bool = False,
    reward_schema_id: str = REWARD_SCHEMA,
) -> dict[str, float]:
    """Return the versioned Red reward components for target-verified requests."""
    verified_read = accepted and verified_unauthorized_rows > 0
    if reward_schema_id == "sqli.reward.v1":
        return {
            "confirmed_unauthorized_read": float(verified_read),
            "returned_row_progress": 0.02 * float(accepted and returned_rows),
            "request_cost": -0.02 * float(accepted),
        }
    if reward_schema_id not in {"sqli.reward.v2", REWARD_SCHEMA}:
        raise ValueError("unsupported SQLi reward contract")
    return {
        "confirmed_unauthorized_read": float(verified_read),
        "accepted_syntax_progress": 0.08
        * float(accepted and accepted_syntax_progress and not verified_read),
        "rejected_request": -0.01 * float(accepted and query_rejected and not verified_read),
    }


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
        challenge_id: UUID | None = None,
        policy_run_id: str | None = None,
        episode_kind: str | None = None,
        reward_schema_id: str = REWARD_SCHEMA,
        checkpoint_variant: str = "final",
        vulnerable_configuration_probability: float = 1.0,
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
        self.challenge_id = challenge_id
        self.policy_run_id = policy_run_id or str(run_id)
        self.episode_kind = episode_kind or ("training" if mode == "train" else "candidate")
        if reward_schema_id not in SUPPORTED_REWARD_SCHEMAS:
            raise ValueError("unsupported SQLi reward contract")
        self.reward_schema_id = reward_schema_id
        self.checkpoint_variant = checkpoint_variant
        self.vulnerable_configuration_probability = float(
            np.clip(vulnerable_configuration_probability, 0.0, 1.0)
        )
        self.telemetry_history: list[dict[str, Any]] = []
        self.active_controls: dict[str, Any] = {
            "parameterized": False,
            "rate_limit_guest": False,
            "isolate_guest_remaining": 0,
        }
        self.episode_start_controls: dict[str, Any] = {}
        self.guest_requests = 0
        self.http_failures = 0
        self.injection_signals = 0
        self.last_red_result: dict[str, Any] = {}
        self.last_red_action: SqliRequestAction | None = None

    async def _event(self, event_type: str, payload: dict[str, Any]) -> None:
        payload.setdefault("interaction_protocol_schema_id", INTERACTION_PROTOCOL_SCHEMA)
        payload.setdefault("reward_schema_id", self.reward_schema_id)
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

    async def _control(
        self,
        action: str,
        decision_id: UUID | None,
        step_id: int | None = None,
        *,
        fixture_seed: int | None = None,
    ) -> dict[str, Any]:
        response = await self.client.post(
            "/range/sqli/controls",
            json={
                "run_id": str(self.run_id),
                "episode_id": str(self.episode_id),
                "decision_id": str(decision_id) if decision_id else None,
                "action": action,
                "vulnerable_configuration_probability": self.vulnerable_configuration_probability,
                "fixture_seed": fixture_seed,
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
                    {
                        "command_id": command_id,
                        "decision_id": str(decision_id) if decision_id else None,
                        "step_id": step_id,
                        "action": action,
                        "acknowledgement": payload,
                    },
                )
                if payload.get("status") == "applied":
                    self.active_controls = dict(payload.get("controls") or {})
                if action == "RESET":
                    self.episode_start_controls = dict(self.active_controls)
                return payload
            await asyncio.sleep(0.08)
        unknown = {"status": "outcome_unknown", "command_id": command_id}
        await self._event(
            "sqli_control_acknowledged",
            {
                "command_id": command_id,
                "decision_id": str(decision_id) if decision_id else None,
                "step_id": step_id,
                "action": action,
                "acknowledgement": unknown,
            },
        )
        return unknown

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
                "challenge_id": (
                    str(self.challenge_id)
                    if purpose == "policy" and self.challenge_id is not None
                    else None
                ),
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
            "method": result.get("method", "POST"),
            "route_id": result.get("route_id", "lookup"),
            "route": "/api/lookup",
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
        self.red_policy.reset_hidden_state()
        if self.blue_policy is not None:
            self.blue_policy.reset_hidden_state()
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
                "reward_schema_id": self.reward_schema_id,
                "target_fixture_set": "synthetic_records.v1",
                "comparator": "no_defense_control" if self.comparator else None,
                "episode_kind": self.episode_kind,
                "challenge_id": str(self.challenge_id) if self.challenge_id else None,
                "checkpoint_run_id": self.policy_run_id,
                "checkpoint_variant": self.checkpoint_variant,
                "vulnerable_configuration_probability": self.vulnerable_configuration_probability,
                "policy_versions": {
                    "red": self.red_policy.policy_version,
                    "blue": self.blue_policy.policy_version if self.blue_policy else None,
                },
            },
        )
        reset = await self._control("RESET", None, fixture_seed=self.seed)
        if reset.get("status") != "applied":
            summary = {
                "schema_id": "sqli.episode_outcome.v2",
                "episode_id": str(self.episode_id),
                "seed": self.seed,
                "status": "incomplete",
                "reason": "target_reset_unknown",
                "kind": self.episode_kind,
                "winner": None,
                "steps": 0,
                "verified_exposure": False,
                "challenge_disclosed": False,
                "exposure_verification": {"status": "incomplete", "evidence_verified": False},
                "legitimate_query_served": False,
                "legitimate_query_availability": None,
                "red_return": 0.0,
                "blue_return": 0.0,
                "policy_versions": {
                    "red": self.red_policy.policy_version,
                    "blue": self.blue_policy.policy_version if self.blue_policy else None,
                },
                "target_version": TARGET_VERSION,
            }
            await self._event("sqli_episode_summary", summary)
            await self._event(
                "sqli_episode_outcome",
                {"phase": self.episode_kind, **summary, "checkpoint_run_id": self.policy_run_id},
            )
            finish = await self.client.post(
                f"/range/runs/{self.run_id}/episodes/{self.episode_id}/finish",
                json={"status": "incomplete", "reason": "target_reset_unknown"},
            )
            finish.raise_for_status()
            return {"status": "incomplete", "reason": "target_reset_unknown", "trace": [], "summary": summary}

        trace: list[dict[str, Any]] = []
        terminal_status = "completed"
        reason: str | None = None
        for step_id in range(self.max_steps):
            # Both policies receive observations assembled from the state at the
            # start of this round. Neither action or result below feeds the other
            # policy's same-round observation.
            red_observation = self._red_observation(step_id)
            blue_observation = self._blue_observation(step_id)
            red_obs_array = np.asarray(red_observation.features, dtype=np.float32)
            blue_obs_array = np.asarray(blue_observation.features, dtype=np.float32)
            snapshot_id = str(uuid4())

            red_action_dict, red_sample = self.red_policy.act(
                red_obs_array, stochastic=self.mode == "train"
            )
            red_action = SqliRequestAction.model_validate(red_action_dict)
            red_decision_id = uuid4()

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
                blue_policy_id = f"blue-ppo-{self.policy_run_id}"

            # Persist both committed decisions before applying the control or
            # sending Red's request, so the stored actions also share one cut.
            await self._event(
                "sqli_red_decision",
                {
                    "decision_id": str(red_decision_id),
                    "round_snapshot_id": snapshot_id,
                    "step_id": step_id,
                    "policy_id": f"red-ppo-{self.policy_run_id}",
                    "policy_role": "red",
                    "observation_phase": "round_start",
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
                    "joint_log_probability": red_sample.old_log_probability,
                    "sampled_action_probability": float(np.exp(red_sample.old_log_probability)),
                    "value_estimate": red_sample.value_estimate,
                    "critic_value": red_sample.value_estimate,
                    "entropy": red_sample.entropy,
                    "policy_version": red_sample.policy_version,
                    "inference_mode": self.mode,
                    "action_schema_id": ACTION_SCHEMA,
                    "available_action_mask": list(red_observation.legal_action_mask),
                    "inference_ms": red_sample.inference_ms,
                    "recurrent_state_before": list(red_sample.recurrent_state),
                    "recurrent_state_after": list(red_sample.recurrent_state_after),
                },
            )
            await self._event(
                "sqli_blue_decision",
                {
                    "phase": "policy_decision",
                    "decision_id": str(blue_decision_id),
                    "round_snapshot_id": snapshot_id,
                    "step_id": step_id,
                    "policy_id": blue_policy_id,
                    "policy_role": "blue",
                    "observation_phase": "round_start",
                    "observation": blue_observation.model_dump(mode="json"),
                    "feature_names": list(FEATURES["blue"]),
                    "selected_action": {"operation": blue_action},
                    "selected_action_indices": blue_indices,
                    "component_distributions": blue_probabilities,
                    "selected_component_probabilities": list(blue_sample.selected_probabilities)
                    if blue_sample
                    else [1.0],
                    "old_log_probability": blue_log_prob,
                    "joint_log_probability": blue_log_prob,
                    "sampled_action_probability": float(np.exp(blue_log_prob)),
                    "value_estimate": blue_value,
                    "critic_value": blue_value,
                    "entropy": blue_entropy,
                    "policy_version": blue_sample.policy_version if blue_sample else 0,
                    "inference_mode": "comparator" if self.comparator else self.mode,
                    "action_schema_id": "sqli.blue_action_catalog.v1",
                    "available_action_mask": list(blue_observation.legal_action_mask),
                    "inference_ms": blue_sample.inference_ms if blue_sample else 0.0,
                    "recurrent_state_before": list(blue_sample.recurrent_state) if blue_sample else [],
                    "recurrent_state_after": list(blue_sample.recurrent_state_after) if blue_sample else [],
                },
            )

            control_ack: dict[str, Any] = {"status": "applied", "controls": self.active_controls}
            if not self.comparator and blue_action != "MONITOR":
                control_ack = await self._control(blue_action, blue_decision_id, step_id)
            else:
                await self._event(
                    "sqli_control_acknowledged",
                    {
                        "decision_id": str(blue_decision_id),
                        "round_snapshot_id": snapshot_id,
                        "step_id": step_id,
                        "action": blue_action,
                        "acknowledgement": control_ack,
                    },
                )
            if control_ack.get("status") != "applied":
                terminal_status, reason = "incomplete", "blue_control_not_acknowledged"
                break

            # Blue's control acknowledgement is persisted before this request is
            # submitted to the target. A failed or unknown control never permits
            # an unprotected same-round Red request through.
            red_result = await self._job(red_action, red_decision_id, step_id, "policy")
            await self._request_event(red_result, purpose="policy", step_id=step_id)
            red_response = _object(red_result.get("response"))
            red_telemetry = _object(red_result.get("telemetry"))
            response_unauthorized_rows = _unauthorized_count(red_response, "guest")
            unauthorized_rows = int(red_telemetry.get("unauthorized_rows") or 0)
            red_http_status = int(red_result.get("http_status") or 0)
            red_isolation_applied = _intentional_guest_isolation(
                red_result, self.active_controls
            )
            red_accepted = (
                red_result.get("status") == "completed"
                and red_telemetry.get("evidence_status") == "verified"
                and (red_http_status < 500 or red_isolation_applied)
                and int(red_telemetry.get("unauthorized_rows") or 0)
                == response_unauthorized_rows
            )
            if not red_accepted:
                unauthorized_rows = 0
            verified_exposure = red_accepted and unauthorized_rows > 0
            challenge_disclosed = (
                red_accepted
                and self.challenge_id is not None
                and red_telemetry.get("challenge_id") == str(self.challenge_id)
                and bool(red_telemetry.get("challenge_disclosed"))
                and bool(red_response.get("challenge_disclosed"))
            )
            query_rejected = bool(red_telemetry.get("query_rejected"))
            syntax_progress = bool(
                red_accepted
                and not query_rejected
                and not red_isolation_applied
                and red_telemetry.get("query_syntax_signal")
            )
            red_components = _red_reward_components(
                accepted=red_accepted,
                verified_unauthorized_rows=unauthorized_rows,
                accepted_syntax_progress=syntax_progress,
                query_rejected=query_rejected,
                returned_rows=bool(_row_list(red_response)),
            reward_schema_id=self.reward_schema_id,
        )
            red_reward = sum(red_components.values()) if red_accepted else 0.0

            # Verifier traffic is external evidence about the Blue decision. It
            # is never exposed as a policy observation or inserted as a PPO
            # action/transition of its own.
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
            attack_isolation_applied = (
                blue_action == "ISOLATE_GUEST_LOOKUP"
                and _intentional_guest_isolation(
                    verifier_attack, _object(control_ack.get("controls"))
                )
            )
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
                and (
                    int(verifier_attack.get("http_status") or 0) < 500
                    or attack_isolation_applied
                )
            )
            legit_verified = (
                verifier_legit.get("status") == "completed"
                and int(verifier_legit.get("http_status") or 0) < 500
                and _object(verifier_legit.get("telemetry")).get("evidence_status") == "verified"
            )
            prevented = attack_verified and attack_rows == 0
            false_positive = legit_verified and not legitimate_served
            verification = {
                "schema_id": "sqli.control_verification.v1",
                "step_id": step_id,
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
            blue_accepted = attack_verified and legit_verified
            blue_reward = (
                sum(blue_components.values()) if blue_accepted and not self.comparator else 0.0
            )
            reward_payload = {
                "schema_id": self.reward_schema_id,
                "role": "red",
                "step_id": step_id,
                "components": red_components,
                "total": red_reward,
                "reward_version": self.reward_schema_id,
                "verified_unauthorized_read": unauthorized_rows > 0,
                "accepted": red_accepted,
            }
            blue_reward_payload = {
                "schema_id": self.reward_schema_id,
                "role": "blue",
                "step_id": step_id,
                "components": blue_components,
                "total": blue_reward,
                "reward_version": self.reward_schema_id,
                "accepted": blue_accepted and not self.comparator,
                "verification": verification,
            }
            await self._event("sqli_reward", {"red": reward_payload, "blue": blue_reward_payload})

            item = {
                "step_id": step_id,
                "round_snapshot_id": snapshot_id,
                "red_observation": red_observation,
                "red_observation_array": red_obs_array,
                "red_action": red_action,
                "red_sample": red_sample,
                "red_decision_id": red_decision_id,
                "red_result": red_result,
                "red_accepted": red_accepted,
                "red_reward": red_reward,
                "red_reward_components": red_components,
                "verified_exposure": verified_exposure,
                "challenge_disclosed": challenge_disclosed,
                "response_unauthorized_rows": response_unauthorized_rows,
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
            if not red_accepted or not blue_accepted or not attack_verified or not legit_verified:
                terminal_status, reason = "incomplete", "unknown_or_unverified_outcome"
                break
            if challenge_disclosed:
                reason = "verified_secret_disclosure"
                break
            if self.challenge_id is None and verified_exposure:
                reason = "verified_unauthorized_read"
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
                next_red_state = (
                    np.asarray(trace[index + 1]["red_sample"].recurrent_state, dtype=np.float32)
                    if index + 1 < len(trace)
                    else np.asarray(item["red_sample"].recurrent_state_after, dtype=np.float32)
                )
                next_red_value = self.red_policy.value(next_red, next_red_state)
                next_blue_value = (
                    self.blue_policy.value(
                        next_blue,
                        np.asarray(
                            trace[index + 1]["blue_sample"].recurrent_state
                            if index + 1 < len(trace)
                            else item["blue_sample"].recurrent_state_after,
                            dtype=np.float32,
                        ),
                    )
                    if self.blue_policy is not None
                    else 0.0
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
                        "joint_log_probability": item["red_sample"].old_log_probability,
                        "sampled_action_probability": float(
                            np.exp(item["red_sample"].old_log_probability)
                        ),
                        "value_estimate": item["red_sample"].value_estimate,
                        "recurrent_state": list(item["red_sample"].recurrent_state),
                        "recurrent_state_after": list(item["red_sample"].recurrent_state_after),
                        "available_action_mask": list(item["red_observation"].legal_action_mask),
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
                        "reward_schema_id": self.reward_schema_id,
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
                            "joint_log_probability": item["blue_sample"].old_log_probability,
                            "sampled_action_probability": float(
                                np.exp(item["blue_sample"].old_log_probability)
                            ),
                            "value_estimate": item["blue_sample"].value_estimate,
                            "recurrent_state": list(item["blue_sample"].recurrent_state),
                            "recurrent_state_after": list(item["blue_sample"].recurrent_state_after),
                            "available_action_mask": list(item["blue_observation"].legal_action_mask),
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
                            "reward_schema_id": self.reward_schema_id,
                        },
                    )

        unauthorized = sum(int(item["verified_exposure"]) for item in trace)
        verified_exposure = any(bool(item["verified_exposure"]) for item in trace)
        challenge_disclosed = any(bool(item["challenge_disclosed"]) for item in trace)
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
        final_legitimate_served = bool(
            trace
            and trace[-1]["verification"].get("evidence_verified")
            and trace[-1]["verification"].get("legitimate_query_served")
        )
        winner, winner_reason = _episode_winner(
            status=terminal_status,
            challenge_mode=self.challenge_id is not None,
            challenge_disclosed=challenge_disclosed,
            verified_exposure=verified_exposure,
            steps=len(trace),
            max_steps=self.max_steps,
            final_legitimate_served=final_legitimate_served,
        )
        if winner_reason is not None:
            reason = winner_reason
        elif terminal_status == "completed":
            terminal_status = "incomplete"
            reason = "legitimate_access_not_verified_at_round_limit"
        exposure_verification = {
            "status": "verified" if any(item["red_accepted"] for item in trace) else "incomplete",
            "evidence_verified": bool(trace) and all(item["red_accepted"] for item in trace),
            "verified_exposure": verified_exposure,
            "challenge_disclosed": challenge_disclosed,
            "exposure_count": unauthorized,
        }
        summary = {
            "schema_id": "sqli.episode_outcome.v2",
            "episode_id": str(self.episode_id),
            "seed": self.seed,
            "status": terminal_status,
            "reason": reason,
            "kind": "evaluation_comparator" if self.comparator else self.episode_kind,
            "winner": winner,
            "steps": len(trace),
            "max_steps": self.max_steps,
            "verified_exposure": verified_exposure,
            "challenge_disclosed": challenge_disclosed,
            "initial_server_configuration": self.episode_start_controls,
            "exposure_verification": exposure_verification,
            "legitimate_query_served": final_legitimate_served,
            "red_return": round(sum(float(item["red_reward"]) for item in trace), 6),
            "blue_return": round(sum(float(item["blue_reward"]) for item in trace), 6),
            "unauthorized_read_rate": _normal(
                unauthorized, max(1, sum(bool(item["red_accepted"]) for item in trace))
            ),
            "verified_prevention_rate": prevention_rate,
            "legitimate_query_availability": availability_rate,
            "false_positive_rate": false_positive_rate,
            "verification_count": len(prevention_checks),
            "policy_versions": {
                "red": self.red_policy.policy_version,
                "blue": self.blue_policy.policy_version if self.blue_policy else None,
            },
            "checkpoint_run_id": self.policy_run_id,
            "target_version": TARGET_VERSION,
            "action_schema_id": ACTION_SCHEMA,
            "reward_schema_id": self.reward_schema_id,
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
        role_sample = item[sample_key]
        next_state = (
            trace[index + 1][sample_key].recurrent_state
            if index + 1 < len(trace)
            else role_sample.recurrent_state_after
        )
        next_gap = index + 1 < len(trace) and not trace[index + 1].get(accepted_key)
        result.append(
            {
                "observation": item[obs_key],
                "decision_id": str(item[f"{role}_decision_id"]),
                "step_id": item["step_id"],
                "policy_version": sample.policy_version,
                "action_indices": sample.action_indices,
                "recurrent_state": sample.recurrent_state,
                "component_probabilities": sample.component_probabilities,
                "selected_probabilities": sample.selected_probabilities,
                "action": (
                    item["red_action"].model_dump(mode="json")
                    if role == "red"
                    else {"operation": item["blue_action"]}
                ),
                "old_log_probability": sample.old_log_probability,
                "value": sample.value_estimate,
                "reward": item[reward_key],
                "next_value": policy.value(
                    next_obs, np.asarray(next_state, dtype=np.float32)
                ),
                "terminated": False,
                "truncated": index + 1 == len(trace) or next_gap,
            }
        )
    if result:
        result[-1]["truncated"] = True
    return result


def _episode_winner(
    *,
    status: str,
    challenge_mode: bool,
    challenge_disclosed: bool,
    verified_exposure: bool,
    steps: int,
    max_steps: int,
    final_legitimate_served: bool,
) -> tuple[str | None, str | None]:
    """Return a winner only when the corresponding target evidence is verified."""
    if status != "completed":
        return None, None
    if (challenge_mode and challenge_disclosed) or (not challenge_mode and verified_exposure):
        reason = "verified_secret_disclosure" if challenge_mode else "verified_unauthorized_read"
        return "red", reason
    if steps == max_steps and final_legitimate_served:
        return "blue", f"{max_steps}_rounds_without_disclosure"
    return None, None


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
    completed_training_updates: int,
    campaign_revision: str | None,
    reward_schema_id: str,
    interaction_protocol_schema_id: str,
    optimizer_settings: dict[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "sqli.joint_ppo.v4",
        "model_architecture": "factorized_gru_actor_critic.v1",
        "scenario_id": SCENARIO_ID,
        "target_version": TARGET_VERSION,
        "observation_schema_id": OBSERVATION_SCHEMA,
        "feature_schema_id": FEATURE_SCHEMA,
        "action_schema_id": ACTION_SCHEMA,
        "reward_schema_id": reward_schema_id,
        "interaction_protocol_schema_id": interaction_protocol_schema_id,
        "red_component_catalog": {key: list(value) for key, value in RED_COMPONENTS.items()},
        "blue_action_catalog": list(BLUE_ACTIONS),
        "training_run_id": training_id,
        "seed": seed,
        "episode_count": episode_count,
        "policy_versions": {"red": red.policy_version, "blue": blue.policy_version},
        "update_counts": {"red": red_updates, "blue": blue_updates},
        "completed_training_updates": completed_training_updates,
        "campaign_revision": campaign_revision,
        "ppo_settings": optimizer_settings,
        "checkpoint_metadata": {
            "schema_id": "sqli.checkpoint_metadata.v3",
            "model_architecture": "factorized_gru_actor_critic.v1",
            "training_run_id": training_id,
            "target_version": TARGET_VERSION,
            "reward_schema_id": reward_schema_id,
            "interaction_protocol_schema_id": interaction_protocol_schema_id,
            "completed_training_updates": completed_training_updates,
            "policy_versions": {"red": red.policy_version, "blue": blue.policy_version},
            "optimization_passes": PPO_OPTIMIZATION_PASSES,
            "campaign_revision": campaign_revision,
            "ppo_settings": optimizer_settings,
        },
        "policy_rng_states": {
            "red": red.rng.bit_generator.state,
            "blue": blue.rng.bit_generator.state,
        },
        "model_state_dicts": {"red": red.model.state_dict(), "blue": blue.model.state_dict()},
        "optimizer_state_dicts": {
            "red": red.optimizer.state_dict(),
            "blue": blue.optimizer.state_dict(),
        },
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
    }
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save(payload, temp_path)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)
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


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial rate."""
    if total <= 0:
        return 0.0, 1.0
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


async def run_sqli_job(client: httpx.AsyncClient, job: dict[str, Any]) -> dict[str, Any]:
    """Train a frozen policy pair at episode boundaries or evaluate one checkpoint."""
    training_id = str(job["training_run_id"])
    training_uuid = UUID(training_id)
    config = job["config"]
    mode = str(job["mode"])
    reward_schema_id = str(config.get("reward_schema_id", REWARD_SCHEMA))
    interaction_protocol_schema_id = str(config.get("interaction_protocol_schema_id", ""))
    if config.get("scenario_id") != SCENARIO_ID or config.get("target_version") != TARGET_VERSION:
        raise ValueError("training job does not select the versioned SQLi lab scenario")
    if (
        config.get("observation_schema_id") != OBSERVATION_SCHEMA
        or config.get("action_schema_id") != ACTION_SCHEMA
        or reward_schema_id not in SUPPORTED_REWARD_SCHEMAS
        or interaction_protocol_schema_id != INTERACTION_PROTOCOL_SCHEMA
    ):
        raise ValueError(
            "training job has an incompatible SQLi observation, action, reward, or interaction contract"
        )

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    learning_rate = float(config["learning_rate"])
    red = SqliPPOPolicy("red", seed=seed, learning_rate=learning_rate)
    blue = SqliPPOPolicy("blue", seed=seed + 1, learning_rate=learning_rate)
    run_dir = ARTIFACT_ROOT / training_id
    config_dir = run_dir / "config"
    metrics_dir = run_dir / "metrics"
    checkpoint_path = run_dir / "checkpoints" / "joint" / "final.pt"
    initial_checkpoint_path = run_dir / "checkpoints" / "joint" / "initial.pt"
    metrics_name = "training.jsonl" if mode == "train" else "evaluation.jsonl"
    metrics_path = metrics_dir / metrics_name
    for directory in (config_dir, metrics_dir, checkpoint_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    source_id = str(config["checkpoint_run_id"]) if config.get("checkpoint_run_id") else None
    source: dict[str, Any] | None = None
    source_checkpoint_variant = str(config.get("checkpoint_variant", "final"))
    resume_episode_count = 0
    resume_update_count = 0
    own_checkpoint_loaded = False
    if mode == "train" and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _validate_checkpoint_contract(
            payload,
            expected_training_run_id=training_id,
            expected_reward_schema_id=reward_schema_id,
            expected_interaction_protocol_schema_id=interaction_protocol_schema_id,
        )
        if int(payload.get("seed", -1)) != seed:
            raise ValueError("existing SQLi checkpoint seed does not match its run record")
        checkpoint_format = str(payload.get("format"))
        red.load_model_state(payload["model_state_dicts"]["red"], checkpoint_format)
        blue.load_model_state(payload["model_state_dicts"]["blue"], checkpoint_format)
        red.load_optimizer_state(payload["optimizer_state_dicts"]["red"], checkpoint_format)
        blue.load_optimizer_state(payload["optimizer_state_dicts"]["blue"], checkpoint_format)
        red.policy_version = int(payload["policy_versions"]["red"])
        blue.policy_version = int(payload["policy_versions"]["blue"])
        update_counts = payload.get("update_counts") or {}
        red.optimizer_step_count = int(update_counts.get("red", 0))
        blue.optimizer_step_count = int(update_counts.get("blue", 0))
        rng_states = payload.get("policy_rng_states") or {}
        if isinstance(rng_states.get("red"), dict):
            red.rng.bit_generator.state = rng_states["red"]
        if isinstance(rng_states.get("blue"), dict):
            blue.rng.bit_generator.state = rng_states["blue"]
        resume_episode_count = int(payload.get("episode_count", 0))
        resume_update_count = int(payload.get("completed_training_updates", 0))
        own_checkpoint_loaded = True
    elif source_id:
        response = await client.get(f"/range/training-runs/{source_id}")
        response.raise_for_status()
        source = response.json()
        source_status = source.get("status")
        if source_status not in {"completed", "stopped"} or source.get("role") != "joint":
            raise ValueError("SQLi checkpoint source is not a stopped or completed joint run")
        source_config = source.get("config") or {}
        source_result = source.get("result") or {}
        if (
            source_config.get("scenario_id") != SCENARIO_ID
            or source_config.get("target_version") != TARGET_VERSION
            or source_config.get("interaction_protocol_schema_id")
            != interaction_protocol_schema_id
        ):
            raise ValueError("SQLi checkpoint belongs to an incompatible scenario or interaction protocol")
        filename = "initial.pt" if source_checkpoint_variant == "initial" else "final.pt"
        path = ARTIFACT_ROOT / source_id / "checkpoints" / "joint" / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_digest = (
            source_result.get("initial_checkpoint_sha256")
            if source_checkpoint_variant == "initial"
            else source_result.get("checkpoint_sha256")
        ) or config.get("source_checkpoint_sha256")
        if not isinstance(expected_digest, str) or digest != expected_digest:
            raise ValueError("SQLi checkpoint digest does not match its persisted run record")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _validate_checkpoint_contract(
            payload,
            expected_training_run_id=source_id,
            expected_reward_schema_id=source_config.get("reward_schema_id"),
            expected_interaction_protocol_schema_id=interaction_protocol_schema_id,
        )
        source_versions = (
            source_result.get("initial_policy_versions")
            if source_checkpoint_variant == "initial"
            else source_result.get("policy_versions")
        )
        if isinstance(source_versions, dict) and source_versions != payload.get("policy_versions"):
            raise ValueError("SQLi checkpoint policy versions do not match the run record")
        expected_source_digest = config.get("source_checkpoint_sha256")
        if expected_source_digest and expected_source_digest != digest:
            raise ValueError("SQLi continuation source digest changed after campaign creation")
        checkpoint_format = str(payload.get("format"))
        red.load_model_state(payload["model_state_dicts"]["red"], checkpoint_format)
        blue.load_model_state(payload["model_state_dicts"]["blue"], checkpoint_format)
        red.policy_version = int(payload["policy_versions"]["red"])
        blue.policy_version = int(payload["policy_versions"]["blue"])
        update_counts = payload.get("update_counts") or {}
        red.optimizer_step_count = int(update_counts.get("red", 0))
        blue.optimizer_step_count = int(update_counts.get("blue", 0))
        if mode == "train":
            red.load_optimizer_state(payload["optimizer_state_dicts"]["red"], checkpoint_format)
            blue.load_optimizer_state(payload["optimizer_state_dicts"]["blue"], checkpoint_format)
        rng_states = payload.get("policy_rng_states") or {}
        if isinstance(rng_states.get("red"), dict):
            red.rng.bit_generator.state = rng_states["red"]
        if isinstance(rng_states.get("blue"), dict):
            blue.rng.bit_generator.state = rng_states["blue"]

    # Optimizer state restoration can replace the learning rate from the
    # requested config. Record the rate the actual Adam optimizers will use.
    optimizer_settings = ppo_settings(
        float(red.optimizer.param_groups[0]["lr"]),
        optimization_passes=PPO_OPTIMIZATION_PASSES,
    )

    # Preserve the policy pair as it entered this run before any episode can
    # update it. A resumed run keeps its original initial snapshot unchanged.
    if mode == "train" and not own_checkpoint_loaded and not initial_checkpoint_path.exists():
        _save_checkpoint(
            initial_checkpoint_path,
            training_id=training_id,
            seed=seed,
            episode_count=0,
            red=red,
            blue=blue,
            red_updates=red.optimizer_step_count,
            blue_updates=blue.optimizer_step_count,
            completed_training_updates=0,
            campaign_revision=config.get("campaign_revision"),
            reward_schema_id=reward_schema_id,
            interaction_protocol_schema_id=interaction_protocol_schema_id,
            optimizer_settings=optimizer_settings,
        )

    (config_dir / "run.json").write_text(
        json.dumps(
            {
                "training_run_id": training_id,
                "scenario_id": SCENARIO_ID,
                "target_version": TARGET_VERSION,
                "observation_schema_id": OBSERVATION_SCHEMA,
                "feature_schema_id": FEATURE_SCHEMA,
                "action_schema_id": ACTION_SCHEMA,
                "reward_schema_id": reward_schema_id,
                "interaction_protocol_schema_id": interaction_protocol_schema_id,
                "seed": seed,
                "mode": mode,
                "config": config,
                "ppo_settings": optimizer_settings,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    requested = int(config["episodes"])
    max_steps = int(config["max_steps"])
    training = mode == "train"
    demo_evaluation = config.get("campaign_label") == "classroom_evaluation_v2"
    pilot_evaluation = config.get("campaign_label") == "balanced_pilot_evaluation_v1"
    learning_evaluation = config.get("campaign_label") == "learning_evaluation_v1"
    exposure_experiment = config.get("campaign_label") == "exposure_experiment_v1"
    exposure_evaluation = config.get("campaign_label") == "exposure_experiment_evaluation_v1"
    held_out_evaluation = demo_evaluation or pilot_evaluation or learning_evaluation or exposure_evaluation
    research = config.get("run_kind") == "research"
    episode_summaries: list[dict[str, Any]] = []
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    episode_summaries.append(item)

    if resume_update_count == 0 and own_checkpoint_loaded:
        resume_update_count = sum(
            bool(item.get("joint_update_applied"))
            or (
                item.get("status") == "completed"
                and float(((item.get("optimizer_metrics") or {}).get("red") or {}).get("sample_count", 0)) > 0
                and float(((item.get("optimizer_metrics") or {}).get("blue") or {}).get("sample_count", 0)) > 0
            )
            for item in episode_summaries
        )
    update_count = resume_update_count
    completed_matches = sum(
        item.get("status") == "completed"
        and (not training or item.get("joint_update_applied", True))
        for item in episode_summaries
    )
    global_steps = {
        role: sum(
            int(((item.get("optimizer_metrics") or {}).get(role) or {}).get("sample_count", 0))
            for item in episode_summaries
        )
        for role in ("red", "blue")
    }
    attempts_by_slot: dict[int, int] = {}
    for index, item in enumerate(episode_summaries):
        slot = item.get("requested_episode_index")
        slot_index = int(slot) - 1 if isinstance(slot, int) and slot > 0 else index
        attempts_by_slot[slot_index] = max(
            attempts_by_slot.get(slot_index, 0), int(item.get("attempt", 1))
        )
    attempt_index = max(resume_episode_count, len(episode_summaries))
    candidate_scores: list[float] = []
    comparator_scores: list[float] = []
    stop_requested = False
    source_config = dict((source or {}).get("config") or {})

    if demo_evaluation:
        if source is None:
            raise ValueError("frozen classroom evaluation requires a recorded source run")
        source_result = source.get("result") or {}
        training_seed = int(source_config.get("seed", seed))
        training_attempts = int(source_result.get("episode_summary_count", source_config.get("episodes", 40)))
        evaluation_seeds = _demo_held_out_seeds(training_seed, training_attempts, requested)
    elif pilot_evaluation:
        configured_seeds = config.get("pilot_evaluation_seeds")
        if (
            not isinstance(configured_seeds, list)
            or len(configured_seeds) != requested
            or any(not isinstance(item, int) or item < 0 for item in configured_seeds)
            or len(set(configured_seeds)) != requested
        ):
            raise ValueError("balanced pilot evaluation requires a complete disjoint seed schedule")
        training_seeds = {
            (int(config["seed"]) + index) % (2**31 - 1)
            for index in range(int(config.get("pilot_training_episodes_per_seed", 50)))
        }
        if training_seeds.intersection(configured_seeds):
            raise ValueError("balanced pilot evaluation seeds overlap the training seed schedule")
        evaluation_seeds = configured_seeds
    elif learning_evaluation or exposure_evaluation:
        configured_seeds = config.get(
            "learning_evaluation_seeds" if learning_evaluation else "experiment_episode_seeds"
        )
        if (
            not isinstance(configured_seeds, list)
            or len(configured_seeds) != requested
            or any(not isinstance(item, int) or item < 0 for item in configured_seeds)
            or len(set(configured_seeds)) != requested
        ):
            raise ValueError("checkpoint comparison requires its complete matched seed schedule")
        evaluation_seeds = configured_seeds
    else:
        evaluation_seeds = [(seed + index) % (2**31 - 1) for index in range(requested)]

    while completed_matches < requested and attempt_index < max(resume_episode_count, len(episode_summaries)) + requested * 3:
        stop_deadline = (
            config.get("pilot_training_deadline")
            if training
            else config.get("pilot_evaluation_deadline")
        )
        if await _training_stop_requested(client, training_id, stop_deadline):
            stop_requested = True
            break
        slot = update_count if training else completed_matches
        if slot >= requested:
            break
        attempt = attempts_by_slot.get(slot, 0) + 1
        if attempt > 3:
            break
        configured_episode_seeds = (
            config.get("experiment_episode_seeds")
            if exposure_experiment or exposure_evaluation
            else config.get("learning_evaluation_seeds")
            if learning_evaluation
            else None
        )
        episode_seed = (
            int(configured_episode_seeds[slot])
            if isinstance(configured_episode_seeds, list) and slot < len(configured_episode_seeds)
            else evaluation_seeds[slot]
        )
        candidate_episode_id = uuid4()
        candidate = SqliEpisodeRunner(
            client,
            training_uuid,
            candidate_episode_id,
            seed=episode_seed,
            max_steps=max_steps,
            red_policy=red,
            blue_policy=blue,
            mode=mode,
            policy_run_id=training_id,
            episode_kind=("training" if training else "held_out_evaluation" if held_out_evaluation else "candidate"),
            reward_schema_id=reward_schema_id,
            checkpoint_variant=str(config.get("checkpoint_variant", "final")),
            vulnerable_configuration_probability=float(
                1.0
                if config.get("vulnerable_configuration_probability") is None
                else config["vulnerable_configuration_probability"]
            ),
        )
        candidate_result = await candidate.run()
        candidate_summary = candidate_result["summary"]
        entry: dict[str, Any] = dict(candidate_summary)
        entry.update(
            {
                "requested_episode_index": slot + 1,
                "attempt": attempt,
                "episode_index": attempt_index + 1,
            }
        )

        if training:
            trace = candidate_result["trace"]
            red_transitions = _transition_batch(trace, "red", red)
            blue_transitions = _transition_batch(trace, "blue", blue)
            joint_update = bool(
                candidate_result["status"] == "completed" and red_transitions and blue_transitions
            )
            if not joint_update:
                red_transitions = []
                blue_transitions = []

            red_reference = (
                np.asarray(trace[0]["red_observation"].features, dtype=np.float32)
                if trace
                else np.zeros(len(FEATURES["red"]), dtype=np.float32)
            )
            blue_reference = (
                np.asarray(trace[0]["blue_observation"].features, dtype=np.float32)
                if trace
                else np.zeros(len(FEATURES["blue"]), dtype=np.float32)
            )
            red_context = (
                np.asarray(trace[0]["red_sample"].recurrent_state, dtype=np.float32)
                if trace
                else np.zeros(red.hidden_size, dtype=np.float32)
            )
            blue_context = (
                np.asarray(trace[0]["blue_sample"].recurrent_state, dtype=np.float32)
                if trace
                else np.zeros(blue.hidden_size, dtype=np.float32)
            )
            red_before = red.probabilities(red_reference, red_context)
            blue_before = blue.probabilities(blue_reference, blue_context)
            red_metrics = red.update(red_transitions, passes=PPO_OPTIMIZATION_PASSES)
            blue_metrics = blue.update(blue_transitions, passes=PPO_OPTIMIZATION_PASSES)
            if joint_update:
                update_count += 1
            digest = _save_checkpoint(
                checkpoint_path,
                training_id=training_id,
                seed=seed,
                episode_count=attempt_index + 1,
                red=red,
                blue=blue,
                red_updates=red.optimizer_step_count,
                blue_updates=blue.optimizer_step_count,
                completed_training_updates=update_count,
                campaign_revision=config.get("campaign_revision"),
                reward_schema_id=reward_schema_id,
                interaction_protocol_schema_id=interaction_protocol_schema_id,
                optimizer_settings=optimizer_settings,
            )
            red_after = red.probabilities(red_reference, red_context)
            blue_after = blue.probabilities(blue_reference, blue_context)
            role_updates = (
                ("red", red_before, red_after, red_reference, red_metrics, red_transitions),
                ("blue", blue_before, blue_after, blue_reference, blue_metrics, blue_transitions),
            )
            contextual: dict[str, dict[str, Any]] = {}
            for role, _before, _after, _reference, metrics, transitions in role_updates:
                global_steps[role] += len(transitions)
                contextual[role] = {
                    **metrics,
                    "episode_index": float(attempt_index + 1),
                    "requested_episode_index": float(slot + 1),
                    "attempt": float(attempt),
                    "accepted_transitions": float(len(transitions)),
                    "target_version": TARGET_VERSION,
                    "observation_schema_id": OBSERVATION_SCHEMA,
                    "action_schema_id": ACTION_SCHEMA,
                    "reward_schema_id": reward_schema_id,
                    "ppo_settings": optimizer_settings,
                }
            if joint_update:
                for role, _before, _after, _reference, _metrics, _transitions in role_updates:
                    await _record_update(
                        client,
                        training_id,
                        role,
                        update_count,
                        global_steps[role],
                        contextual[role],
                        digest,
                    )
            for role, before, after, reference, metrics, _transitions in role_updates:
                update_policy = red if role == "red" else blue
                await candidate._event(
                    "sqli_policy_probe",
                    {
                        "role": role,
                        "policy_id": f"{role}-ppo-{training_id}",
                        "reference_observation": reference.tolist(),
                        "before_action_probabilities": [list(head) for head in before],
                        "after_action_probabilities": [list(head) for head in after],
                        "policy_version_before": int(metrics["policy_version_before"]),
                        "policy_version_after": int(metrics["policy_version_after"]),
                        "same_reference_observation": True,
                    },
                )
                await candidate._event(
                    "sqli_ppo_update",
                    {
                        "role": role,
                        "policy_id": f"{role}-ppo-{training_id}",
                        "update_index": update_count if joint_update else None,
                        "episode_index": attempt_index + 1,
                        "requested_episode_index": slot + 1,
                        "attempt": attempt,
                        "global_step": global_steps[role],
                        "metrics": contextual[role],
                        "checkpoint_sha256": digest,
                        "policy_version_before": int(metrics["policy_version_before"]),
                        "policy_version": int(metrics["policy_version_after"]),
                        "updated": joint_update,
                        "update_status": "applied" if joint_update else "skipped_incomplete_joint_episode",
                        "algorithm": "PPO_clipped_surrogate_GAE",
                        "event_schema": "sqli.ppo_update.v2",
                        "optimization_passes": int(metrics.get("optimizer_steps", 0)),
                        "ppo_settings": optimizer_settings,
                        "transition_diagnostics": update_policy.last_update_diagnostics,
                    },
                )
            entry["optimizer_metrics"] = {"red": red_metrics, "blue": blue_metrics}
            entry["joint_update_applied"] = joint_update
            entry["policy_versions"] = {"red": red.policy_version, "blue": blue.policy_version}
            entry["checkpoint_sha256"] = digest
            await candidate._event(
                "sqli_episode_outcome",
                {
                    "schema_id": "sqli.episode_outcome.v2",
                    "phase": "training",
                    **candidate_summary,
                    "episode_index": attempt_index + 1,
                    "requested_episode_index": slot + 1,
                    "attempt": attempt,
                    "joint_update_applied": joint_update,
                    "checkpoint_run_id": training_id,
                    "checkpoint_sha256": digest,
                    "policy_versions": entry["policy_versions"],
                    "update_metrics": {"red": red_metrics, "blue": blue_metrics},
                    "ppo_update_count": update_count,
                },
            )
            attempts_by_slot[slot] = attempt
        else:
            comparator_summary: dict[str, Any] | None = None
            comparator_result: dict[str, Any] | None = None
            if not held_out_evaluation:
                comparator_id = uuid4()
                comparator_blue = SqliPPOPolicy(
                    "blue", seed=seed + attempt_index + 10000, learning_rate=learning_rate
                )
                comparator = SqliEpisodeRunner(
                    client,
                    training_uuid,
                    comparator_id,
                    seed=episode_seed,
                    max_steps=max_steps,
                    red_policy=red,
                    blue_policy=comparator_blue,
                    mode="evaluate",
                    comparator=True,
                    policy_run_id=str(source_id or training_id),
                    episode_kind="evaluation_comparator",
                    reward_schema_id=reward_schema_id,
                )
                comparator_result = await comparator.run()
                comparator_summary = comparator_result["summary"]
                entry["paired_comparator_episode_id"] = str(comparator_id)
                entry["paired_comparator"] = comparator_summary
                entry["comparator_label"] = "No-defense control comparator (evaluation only)"
                if candidate_result["status"] == "completed" and comparator_result["status"] == "completed":
                    candidate_score = (
                        float(candidate_summary["verified_prevention_rate"] or 0.0)
                        + float(candidate_summary["legitimate_query_availability"] or 0.0)
                        - float(candidate_summary["unauthorized_read_rate"] or 0.0)
                        - float(candidate_summary["false_positive_rate"] or 0.0)
                    )
                    comparator_score = (
                        float(comparator_summary["verified_prevention_rate"] or 0.0)
                        + float(comparator_summary["legitimate_query_availability"] or 0.0)
                        - float(comparator_summary["unauthorized_read_rate"] or 0.0)
                        - float(comparator_summary["false_positive_rate"] or 0.0)
                    )
                    candidate_scores.append(candidate_score)
                    comparator_scores.append(comparator_score)
            phase = "held_out_evaluation" if held_out_evaluation else "evaluation"
            await candidate._event(
                "sqli_episode_outcome",
                {
                    "schema_id": "sqli.episode_outcome.v2",
                    "phase": phase,
                    **candidate_summary,
                    "episode_index": attempt_index + 1,
                    "requested_episode_index": slot + 1,
                    "attempt": attempt,
                    "checkpoint_run_id": str(source_id or training_id),
                    "checkpoint_sha256": config.get("source_checkpoint_sha256"),
                    "update_metrics": None,
                    "pair_frozen": True,
                    "scripted_opponents": False if held_out_evaluation else None,
                },
            )
            attempts_by_slot[slot] = attempt

        episode_summaries.append(entry)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        attempt_index += 1
        accepted_match = candidate_summary.get("status") == "completed"
        if not training and not held_out_evaluation:
            accepted_match = accepted_match and bool(comparator_summary and comparator_summary.get("status") == "completed")
        if training:
            accepted_match = bool(entry.get("joint_update_applied"))
        completed_matches += int(accepted_match)
        if not accepted_match and attempts_by_slot.get(slot, 0) >= 3:
            break

    if training:
        if checkpoint_path.exists():
            checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            checkpoint_metadata = {
                "schema_id": "sqli.checkpoint_metadata.v3",
                "model_architecture": "factorized_gru_actor_critic.v1",
                "training_run_id": training_id,
                "format": "sqli.joint_ppo.v4",
                "target_version": TARGET_VERSION,
                "reward_schema_id": reward_schema_id,
                "interaction_protocol_schema_id": interaction_protocol_schema_id,
                "completed_training_updates": update_count,
                "policy_versions": {"red": red.policy_version, "blue": blue.policy_version},
                "optimization_passes": PPO_OPTIMIZATION_PASSES,
                "campaign_revision": config.get("campaign_revision"),
                "checkpoint_sha256": checkpoint_digest,
                "ppo_settings": optimizer_settings,
            }
        else:
            checkpoint_digest = None
            checkpoint_metadata = None
        initial_checkpoint_digest = (
            hashlib.sha256(initial_checkpoint_path.read_bytes()).hexdigest()
            if initial_checkpoint_path.exists()
            else None
        )
        if initial_checkpoint_digest is not None:
            initial_payload = torch.load(
                initial_checkpoint_path, map_location="cpu", weights_only=True
            )
            initial_policy_versions = initial_payload.get("policy_versions")
        else:
            initial_policy_versions = None
    else:
        checkpoint_digest = config.get("source_checkpoint_sha256")
        checkpoint_metadata = None
        initial_checkpoint_digest = None
        initial_policy_versions = None

    training_complete = training and update_count >= requested
    evaluation_complete = not training and completed_matches >= requested
    result: dict[str, Any] = {
        "scenario_id": SCENARIO_ID,
        "target_version": TARGET_VERSION,
        "observation_schema_id": OBSERVATION_SCHEMA,
        "feature_schema_id": FEATURE_SCHEMA,
        "action_schema_id": ACTION_SCHEMA,
        "reward_schema_id": reward_schema_id,
        "interaction_protocol_schema_id": interaction_protocol_schema_id,
        "episodes": episode_summaries[-100:] if config.get("campaign_label") else episode_summaries,
        "episode_summary_count": len(episode_summaries),
        "completed_episodes": update_count if training else completed_matches,
        "completed_training_episodes": update_count if training else 0,
        "valid_ppo_updated_episodes": update_count if training else 0,
        "requested_episodes": requested,
        "independent_training_seeds": 1 if training else None,
        "episode_updates": update_count,
        "policy_versions": {"red": red.policy_version, "blue": blue.policy_version},
        "checkpoint": str(checkpoint_path) if training and checkpoint_digest else None,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_metadata": checkpoint_metadata,
        "initial_checkpoint_sha256": initial_checkpoint_digest,
        "initial_policy_versions": initial_policy_versions,
        "checkpoint_variant": config.get("checkpoint_variant", "final"),
        "ppo_settings": optimizer_settings if training else None,
        "demonstration_label": "single_seed_classroom_demo"
        if config.get("campaign_label") == "live_demo_v1"
        else None,
        "training_campaign_complete": training_complete,
        "evaluation_complete": evaluation_complete if held_out_evaluation else None,
        "research_criteria_met": training_complete if training else evaluation_complete if held_out_evaluation else (
            not research or len(candidate_scores) >= requested
        ),
        "evidence_level": "real_http_target" if episode_summaries else "no_episode_completed",
        "stop_requested": stop_requested,
    }

    if held_out_evaluation:
        completed = [item for item in episode_summaries if item.get("status") == "completed"]
        red_wins = sum(item.get("winner") == "red" for item in completed)
        blue_wins = sum(item.get("winner") == "blue" for item in completed)
        def value_mean(key: str) -> float | None:
            return (
                float(np.mean([float(item[key]) for item in completed]))
                if completed
                else None
            )

        result["evaluation"] = {
            "schema_id": "sqli.held_out_evaluation.v2",
            "phase": "held_out_evaluation",
            "status": "completed" if len(completed) == requested else "incomplete",
            "requested_matches": requested,
            "completed_matches": len(completed),
            "attempted_matches": len(episode_summaries),
            "incomplete_matches": max(0, requested - len(completed)),
            "red_wins": red_wins,
            "blue_wins": blue_wins,
            "undecided_matches": len(completed) - red_wins - blue_wins,
            "red_win_rate": red_wins / len(completed) if completed else None,
            "blue_win_rate": blue_wins / len(completed) if completed else None,
            "red_win_rate_wilson_95": list(_wilson_interval(red_wins, len(completed)))
            if completed
            else None,
            "blue_win_rate_wilson_95": list(_wilson_interval(blue_wins, len(completed)))
            if completed
            else None,
            "verified_disclosures": sum(bool(item.get("verified_exposure")) for item in completed),
            "verified_exposure_rate": value_mean("verified_exposure"),
            "verified_prevention_rate": value_mean("verified_prevention_rate"),
            "legitimate_service_availability": value_mean("legitimate_query_availability"),
            "mean_red_return": value_mean("red_return"),
            "mean_blue_return": value_mean("blue_return"),
            "checkpoint_run_id": source_id,
            "checkpoint_sha256": checkpoint_digest,
            "policy_versions": {"red": red.policy_version, "blue": blue.policy_version},
            "pair_frozen": True,
            "scripted_opponents": False,
            "evidence_source": "correlated target HTTP evidence",
            "training_seeds": [
                (int(source_config.get("seed", seed)) + index) % (2**31 - 1)
                for index in range(int(source_config.get("episodes", 50)))
            ] if pilot_evaluation else None,
            "evaluation_seeds": evaluation_seeds,
            "uncertainty_method": "Wilson score 95% intervals for Red and Blue win rates",
            "interaction_protocol_schema_id": interaction_protocol_schema_id,
            "reward_schema_id": reward_schema_id,
        }
        (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        (run_dir / "evaluation" / "summary.json").write_text(
            json.dumps(result["evaluation"], sort_keys=True, indent=2) + "\n"
        )
    elif not training:
        interval = _bootstrap_interval(candidate_scores, comparator_scores, seed)
        source_result = (source or {}).get("result") or {}
        independent_seeds = int(source_result.get("independent_training_seeds", 1))
        availability = [
            float(item["legitimate_query_availability"])
            for item in episode_summaries
            if isinstance(item.get("legitimate_query_availability"), (int, float))
        ]
        supported = (
            len(candidate_scores) >= 100
            and interval is not None
            and interval[1] > 0.0
            and independent_seeds >= 5
            and bool(availability)
            and all(value == 1.0 for value in availability)
        )
        result["evaluation"] = {
            "schema_id": "sqli.evaluation.v1",
            "target_version": TARGET_VERSION,
            "checkpoint_run_id": source_id,
            "comparator": "No-defense control comparator (evaluation only)",
            "paired_episodes": len(candidate_scores),
            "requested_pairs": requested,
            "candidate_mean_net_security_score": float(np.mean(candidate_scores)) if candidate_scores else None,
            "comparator_mean_net_security_score": float(np.mean(comparator_scores)) if comparator_scores else None,
            "paired_mean_difference": interval[0] if interval else None,
            "paired_bootstrap_ci_95": list(interval[1:]) if interval else None,
            "training_seed_count": independent_seeds,
            "required_training_seeds": 5,
            "required_held_out_pairs": 100,
            "availability_criterion_met": bool(availability) and all(value == 1.0 for value in availability),
            "supported_improvement": supported,
            "claim": "supported improvement" if supported else "no supported improvement",
        }
    return result


async def run_sqli_challenge(client: httpx.AsyncClient, job: dict[str, Any]) -> dict[str, Any]:
    """Run the verified balanced-pilot pair against one volatile target phrase."""
    challenge_id = UUID(str(job["challenge_id"]))
    run_id = UUID(str(job["run_id"]))
    episode_id = UUID(str(job["episode_id"]))
    checkpoint_run_id = str(job["training_run_id"])
    seed = int(job["seed"])
    max_steps = min(32, max(1, int(job.get("max_steps", 32))))
    source_response = await client.get(f"/range/training-runs/{checkpoint_run_id}")
    source_response.raise_for_status()
    source = source_response.json()
    source_config = source.get("config") or {}
    source_result = source.get("result") or {}
    expected_digest = job.get("checkpoint_sha256")
    expected_versions = job.get("checkpoint_policy_versions")
    if (
        source.get("status") not in {"running", "completed", "stopped"}
        or source.get("role") != "joint"
        or source_config.get("scenario_id") != SCENARIO_ID
        or source_config.get("campaign_revision") not in {"balanced_adversarial_ppo4_v3", "balanced_adversarial_gru_ppo4_v4"}
        or source_config.get("episodes") != 50
        or source_config.get("reward_schema_id") != REWARD_SCHEMA
        or source_config.get("interaction_protocol_schema_id") != INTERACTION_PROTOCOL_SCHEMA
        or not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or not isinstance(expected_versions, dict)
    ):
        raise ValueError("challenge checkpoint is not a verified 50-update balanced-pilot campaign")
    checkpoint_path = ARTIFACT_ROOT / checkpoint_run_id / "checkpoints" / "joint" / "final.pt"
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != expected_digest or (
        source_result.get("checkpoint_sha256") and digest != source_result.get("checkpoint_sha256")
    ):
        raise ValueError("challenge checkpoint digest does not match its run record")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint_contract(
        payload,
        expected_training_run_id=checkpoint_run_id,
        expected_reward_schema_id=REWARD_SCHEMA,
        expected_interaction_protocol_schema_id=INTERACTION_PROTOCOL_SCHEMA,
    )
    if (
        int(payload.get("completed_training_updates", 0)) < 50
        or payload.get("campaign_revision") not in {"balanced_adversarial_ppo4_v3", "balanced_adversarial_gru_ppo4_v4"}
        or payload.get("policy_versions") != expected_versions
    ):
        raise ValueError("challenge checkpoint policy metadata does not verify the 50 updates")
    red = SqliPPOPolicy("red", seed=seed, learning_rate=3e-4)
    blue = SqliPPOPolicy("blue", seed=seed + 1, learning_rate=3e-4)
    red.load_model_state(payload["model_state_dicts"]["red"], str(payload.get("format")))
    blue.load_model_state(payload["model_state_dicts"]["blue"], str(payload.get("format")))
    red.policy_version = int(payload["policy_versions"]["red"])
    blue.policy_version = int(payload["policy_versions"]["blue"])
    runner = SqliEpisodeRunner(
        client,
        run_id,
        episode_id,
        seed=seed,
        max_steps=max_steps,
        red_policy=red,
        blue_policy=blue,
        mode="evaluate",
        challenge_id=challenge_id,
        policy_run_id=checkpoint_run_id,
        episode_kind="challenge",
        reward_schema_id=REWARD_SCHEMA,
    )
    result = await runner.run()
    summary = result["summary"]
    outcome = {
        "schema_id": "sqli.episode_outcome.v2",
        "phase": "challenge",
        **summary,
        "checkpoint_run_id": checkpoint_run_id,
        "checkpoint_sha256": digest,
        "policy_versions": {"red": red.policy_version, "blue": blue.policy_version},
        "update_metrics": None,
    }
    await runner._event("sqli_episode_outcome", outcome)
    return {"status": summary["status"], "summary": outcome}


async def _training_stop_requested(
    client: httpx.AsyncClient, training_id: str, stop_deadline: str | None = None
) -> bool:
    heartbeat = await client.post(f"/range/training-runs/{training_id}/heartbeat")
    heartbeat.raise_for_status()
    record = await client.get(f"/range/training-runs/{training_id}")
    record.raise_for_status()
    current = record.json()
    if current.get("stop_requested"):
        return True
    if stop_deadline:
        try:
            deadline = datetime.fromisoformat(stop_deadline)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            if datetime.now(UTC) >= deadline.astimezone(UTC):
                response = await client.post(f"/range/training-runs/{training_id}/stop")
                response.raise_for_status()
                return True
        except (TypeError, ValueError):
            raise ValueError("pilot stop deadline is not a valid ISO timestamp") from None
    return False


def _demo_held_out_seeds(seed: int, training_episode_count: int, count: int) -> list[int]:
    modulus = 2**31 - 1
    training = {(seed + index) % modulus for index in range(training_episode_count)}
    candidate = (seed + training_episode_count + 10_000) % modulus
    result: list[int] = []
    while len(result) < count:
        if candidate not in training and candidate not in result:
            result.append(candidate)
        candidate = (candidate + 1) % modulus
    return result


async def _run_demo_held_out_evaluation(
    client: httpx.AsyncClient,
    *,
    run_id: UUID,
    training_id: str,
    seed: int,
    training_episode_count: int,
    max_steps: int,
    red: SqliPPOPolicy,
    blue: SqliPPOPolicy,
    metrics_dir: Path,
) -> dict[str, Any]:
    """Run 500 new deterministic matches with the trained pair frozen."""
    count = 500
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "held_out.jsonl"
    summaries: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    summaries.append(item)
    seeds = _demo_held_out_seeds(seed, training_episode_count, count)
    for match_index, match_seed in enumerate(seeds):
        prior = [item for item in summaries if int(item.get("seed", -1)) == match_seed]
        if any(item.get("status") == "completed" for item in prior):
            continue
        attempts = len(prior)
        while attempts < 2:
            if await _training_stop_requested(client, training_id):
                break
            runner = SqliEpisodeRunner(
                client,
                run_id,
                uuid4(),
                seed=match_seed,
                max_steps=max_steps,
                red_policy=red,
                blue_policy=blue,
                mode="evaluate",
                policy_run_id=training_id,
                episode_kind="held_out_evaluation",
            )
            result = await runner.run()
            summary = {
                **result["summary"],
                "match_index": match_index + 1,
                "attempt": attempts + 1,
                "checkpoint_run_id": training_id,
                "frozen_policy_versions": {
                    "red": red.policy_version,
                    "blue": blue.policy_version,
                },
            }
            await runner._event(
                "sqli_episode_outcome",
                {
                    "schema_id": "sqli.episode_outcome.v2",
                    "phase": "held_out_evaluation",
                    **summary,
                    "pair_frozen": True,
                    "scripted_opponents": False,
                    "update_metrics": None,
                },
            )
            summaries.append(summary)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            attempts += 1
            if result["status"] == "completed":
                break

    completed_by_seed = {
        int(item["seed"]): item
        for item in summaries
        if item.get("status") == "completed"
    }
    completed = [completed_by_seed[item] for item in seeds if item in completed_by_seed]
    red_wins = sum(item.get("winner") == "red" for item in completed)
    blue_wins = sum(item.get("winner") == "blue" for item in completed)
    verified_exposures = [float(bool(item.get("verified_exposure"))) for item in completed]
    prevention = [
        float(item["verified_prevention_rate"])
        for item in completed
        if isinstance(item.get("verified_prevention_rate"), (int, float))
    ]
    availability = [
        float(item["legitimate_query_availability"])
        for item in completed
        if isinstance(item.get("legitimate_query_availability"), (int, float))
    ]
    return {
        "schema_id": "sqli.held_out_evaluation.v1",
        "phase": "held_out_evaluation",
        "status": "completed" if len(completed) == count else "incomplete",
        "requested_matches": count,
        "completed_matches": len(completed),
        "attempted_matches": len(summaries),
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "undecided_matches": len(completed) - red_wins - blue_wins,
        "red_win_rate": red_wins / len(completed) if completed else None,
        "blue_win_rate": blue_wins / len(completed) if completed else None,
        "verified_exposure_rate": float(np.mean(verified_exposures)) if completed else None,
        "verified_prevention_rate": float(np.mean(prevention)) if prevention else None,
        "legitimate_service_availability": float(np.mean(availability)) if availability else None,
        "mean_red_return": float(np.mean([item["red_return"] for item in completed]))
        if completed
        else None,
        "mean_blue_return": float(np.mean([item["blue_return"] for item in completed]))
        if completed
        else None,
        "evaluation_seed_start": seeds[0],
        "training_seed_count": 1,
        "claim": "single-seed demonstration; not a multi-seed research claim",
        "checkpoint_run_id": training_id,
        "policy_versions": {"red": red.policy_version, "blue": blue.policy_version},
        "pair_frozen": True,
        "scripted_opponents": False,
        "evidence_source": "correlated target HTTP evidence",
    }
