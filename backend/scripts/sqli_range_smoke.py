"""Verify the SQLi lab's real HTTP behavior through its fixed executor.

This acceptance probe uses a named verifier action and never enters PPO
training. It checks the isolated target before and after a target control.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, cast
from uuid import uuid4

import httpx

from violet.sqli_lab.contracts import SqliRequestAction

API_URL = os.getenv("VIOLET_API_URL", "http://127.0.0.1:8000").rstrip("/")
TRAINER_TOKEN = os.getenv("VIOLET_TRAINER_TOKEN", "")
ATTACK_ACTION = {
    "schema_id": "sqli.composed_action.v1",
    "field": "handle",
    "base_term": "rhea",
    "close_quote": True,
    "connector": "OR",
    "predicate": "tautology",
    "comment": True,
}
LEGITIMATE_ACTION = {
    "schema_id": "sqli.composed_action.v1",
    "field": "handle",
    "base_term": "rhea",
    "close_quote": False,
    "connector": "none",
    "predicate": "none",
    "comment": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


async def _event(
    client: httpx.AsyncClient,
    run_id: str,
    episode_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    response = await client.post(
        "/range/sqli/events",
        json={
            "event_id": str(uuid4()),
            "run_id": run_id,
            "episode_id": episode_id,
            "seed": 93017,
            "event_type": event_type,
            "payload": payload,
        },
    )
    response.raise_for_status()


async def _control(
    client: httpx.AsyncClient, run_id: str, episode_id: str, action: str
) -> dict[str, Any]:
    response = await client.post(
        "/range/sqli/controls",
        json={"run_id": run_id, "episode_id": episode_id, "action": action},
    )
    response.raise_for_status()
    command_id = response.json()["command_id"]
    for _ in range(80):
        result = await client.get(f"/range/sqli/controls/{command_id}")
        result.raise_for_status()
        payload = result.json()
        if payload.get("status") != "queued":
            await _event(
                client,
                run_id,
                episode_id,
                "sqli_control_acknowledged",
                {"command_id": command_id, "action": action, "acknowledgement": payload},
            )
            return cast("dict[str, Any]", payload)
        await asyncio.sleep(0.05)
    raise RuntimeError(f"control {action} did not receive an acknowledgement")


async def _request(
    client: httpx.AsyncClient,
    run_id: str,
    episode_id: str,
    step_id: int,
    action: dict[str, Any],
    purpose: str,
) -> dict[str, Any]:
    decision_id = str(uuid4())
    response = await client.post(
        "/range/sqli/jobs",
        json={
            "run_id": run_id,
            "episode_id": episode_id,
            "decision_id": decision_id,
            "step_id": step_id,
            "action": action,
            "purpose": purpose,
        },
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    for _ in range(160):
        result_response = await client.get(f"/range/sqli/jobs/{job_id}")
        result_response.raise_for_status()
        result = result_response.json()
        if result.get("status") != "queued":
            await _event(
                client,
                run_id,
                episode_id,
                "sqli_target_response",
                {
                    "purpose": purpose,
                    "step_id": step_id,
                    "job_id": job_id,
                    "decision_id": decision_id,
                    "request_id": result.get("request_id"),
                    "status": result.get("status"),
                    "actor_ref": result.get("actor_ref"),
                    "http_status": result.get("http_status"),
                    "target_event_id": result.get("target_event_id"),
                    "request": result.get("request", {}),
                    "response": result.get("response", {}),
                    "telemetry": result.get("telemetry", {}),
                    "duration_ms": result.get("duration_ms"),
                    "evidence_status": (result.get("telemetry") or {}).get("evidence_status"),
                    "acceptance_smoke": True,
                },
            )
            return cast("dict[str, Any]", result)
        await asyncio.sleep(0.05)
    raise RuntimeError(f"executor job {job_id} did not finish")


async def _verify_ppo_episode(client: httpx.AsyncClient) -> dict[str, Any]:
    create = await client.post(
        "/range/training-runs",
        json={
            "role": "joint",
            "mode": "train",
            "run_kind": "teaching",
            "scenario_id": "sqli_lab_v1",
            "episodes": 1,
            "max_steps": 2,
            "seed": 93018,
            "opponent": "none",
        },
    )
    create.raise_for_status()
    training_run_id = str(create.json()["training_run_id"])
    deadline = asyncio.get_running_loop().time() + 180
    training_run: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/range/training-runs/{training_run_id}")
        response.raise_for_status()
        training_run = response.json()
        if training_run.get("status") in {"completed", "failed", "stopped"}:
            break
        await asyncio.sleep(0.5)
    require(
        training_run.get("status") == "completed",
        f"PPO teaching episode did not complete: {training_run}",
    )
    episodes = await client.get(f"/range/runs/{training_run_id}/episodes?limit=10")
    episodes.raise_for_status()
    episode_items = episodes.json().get("items", [])
    require(
        len(episode_items) == 1,
        f"PPO training did not persist exactly one episode: {episode_items}",
    )
    episode_id = str(episode_items[0]["episode_id"])
    timeline = await client.get(
        f"/range/runs/{training_run_id}/episodes/{episode_id}/timeline?limit=500"
    )
    timeline.raise_for_status()
    events = timeline.json().get("items", [])
    red_decisions = [event for event in events if event["event_type"] == "sqli_red_decision"]
    blue_decisions = [
        event
        for event in events
        if event["event_type"] == "sqli_blue_decision"
        and (event.get("payload") or {}).get("phase") == "policy_decision"
    ]
    policy_requests = [
        event
        for event in events
        if event["event_type"] == "sqli_target_response"
        and (event.get("payload") or {}).get("phase") == "request_submitted"
        and (event.get("payload") or {}).get("purpose") == "policy"
    ]
    require(len(red_decisions) == 2, f"expected two PPO Red decisions, found {len(red_decisions)}")
    require(
        len(blue_decisions) == 2, f"expected two PPO Blue decisions, found {len(blue_decisions)}"
    )
    require(
        len(policy_requests) == 2,
        f"expected two policy-selected real HTTP requests, found {len(policy_requests)}",
    )
    require(
        red_decisions[0]["payload"]["policy_id"].startswith("red-ppo-"),
        "Red decisions were not attributed to the PPO policy",
    )
    first_observation = red_decisions[0]["payload"]["observation"]["features"]
    second_observation = red_decisions[1]["payload"]["observation"]["features"]
    require(
        first_observation[2] == 0, "Red's initial state unexpectedly contained an HTTP response"
    )
    require(second_observation[2] > 0, "Red's later state omitted the preceding HTTP status")
    first_distributions = red_decisions[0]["payload"]["component_distributions"]
    second_distributions = red_decisions[1]["payload"]["component_distributions"]
    require(
        any(
            any(
                abs(float(left) - float(right)) > 1e-7
                for left, right in zip(first, second, strict=True)
            )
            for first, second in zip(first_distributions, second_distributions, strict=True)
        ),
        "Red action probabilities did not respond to the updated observation",
    )
    for decision in red_decisions:
        payload = decision["payload"]
        request = next(
            (
                item["payload"]
                for item in policy_requests
                if item["payload"].get("decision_id") == payload["decision_id"]
            ),
            None,
        )
        if not isinstance(request, dict):
            raise RuntimeError("PPO Red decision has no correlated executor request")
        composed = SqliRequestAction.model_validate(payload["action"]).compose_value()
        require(
            request.get("components") == payload["components"],
            "executor request components differ from the PPO decision",
        )
        require(
            request.get("composed_query_value") == composed,
            "executor did not compose the PPO-selected request parts",
        )
        require(len(composed) <= 96, "PPO-selected request exceeded the target input limit")

    transitions = [event for event in events if event["event_type"] == "sqli_transition"]
    updates = [event for event in events if event["event_type"] == "sqli_ppo_update"]
    probes = [event for event in events if event["event_type"] == "sqli_policy_probe"]
    require(len(probes) == 2, f"expected one before/after probe per role, found {len(probes)}")
    for role in ("red", "blue"):
        decisions = red_decisions if role == "red" else blue_decisions
        require(
            len({event["payload"].get("policy_version") for event in decisions}) == 1,
            f"the {role} policy version changed during the episode",
        )
        accepted = [
            event
            for event in transitions
            if event["payload"].get("role") == role and event["payload"].get("accepted") is True
        ]
        role_update = next(
            (event for event in updates if event["payload"].get("role") == role), None
        )
        require(bool(accepted), f"no accepted persisted {role} transitions were used")
        if role_update is None:
            raise RuntimeError(f"no persisted {role} episode-boundary update was recorded")
        require(
            role_update["payload"].get("updated") is True, f"{role} policy update was not applied"
        )
        require(
            role_update["payload"]["metrics"].get("sample_count", 0) > 0,
            f"{role} PPO update did not use accepted transitions",
        )
        require(
            max(event["seq"] for event in accepted) < role_update["seq"],
            f"{role} PPO update was persisted before its accepted transitions",
        )
        probe = next(
            (event["payload"] for event in probes if event["payload"].get("role") == role), None
        )
        if not isinstance(probe, dict):
            raise RuntimeError(
                f"no persisted before/after probability probe was recorded for {role}"
            )
        require(
            probe.get("same_reference_observation") is True
            and probe.get("reference_observation") is not None,
            f"{role} probability probe did not reuse its reference observation",
        )
        require(
            probe.get("policy_version_after", 0) > probe.get("policy_version_before", 0),
            f"{role} probability probe does not show an episode-boundary policy update",
        )
    update_records: dict[str, list[dict[str, Any]]] = {}
    for role in ("red", "blue"):
        response = await client.get(f"/range/training-runs/{training_run_id}/updates?role={role}")
        response.raise_for_status()
        update_records[role] = response.json().get("items", [])
        require(
            len(update_records[role]) == 1,
            f"the {role} optimizer update was not persisted in the run record",
        )
    return {
        "training_run_id": training_run_id,
        "episode_id": episode_id,
        "red_policy_requests": len(policy_requests),
        "blue_policy_decisions": len(blue_decisions),
        "accepted_red_transitions": sum(
            event["payload"].get("role") == "red" and event["payload"].get("accepted") is True
            for event in transitions
        ),
        "accepted_blue_transitions": sum(
            event["payload"].get("role") == "blue" and event["payload"].get("accepted") is True
            for event in transitions
        ),
        "role_updates_persisted": {role: len(records) for role, records in update_records.items()},
    }


async def run() -> dict[str, Any]:
    require(bool(TRAINER_TOKEN), "VIOLET_TRAINER_TOKEN is required for the executor smoke")
    headers = {"x-violet-trainer-token": TRAINER_TOKEN, "x-violet-worker-id": "sqli-smoke"}
    run_id, episode_id = str(uuid4()), str(uuid4())
    async with httpx.AsyncClient(base_url=API_URL, headers=headers, timeout=8.0) as client:
        status = (await client.get("/range/sqli/target/status")).json()
        require(status.get("status") == "available", f"SQLi target is unavailable: {status}")
        await _event(
            client,
            run_id,
            episode_id,
            "sqli_episode_started",
            {
                "scenario_id": "sqli_lab_v1",
                "target_version": "sqli_target_v1",
                "seed": 93017,
                "acceptance_smoke": True,
                "policy_training": False,
            },
        )
        reset = await _control(client, run_id, episode_id, "RESET")
        require(reset.get("status") == "applied", f"target reset failed: {reset}")
        before = await _request(client, run_id, episode_id, 0, ATTACK_ACTION, "verifier_attack")
        before_rows = (before.get("response") or {}).get("rows", [])
        before_unauthorized = [row for row in before_rows if row.get("owner_ref") != "guest"]
        require(before.get("status") == "completed", f"pre-defense request failed: {before}")
        require(
            (before.get("telemetry") or {}).get("evidence_status") == "verified",
            f"pre-defense evidence was not correlated: {before}",
        )
        require(
            bool(before_unauthorized),
            "the intended vulnerable request returned no seeded foreign records",
        )
        require(
            all(str(row.get("email", "")).endswith("@example.test") for row in before_unauthorized),
            "pre-defense response included a non-synthetic record",
        )

        parameterized = await _control(client, run_id, episode_id, "PARAMETERIZE_QUERY")
        require(
            parameterized.get("status") == "applied", f"parameterization failed: {parameterized}"
        )
        after = await _request(client, run_id, episode_id, 1, ATTACK_ACTION, "verifier_attack")
        after_rows = (after.get("response") or {}).get("rows", [])
        require(after.get("status") == "completed", f"post-defense request failed: {after}")
        require(
            (after.get("telemetry") or {}).get("evidence_status") == "verified",
            f"post-defense evidence was not correlated: {after}",
        )
        require(
            not [row for row in after_rows if row.get("owner_ref") != "guest"],
            f"same attack class returned foreign rows after defense: {after_rows}",
        )
        legitimate = await _request(
            client, run_id, episode_id, 2, LEGITIMATE_ACTION, "verifier_legitimate"
        )
        legitimate_rows = (legitimate.get("response") or {}).get("rows", [])
        require(
            legitimate.get("http_status") == 200, f"legitimate query was not served: {legitimate}"
        )
        require(
            any(row.get("record_ref") == "rec-001" for row in legitimate_rows),
            f"legitimate seeded record was unavailable: {legitimate_rows}",
        )
        await _event(
            client,
            run_id,
            episode_id,
            "sqli_verification",
            {
                "schema_id": "sqli.control_verification.v1",
                "acceptance_smoke": True,
                "before_unauthorized_rows": len(before_unauthorized),
                "after_unauthorized_rows": 0,
                "legitimate_query_served": True,
            },
        )
        finish = await client.post(
            f"/range/runs/{run_id}/episodes/{episode_id}/finish",
            json={"status": "completed", "reason": "sqli_acceptance_smoke"},
        )
        finish.raise_for_status()
        timeline = await client.get(
            f"/range/runs/{run_id}/episodes/{episode_id}/timeline?limit=500"
        )
        timeline.raise_for_status()
        event_types = [item["event_type"] for item in timeline.json()["items"]]
        require(
            "sqli_target_response" in event_types and "sqli_verification" in event_types,
            "acceptance results were not replayable",
        )
        ppo_probe = await _verify_ppo_episode(client)
    return {
        "status": "passed",
        "scenario_id": "sqli_lab_v1",
        "target_version": "sqli_target_v1",
        "run_id": run_id,
        "episode_id": episode_id,
        "unauthorized_rows_before_defense": len(before_unauthorized),
        "unauthorized_rows_after_defense": 0,
        "legitimate_query_served": True,
        "timeline_events": len(timeline.json()["items"]),
        "ppo_episode": ppo_probe,
        "comparator_label": "acceptance smoke verifier request; not a learned policy and not training data",
    }


async def main() -> int:
    try:
        report = await run()
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
