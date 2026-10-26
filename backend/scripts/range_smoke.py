"""Exercise the local Violet Ledger attack, control, and verification slice."""

from __future__ import annotations

import json
import os
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

API_URL = os.getenv("VIOLET_API_URL", "http://127.0.0.1:8000")
POLL_TIMEOUT_SECONDS = 60


def validate_api_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("smoke script API URL must be a loopback HTTP origin")


def wait_for(client: httpx.Client, path: str, field: str, expected: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = client.get(path)
        if response.status_code == 200:
            value = response.json()
            if value.get(field) == expected:
                return value
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for {path} to reach {expected}")


def submit_job(
    client: httpx.Client,
    action: dict,
    run_id: str,
    episode_id: str,
    step_id: int,
) -> dict:
    response = client.post(
        "/range/jobs",
        json={
            "action": action,
            "purpose": "evaluator",
            "run_id": run_id,
            "episode_id": episode_id,
            "step_id": step_id,
        },
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        result = client.get(f"/range/jobs/{job_id}")
        result.raise_for_status()
        value = result.json()
        if "result" in value:
            return value
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for executor job {job_id}")


def wait_for_control(client: httpx.Client, command_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = client.get(f"/range/controls/{command_id}")
        response.raise_for_status()
        value = response.json()
        if value.get("status") != "queued_or_running":
            return value
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for target control {command_id}")


def main() -> None:
    validate_api_url(API_URL)
    run_id, episode_id = str(uuid4()), str(uuid4())
    with httpx.Client(base_url=API_URL, timeout=5.0) as client:
        health = client.get("/range/health")
        health.raise_for_status()
        print("Range health:", json.dumps(health.json(), sort_keys=True))

        reset = client.post(
            "/range/reset",
            json={"seed": 17, "run_id": run_id, "episode_id": episode_id},
        )
        reset.raise_for_status()
        reset_ack = wait_for_control(client, reset.json()["command_id"])
        assert reset_ack["status"] == "applied"
        assert reset_ack["fixture_seed"] == 17

        login_alice = submit_job(
            client,
            {"operation": "AUTH_TEST", "credential_ref": "synthetic_alice"},
            run_id,
            episode_id,
            0,
        )
        alice_result = login_alice["result"]
        assert alice_result["http_status"] == 200 and alice_result["session_alias"]
        alice_session = alice_result["session_alias"]

        vulnerable = submit_job(
            client,
            {
                "operation": "CHECK_DOCUMENT_ACCESS",
                "document_ref": "doc_bob",
                "session_alias": alice_session,
            },
            run_id,
            episode_id,
            1,
        )
        assert vulnerable["result"]["http_status"] == 200
        assert vulnerable["finding"]["status"] == "confirmed"
        assert vulnerable["result"]["target_event_id"]
        print("Verified IDOR:", json.dumps(vulnerable["finding"], sort_keys=True))

        control = client.post(
            "/range/controls",
            json={
                "action": "HARDEN_DOCUMENT_OWNERSHIP",
                "purpose": "evaluator",
                "run_id": run_id,
                "episode_id": episode_id,
            },
        )
        control.raise_for_status()
        control_ack = wait_for_control(client, control.json()["command_id"])
        assert control_ack["status"] == "applied"
        assert control_ack["controls"]["ownership_hardened"] is True

        denied = submit_job(
            client,
            {
                "operation": "CHECK_DOCUMENT_ACCESS",
                "document_ref": "doc_bob",
                "session_alias": alice_session,
            },
            run_id,
            episode_id,
            2,
        )
        assert denied["result"]["http_status"] == 403
        assert denied["finding"]["status"] != "confirmed"

        alice_owner = submit_job(
            client,
            {
                "operation": "CHECK_DOCUMENT_ACCESS",
                "document_ref": "doc_alice",
                "session_alias": alice_session,
            },
            run_id,
            episode_id,
            3,
        )
        assert alice_owner["result"]["http_status"] == 200
        assert alice_owner["finding"]["status"] != "confirmed"

        login_bob = submit_job(
            client,
            {"operation": "AUTH_TEST", "credential_ref": "synthetic_bob"},
            run_id,
            episode_id,
            4,
        )
        assert login_bob["result"]["http_status"] == 200
        bob_session = login_bob["result"]["session_alias"]
        owner_access = submit_job(
            client,
            {
                "operation": "CHECK_DOCUMENT_ACCESS",
                "document_ref": "doc_bob",
                "session_alias": bob_session,
            },
            run_id,
            episode_id,
            5,
        )
        assert owner_access["result"]["http_status"] == 200
        assert owner_access["finding"]["status"] != "confirmed"
        assert owner_access["result"]["target_event_id"]

        verification_payload = {
            "unauthorized_job_id": denied["job"]["job_id"],
            "alice_owner_job_id": alice_owner["job"]["job_id"],
            "owner_job_id": owner_access["job"]["job_id"],
        }
        verification = client.post(
            f"/range/controls/{control.json()['command_id']}/verify",
            json=verification_payload,
        )
        if verification.is_error:
            raise RuntimeError(
                f"control verification failed: {verification.status_code} {verification.text}"
            )
        verification.raise_for_status()
        assert verification.json()["status"] == "verified"
        assert all(verification.json()["checks"].values())
        repeated_verification = client.post(
            f"/range/controls/{control.json()['command_id']}/verify",
            json=verification_payload,
        )
        repeated_verification.raise_for_status()
        assert repeated_verification.json() == verification.json()

        timeline = client.get(f"/range/runs/{run_id}/events?limit=1000")
        timeline.raise_for_status()
        events = timeline.json()["events"]
        assert events and [event["seq"] for event in events] == list(range(1, len(events) + 1))
        assert any(
            event["event_type"] == "target_evidence_ingested"
            and event["payload"]["target_event_id"] == vulnerable["result"]["target_event_id"]
            for event in events
        )
        assert any(
            event["event_type"] == "executor_result"
            and (event["payload"].get("finding") or {}).get("status") == "confirmed"
            for event in events
        )
        assert sum(event["event_type"] == "control_verification" for event in events) == 1
        assert all(
            events[index]["previous_hash"] == events[index - 1]["event_hash"]
            for index in range(1, len(events))
        )
        resumed = client.get(f"/range/runs/{run_id}/events?after_seq=3&limit=1000")
        resumed.raise_for_status()
        assert [event["seq"] for event in resumed.json()["events"]] == [
            event["seq"] for event in events if event["seq"] > 3
        ]
        print(
            "Control verified: Alice received 403; Bob received 200; "
            f"persisted {len(events)} sequenced events. "
            f"run_id={run_id} episode_id={episode_id}"
        )


if __name__ == "__main__":
    main()
