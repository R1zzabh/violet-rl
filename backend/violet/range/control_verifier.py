"""Evidence-based verification for Violet Ledger ownership hardening."""

from __future__ import annotations

from typing import Any

from violet.range.contracts import (
    CheckDocumentAccessAction,
    ControlAcknowledgement,
    ControlCommand,
    ExecutorJob,
    ExecutorResult,
)


def submission_event_type(purpose: str) -> str:
    """Return the persisted event used to establish request ordering."""
    if purpose == "policy":
        return "action_submitted"
    if purpose == "evaluator":
        return "evaluator_probe_submitted"
    if purpose == "manual":
        return "manual_action_submitted"
    raise ValueError(f"unsupported executor job purpose: {purpose}")


def _access_check(
    job: ExecutorJob,
    result: ExecutorResult,
    expected_principal: str,
    expected_document: str,
    expected_status: int,
) -> bool:
    return (
        isinstance(job.action, CheckDocumentAccessAction)
        and job.action.document_ref == expected_document
        and job.purpose == "evaluator"
        and job.action.session_alias is not None
        and result.job_id == job.job_id
        and result.action_id == job.action_id
        and result.status == "completed"
        and result.http_status == expected_status
        and result.principal_ref == expected_principal
        and result.target_event_id is not None
        and result.request_id == job.request_id
    )


def verify_ownership_control(
    command: ControlCommand,
    acknowledgement: ControlAcknowledgement,
    unauthorized_job: ExecutorJob,
    unauthorized_result: ExecutorResult,
    alice_owner_job: ExecutorJob,
    alice_owner_result: ExecutorResult,
    owner_job: ExecutorJob,
    owner_result: ExecutorResult,
) -> dict[str, Any]:
    """Confirm foreign denial while Alice and Bob retain their own document access."""
    same_episode = (
        command.run_id == unauthorized_job.run_id == alice_owner_job.run_id == owner_job.run_id
        and command.episode_id
        == unauthorized_job.episode_id
        == alice_owner_job.episode_id
        == owner_job.episode_id
    )
    ownership_control_applied = (
        command.command == "APPLY_CONTROL"
        and command.action == "HARDEN_DOCUMENT_OWNERSHIP"
        and acknowledgement.command_id == command.command_id
        and acknowledgement.run_id == command.run_id
        and acknowledgement.episode_id == command.episode_id
        and acknowledgement.status == "applied"
        and acknowledgement.controls.get("ownership_hardened") is True
    )
    alice_denied = _access_check(unauthorized_job, unauthorized_result, "alice", "doc_bob", 403)
    alice_owner_served = _access_check(
        alice_owner_job, alice_owner_result, "alice", "doc_alice", 200
    )
    bob_served = _access_check(owner_job, owner_result, "bob", "doc_bob", 200)
    return {
        "status": "verified"
        if same_episode
        and ownership_control_applied
        and alice_denied
        and alice_owner_served
        and bob_served
        else "failed",
        "command_id": str(command.command_id),
        "control_type": command.action,
        "checks": {
            "same_run_episode": same_episode,
            "control_acknowledged": ownership_control_applied,
            "alice_foreign_read_denied": alice_denied,
            "alice_owner_read_served": alice_owner_served,
            "bob_owner_read_served": bob_served,
        },
        "evidence": {
            "unauthorized_job_id": str(unauthorized_job.job_id),
            "unauthorized_request_id": str(unauthorized_result.request_id),
            "unauthorized_target_event_id": str(unauthorized_result.target_event_id)
            if unauthorized_result.target_event_id
            else None,
            "owner_job_id": str(owner_job.job_id),
            "alice_owner_job_id": str(alice_owner_job.job_id),
            "alice_owner_request_id": str(alice_owner_result.request_id),
            "alice_owner_target_event_id": str(alice_owner_result.target_event_id)
            if alice_owner_result.target_event_id
            else None,
            "owner_request_id": str(owner_result.request_id),
            "owner_target_event_id": str(owner_result.target_event_id)
            if owner_result.target_event_id
            else None,
        },
    }
