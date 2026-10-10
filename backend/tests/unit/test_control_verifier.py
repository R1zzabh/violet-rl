from uuid import uuid4

from violet.range.contracts import (
    CheckDocumentAccessAction,
    ControlAcknowledgement,
    ControlCommand,
    ExecutorJob,
    ExecutorResult,
)
from violet.range.control_verifier import verify_ownership_control


def _job(run_id, episode_id, principal: str) -> tuple[ExecutorJob, ExecutorResult]:
    job_id, action_id, request_id = uuid4(), uuid4(), uuid4()
    target_event_id = uuid4()
    job = ExecutorJob(
        job_id=job_id,
        run_id=run_id,
        episode_id=episode_id,
        step_id=1,
        decision_id=uuid4(),
        action_id=action_id,
        request_id=request_id,
        idempotency_key=action_id,
        action=CheckDocumentAccessAction(
            operation="CHECK_DOCUMENT_ACCESS",
            document_ref="doc_bob",
            session_handle="A" * 43,
        ),
    )
    result = ExecutorResult(
        job_id=job_id,
        action_id=action_id,
        request_id=request_id,
        status="completed",
        http_status=403 if principal == "alice" else 200,
        target_event_id=target_event_id,
        principal_ref=principal,
        started_at="2026-10-10T00:00:00+00:00",
        finished_at="2026-10-10T00:00:00.010+00:00",
        duration_ms=10,
    )
    return job, result


def test_control_verification_requires_denial_and_owner_availability() -> None:
    run_id, episode_id, command_id = uuid4(), uuid4(), uuid4()
    command = ControlCommand(
        command_id=command_id,
        command="APPLY_CONTROL",
        action="HARDEN_DOCUMENT_OWNERSHIP",
        run_id=run_id,
        episode_id=episode_id,
    )
    acknowledgement = ControlAcknowledgement(
        command_id=command_id,
        run_id=run_id,
        episode_id=episode_id,
        status="applied",
        state_version=1,
        controls={"ownership_hardened": True},
        fixture_seed=17,
        completed_at="2026-10-10T00:00:00+00:00",
    )
    alice_job, alice_result = _job(run_id, episode_id, "alice")
    bob_job, bob_result = _job(run_id, episode_id, "bob")

    verified = verify_ownership_control(
        command, acknowledgement, alice_job, alice_result, bob_job, bob_result
    )

    assert verified["status"] == "verified"
    assert all(verified["checks"].values())
    assert verified["evidence"]["unauthorized_request_id"] == str(alice_result.request_id)

    mismatched_ack = acknowledgement.model_copy(update={"episode_id": uuid4()})
    rejected = verify_ownership_control(
        command, mismatched_ack, alice_job, alice_result, bob_job, bob_result
    )
    assert rejected["status"] == "failed"
    assert rejected["checks"]["control_acknowledged"] is False


def test_control_verification_rejects_service_disruption() -> None:
    run_id, episode_id, command_id = uuid4(), uuid4(), uuid4()
    command = ControlCommand(
        command_id=command_id,
        command="APPLY_CONTROL",
        action="HARDEN_DOCUMENT_OWNERSHIP",
        run_id=run_id,
        episode_id=episode_id,
    )
    acknowledgement = ControlAcknowledgement(
        command_id=command_id,
        run_id=run_id,
        episode_id=episode_id,
        status="applied",
        state_version=1,
        controls={"ownership_hardened": True},
        fixture_seed=17,
        completed_at="2026-10-10T00:00:00+00:00",
    )
    alice_job, alice_result = _job(run_id, episode_id, "alice")
    bob_job, bob_result = _job(run_id, episode_id, "bob")
    bob_result = bob_result.model_copy(update={"http_status": 503})

    verified = verify_ownership_control(
        command, acknowledgement, alice_job, alice_result, bob_job, bob_result
    )

    assert verified["status"] == "failed"
    assert verified["checks"]["alice_foreign_read_denied"] is True
    assert verified["checks"]["bob_owner_read_served"] is False
