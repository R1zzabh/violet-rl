import asyncio
import json
from uuid import uuid4

import httpx

from violet.range.contracts import (
    AuthTestAction,
    CheckDocumentAccessAction,
    ControlCommand,
    ExecutorJob,
)
from violet.range.executor_main import RangeExecutor
from violet.range.verifier import verify_document_access
from violet.target_app.app import create_app


def make_job(action, run_id, episode_id, step_id):
    action_id = uuid4()
    return ExecutorJob(
        job_id=uuid4(),
        run_id=run_id,
        episode_id=episode_id,
        step_id=step_id,
        decision_id=uuid4(),
        action_id=action_id,
        request_id=uuid4(),
        idempotency_key=action_id,
        action=action,
    )


def test_real_http_idor_blue_control_and_target_evidence(tmp_path) -> None:
    evidence_path = tmp_path / "target-access.jsonl"
    target = create_app(
        database_url=f"sqlite:///{tmp_path / 'ledger.sqlite3'}",
        evidence_path=evidence_path,
        control_inbox=tmp_path / "inbox",
        control_results=tmp_path / "control-results",
    )
    executor = RangeExecutor(
        tmp_path / "executor",
        transport_factory=lambda: httpx.ASGITransport(app=target),
    )
    run_id, episode_id = uuid4(), uuid4()
    expected_target_event: dict[str, str] = {}

    async def scenario() -> None:
        executor.initialize()
        reset = ControlCommand(
            command_id=uuid4(), command="RESET", seed=17, run_id=run_id, episode_id=episode_id
        )
        reset_ack = target.state.runtime.apply(reset)
        assert reset_ack.fixture_seed == 17

        auth_job = make_job(
            AuthTestAction(operation="AUTH_TEST", credential_ref="synthetic_alice"),
            run_id,
            episode_id,
            0,
        )
        auth_result = await executor.execute(auth_job)
        assert auth_result.status == "completed" and auth_result.http_status == 200
        assert auth_result.session_handle and auth_result.principal_ref == "alice"
        assert "Set-Cookie" not in auth_result.response_excerpt

        vulnerable_job = make_job(
            CheckDocumentAccessAction(
                operation="CHECK_DOCUMENT_ACCESS",
                document_ref="doc_bob",
                session_handle=auth_result.session_handle,
            ),
            run_id,
            episode_id,
            1,
        )
        vulnerable_result = await executor.execute(vulnerable_job)
        assert vulnerable_result.http_status == 200
        assert vulnerable_result.target_event_id
        expected_target_event["id"] = str(vulnerable_result.target_event_id)
        assert (
            verify_document_access(vulnerable_job, vulnerable_result, fixture_seed=17)["status"]
            == "confirmed"
        )

        harden = ControlCommand(
            command_id=uuid4(),
            command="APPLY_CONTROL",
            action="HARDEN_DOCUMENT_OWNERSHIP",
            run_id=run_id,
            episode_id=episode_id,
        )
        ack = target.state.runtime.apply(harden)
        assert ack.controls["ownership_hardened"] is True
        blocked_result = await executor.execute(
            make_job(
                CheckDocumentAccessAction(
                    operation="CHECK_DOCUMENT_ACCESS",
                    document_ref="doc_bob",
                    session_handle=auth_result.session_handle,
                ),
                run_id,
                episode_id,
                2,
            )
        )
        assert blocked_result.http_status == 403

        bob_auth = await executor.execute(
            make_job(
                AuthTestAction(operation="AUTH_TEST", credential_ref="synthetic_bob"),
                run_id,
                episode_id,
                3,
            )
        )
        bob_access_job = make_job(
            CheckDocumentAccessAction(
                operation="CHECK_DOCUMENT_ACCESS",
                document_ref="doc_bob",
                session_handle=bob_auth.session_handle,
            ),
            run_id,
            episode_id,
            4,
        )
        bob_access = await executor.execute(bob_access_job)
        assert bob_access.http_status == 200
        assert (
            verify_document_access(bob_access_job, bob_access, fixture_seed=17)["status"]
            == "failed"
        )
        await executor.close()

    asyncio.run(scenario())
    events = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert len(events) == 5
    assert events[1]["run_id"] == str(run_id)
    assert events[1]["episode_id"] == str(episode_id)
    assert events[1]["request_id"]
    assert events[1]["target_event_id"] == expected_target_event["id"]
