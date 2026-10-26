import hashlib
import json
from uuid import uuid4

from violet.range.contracts import CheckDocumentAccessAction, ExecutorJob, ExecutorResult
from violet.range.verifier import verify_document_access


def test_idor_requires_persisted_correlated_target_evidence() -> None:
    run_id, episode_id = uuid4(), uuid4()
    action_id, request_id = uuid4(), uuid4()
    target_event_id = uuid4()
    canary = "VIOLET-CANARY-" + hashlib.sha256(b"violet-ledger:0:doc_bob").hexdigest()[:24].upper()
    job = ExecutorJob(
        job_id=uuid4(),
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
            session_alias="session-0123456789ab",
        ),
    )
    result = ExecutorResult(
        job_id=job.job_id,
        action_id=action_id,
        request_id=request_id,
        status="completed",
        http_status=200,
        response_excerpt=json.dumps(
            {"document_ref": "doc_bob", "owner_ref": "bob", "canary": canary}
        ),
        target_event_id=target_event_id,
        principal_ref="alice",
        started_at="2026-10-13T00:00:00+00:00",
        finished_at="2026-10-13T00:00:00.010+00:00",
        duration_ms=10,
    )
    assert verify_document_access(job, result, fixture_seed=0)["status"] == "inconclusive"
    evidence = {
        "target_event_id": str(target_event_id),
        "request_id": str(request_id),
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "action_id": str(action_id),
    }
    confirmed = verify_document_access(job, result, fixture_seed=0, target_evidence=evidence)
    assert confirmed["status"] == "confirmed"
    evidence["action_id"] = str(uuid4())
    assert (
        verify_document_access(job, result, fixture_seed=0, target_evidence=evidence)["status"]
        == "inconclusive"
    )
