"""Independent verifier for the seeded Violet Ledger IDOR finding."""

from __future__ import annotations

import hashlib
from typing import Any

from violet.range.contracts import CheckDocumentAccessAction, ExecutorJob, ExecutorResult

# Kept in the verifier rather than exposed through target observations or actions.
FIXTURE_OWNERS = {"doc_alice": "alice", "doc_bob": "bob"}


def verify_document_access(
    job: ExecutorJob,
    result: ExecutorResult,
    fixture_seed: int = 0,
    target_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action = job.action
    if not isinstance(action, CheckDocumentAccessAction):
        return {"status": "inconclusive", "reason": "not_a_document_access_action", "evidence": []}
    if result.status == "outcome_unknown":
        return {
            "status": "inconclusive",
            "reason": "executor_outcome_unknown",
            "evidence": [str(result.request_id)],
        }
    if result.status != "completed" or result.http_status != 200:
        return {
            "status": "failed",
            "reason": "request_did_not_return_document",
            "evidence": [str(result.request_id)],
        }
    correlation_matches = bool(
        target_evidence
        and target_evidence.get("target_event_id") == str(result.target_event_id)
        and target_evidence.get("request_id") == str(job.request_id)
        and target_evidence.get("run_id") == str(job.run_id)
        and target_evidence.get("episode_id") == str(job.episode_id)
        and target_evidence.get("action_id") == str(job.action_id)
    )
    if not correlation_matches:
        return {
            "status": "inconclusive",
            "reason": "persisted_target_evidence_missing_or_mismatched",
            "evidence": [str(job.request_id)],
        }
    try:
        import json

        body = json.loads(result.response_excerpt)
    except (ValueError, TypeError):
        return {
            "status": "inconclusive",
            "reason": "response_not_parseable",
            "evidence": [str(result.request_id)],
        }
    expected_owner = FIXTURE_OWNERS.get(action.document_ref)
    expected_canary = (
        "VIOLET-CANARY-"
        + hashlib.sha256(f"violet-ledger:{fixture_seed}:{action.document_ref}".encode())
        .hexdigest()[:24]
        .upper()
    )
    if (
        expected_owner is not None
        and body.get("document_ref") == action.document_ref
        and body.get("owner_ref") == expected_owner
        and body.get("canary") == expected_canary
        and result.principal_ref in {"alice", "bob"}
    ):
        return {
            "status": "confirmed" if expected_owner != result.principal_ref else "failed",
            "reason": "foreign_fixture_canary_returned"
            if expected_owner != result.principal_ref
            else "owner_access_only",
            "document_ref": action.document_ref,
            "fixture_owner_ref": expected_owner,
            "evidence": [
                str(result.request_id),
                str(result.target_event_id) if result.target_event_id else None,
            ],
        }
    return {
        "status": "inconclusive",
        "reason": "fixture_evidence_mismatch",
        "evidence": [str(result.request_id)],
    }
