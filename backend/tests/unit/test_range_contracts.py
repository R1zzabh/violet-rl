import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from violet.range.contracts import (
    CheckDocumentAccessAction,
    DiscoverRouteAction,
    ExecutorJob,
    ExecutorJobRequest,
    ExecutorResult,
)
from violet.range.executor_main import RangeExecutor, _safe_excerpt, validate_target_base_url


def test_action_contract_rejects_raw_urls_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExecutorJobRequest.model_validate(
            {
                "action": {"operation": "DISCOVER_ROUTE", "route_candidate_id": "home"},
                "url": "http://example.com/",
            }
        )


def test_agent_decision_schema_must_match_role_and_blue_catalog_is_bounded() -> None:
    from violet.range.contracts import AgentDecision, BlueAction

    with pytest.raises(ValidationError, match="schema does not match"):
        AgentDecision(
            decision_id=uuid4(),
            run_id=uuid4(),
            episode_id=uuid4(),
            step_id=0,
            role="blue",
            policy_id="baseline",
            observation_schema_id="red.range_observation.v2",
            legal_action_mask=(True,),
            selected_action={"operation": "MONITOR"},
            inference_ms=0,
        )
    with pytest.raises(ValidationError):
        BlueAction.model_validate({"operation": "BLOCK_ACTOR", "principal_ref": "admin"})
    with pytest.raises(ValidationError):
        ExecutorJobRequest.model_validate(
            {"action": {"operation": "DISCOVER_ROUTE", "path": "/anything"}}
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://violet-ledger:8001",
        "http://127.0.0.1:8001",
        "http://violet-ledger:8001/redirect",
        "http://violet-ledger:8080",
        "http://user@violet-ledger:8001",
        "file:///etc/passwd",
    ],
)
def test_executor_target_is_a_fixed_internal_service(url: str) -> None:
    with pytest.raises(ValueError, match="fixed Violet Ledger"):
        validate_target_base_url(url)


def test_executor_job_requires_typed_correlation_ids() -> None:
    with pytest.raises(ValidationError):
        ExecutorJob.model_validate(
            {
                "job_id": str(uuid4()),
                "run_id": str(uuid4()),
                "episode_id": str(uuid4()),
                "step_id": 1,
                "decision_id": str(uuid4()),
                "action_id": str(uuid4()),
                "target_id": "example.com",
                "request_id": str(uuid4()),
                "idempotency_key": str(uuid4()),
                "action": {"operation": "CHECK_TARGET_HEALTH"},
            }
        )


def test_excerpt_is_bounded_and_redacts_secret_shaped_fields() -> None:
    response = httpx.Response(
        200,
        json={
            "principal_ref": "alice",
            "session_token": "secret",
            "nested": {"password": "secret"},
        },
    )
    excerpt = _safe_excerpt(response)
    assert "secret" not in excerpt
    assert "[REDACTED]" in excerpt
    assert "password=[REDACTED]" in _safe_excerpt(
        httpx.Response(200, text="password=hunter2 cookie: abc123")
    )
    assert len(_safe_excerpt(httpx.Response(200, text="x" * 3000))) == 2048


def test_persisted_session_reference_is_an_alias() -> None:
    from violet.range.contracts import ExecutorResult

    result = ExecutorResult(
        job_id=uuid4(),
        action_id=uuid4(),
        request_id=uuid4(),
        status="completed",
        session_alias="session-0123456789ab",
        started_at="2026-10-10T00:00:00+00:00",
        finished_at="2026-10-10T00:00:00.010+00:00",
        duration_ms=10,
    )
    assert "session_alias" in result.model_dump()


def test_legacy_raw_session_handles_are_discarded_on_read() -> None:
    action = CheckDocumentAccessAction.model_validate(
        {
            "operation": "CHECK_DOCUMENT_ACCESS",
            "document_ref": "doc_bob",
            "session_handle": "A" * 43,
        }
    )
    assert action.session_alias is None
    result = ExecutorResult.model_validate(
        {
            "job_id": uuid4(),
            "action_id": uuid4(),
            "request_id": uuid4(),
            "status": "completed",
            "session_handle": "A" * 43,
            "started_at": "2026-10-10T00:00:00+00:00",
            "finished_at": "2026-10-10T00:00:00.010+00:00",
            "duration_ms": 10,
        }
    )
    assert "session_handle" not in result.model_dump()


def test_episode_request_budget_is_a_hard_limit(tmp_path) -> None:
    executor = RangeExecutor(tmp_path)
    executor.initialize()
    assert executor.budget_ledger_valid
    run_id, episode_id = uuid4(), uuid4()
    executor.episode_budgets[f"{run_id}:{episode_id}"] = {"count": 40.0, "started_at": 0.0}
    action_id = uuid4()
    job = ExecutorJob(
        job_id=uuid4(),
        run_id=run_id,
        episode_id=episode_id,
        step_id=40,
        decision_id=uuid4(),
        action_id=action_id,
        request_id=uuid4(),
        idempotency_key=action_id,
        action=DiscoverRouteAction(operation="DISCOVER_ROUTE", route_candidate_id="home"),
    )
    result = asyncio.run(executor.execute(job))
    assert result.status == "rejected"
    assert result.error_code == "request_budget_exhausted"


def test_executor_restart_marks_in_flight_unknown_without_requeue(tmp_path) -> None:
    spool = tmp_path / "spool"
    executor = RangeExecutor(spool)
    executor.initialize()
    action_id = uuid4()
    job = ExecutorJob(
        job_id=uuid4(),
        run_id=uuid4(),
        episode_id=uuid4(),
        step_id=0,
        decision_id=uuid4(),
        action_id=action_id,
        request_id=uuid4(),
        idempotency_key=action_id,
        action=DiscoverRouteAction(operation="DISCOVER_ROUTE", route_candidate_id="home"),
    )
    processing = spool / "processing" / f"{job.job_id}.json"
    processing.write_text(json.dumps(job.model_dump(mode="json")))
    restarted = RangeExecutor(spool)
    restarted.initialize()
    result = ExecutorResult.model_validate(
        json.loads((spool / "results" / processing.name).read_text())
    )
    assert result.status == "outcome_unknown"
    assert result.error_code == "executor_restarted_in_flight"
    assert not (spool / "jobs" / processing.name).exists()
