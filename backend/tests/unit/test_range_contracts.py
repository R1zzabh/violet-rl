import asyncio
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from violet.range.contracts import DiscoverRouteAction, ExecutorJob, ExecutorJobRequest
from violet.range.executor_main import RangeExecutor, _safe_excerpt, validate_target_base_url


def test_action_contract_rejects_raw_urls_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExecutorJobRequest.model_validate(
            {
                "action": {"operation": "DISCOVER_ROUTE", "route_candidate_id": "home"},
                "url": "http://example.com/",
            }
        )
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
    assert len(_safe_excerpt(httpx.Response(200, text="x" * 3000))) == 2048


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
