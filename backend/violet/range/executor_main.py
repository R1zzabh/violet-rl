"""Isolated, typed HTTP executor for Violet Ledger."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from pydantic import ValidationError

from violet.range.contracts import (
    AuthTestAction,
    CheckDocumentAccessAction,
    CheckTargetHealthAction,
    DiscoverRouteAction,
    ExecutorJob,
    ExecutorResult,
)
from violet.range.spool import read_json, write_json_atomic

if TYPE_CHECKING:
    from collections.abc import Callable

TARGET_ID = "violet_ledger_v1"
ROUTES: dict[str, tuple[str, str]] = {
    "home": ("GET", "/"),
    "login": ("GET", "/api/login"),
    "document": ("GET", "/api/documents/doc_bob"),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_target_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "violet-ledger"
        or parsed.port != 8001
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("executor target must be the fixed Violet Ledger service on port 8001")
    return "http://violet-ledger:8001"


def _safe_excerpt(response: httpx.Response) -> str:
    """Bound and redact any credential-shaped fields before persistence."""
    try:
        payload = response.json()
    except (ValueError, UnicodeDecodeError):
        return response.text[:2048]

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            safe: dict[str, Any] = {}
            for key, item in value.items():
                normalized = str(key).lower().replace("-", "_")
                if any(
                    secret in normalized
                    for secret in ("password", "cookie", "token", "authorization", "credential")
                ):
                    safe[str(key)] = "[REDACTED]"
                else:
                    safe[str(key)] = scrub(item)
            return safe
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return json.dumps(scrub(payload), separators=(",", ":"), ensure_ascii=True)[:2048]


@dataclass
class ExecutorSession:
    client: httpx.AsyncClient
    principal_ref: str


class RangeExecutor:
    """Maps versioned actions to fixed methods and route IDs."""

    def __init__(
        self,
        spool: Path,
        target_base_url: str | None = None,
        transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
    ):
        self.spool = spool
        self.jobs_dir = spool / "jobs"
        self.processing_dir = spool / "processing"
        self.results_dir = spool / "results"
        self.target_base_url = validate_target_base_url(
            target_base_url or os.getenv("VIOLET_TARGET_BASE_URL", "http://violet-ledger:8001")
        )
        self.sessions: dict[str, ExecutorSession] = {}
        self.transport_factory = transport_factory
        self.episode_budgets: dict[str, dict[str, float]] = {}
        self.budget_ledger_valid = True
        self.last_request_at = 0.0
        self.client_options: dict[str, Any] = {
            "base_url": self.target_base_url,
            "timeout": httpx.Timeout(5.0, connect=2.0),
            "follow_redirects": False,
            "limits": httpx.Limits(max_connections=1, max_keepalive_connections=1),
        }
        self.last_heartbeat_at = 0.0

    def _new_client(self) -> httpx.AsyncClient:
        options = dict(self.client_options)
        if self.transport_factory is not None:
            options["transport"] = self.transport_factory()
        return httpx.AsyncClient(**options)

    async def close(self) -> None:
        for session in self.sessions.values():
            await session.client.aclose()
        self.sessions.clear()

    def initialize(self) -> None:
        for directory in (self.jobs_dir, self.processing_dir, self.results_dir):
            directory.mkdir(parents=True, exist_ok=True)
        budgets_path = self.spool / "episode-budgets.json"
        if budgets_path.exists():
            try:
                raw_budgets = read_json(budgets_path)
                self.episode_budgets = {
                    key: {"count": float(value["count"]), "started_at": float(value["started_at"])}
                    for key, value in raw_budgets.items()
                    if isinstance(value, dict)
                }
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                # A corrupted budget ledger fails closed in execute() below.
                self.budget_ledger_valid = False
        for in_progress in self.processing_dir.glob("*.json"):
            result = self.results_dir / in_progress.name
            if result.exists():
                in_progress.unlink(missing_ok=True)
            else:
                destination = self.jobs_dir / in_progress.name
                if not destination.exists():
                    os.replace(in_progress, destination)

    async def execute(self, job: ExecutorJob) -> ExecutorResult:
        started_at = utc_now()
        started = time.monotonic()
        action = job.action
        client: httpx.AsyncClient | None = None
        should_close_client = True
        principal_ref: str | None = None
        session_handle: str | None = None
        method: str
        path: str
        json_body: dict[str, str] | None = None

        if not isinstance(action, CheckTargetHealthAction):
            if not self.budget_ledger_valid:
                return self._result(
                    job, "rejected", started_at, started, error_code="request_budget_exhausted"
                )
            budget_key = f"{job.run_id}:{job.episode_id}"
            current = self.episode_budgets.get(budget_key)
            if current is None:
                current = {"count": 0.0, "started_at": time.time()}
            if current["count"] >= 40:
                return self._result(
                    job, "rejected", started_at, started, error_code="request_budget_exhausted"
                )
            if time.time() - current["started_at"] > 300:
                return self._result(
                    job, "rejected", started_at, started, error_code="episode_deadline_exceeded"
                )
            current["count"] += 1
            self.episode_budgets[budget_key] = current
            write_json_atomic(self.spool / "episode-budgets.json", self.episode_budgets)

        if isinstance(action, DiscoverRouteAction):
            method, path = ROUTES[action.route_candidate_id]
            client = self._new_client()
        elif isinstance(action, AuthTestAction):
            method, path = "POST", "/api/login"
            client = self._new_client()
            json_body = {"credential_ref": action.credential_ref}
        elif isinstance(action, CheckDocumentAccessAction):
            method, path = "GET", f"/api/documents/{action.document_ref}"
            if action.session_handle is None or action.session_handle not in self.sessions:
                return self._result(
                    job,
                    "rejected",
                    started_at,
                    started,
                    error_code="unknown_session",
                )
            session = self.sessions[action.session_handle]
            client = session.client
            principal_ref = session.principal_ref
            should_close_client = False
        elif isinstance(action, CheckTargetHealthAction):
            method, path = "GET", "/health"
            client = self._new_client()
        else:  # pragma: no cover - the union fails closed before dispatch
            return self._result(job, "rejected", started_at, started, error_code="invalid_job")

        headers = {
            "X-Violet-Run-ID": str(job.run_id),
            "X-Violet-Episode-ID": str(job.episode_id),
            "X-Violet-Step-ID": str(job.step_id),
            "X-Violet-Decision-ID": str(job.decision_id),
            "X-Violet-Action-ID": str(job.action_id),
            "X-Violet-Request-ID": str(job.request_id),
        }
        try:
            elapsed_since_request = time.monotonic() - self.last_request_at
            if self.last_request_at and elapsed_since_request < 0.1:
                await asyncio.sleep(0.1 - elapsed_since_request)
            self.last_request_at = time.monotonic()
            async with asyncio.timeout(job.deadline_seconds):
                response = await client.request(method, path, headers=headers, json=json_body)
            body_text = _safe_excerpt(response)
            event_id_text = response.headers.get("X-Violet-Target-Event-ID")
            target_event_id = UUID(event_id_text) if event_id_text else None
            if isinstance(action, AuthTestAction) and response.status_code == 200:
                payload = response.json()
                principal_ref = (
                    payload.get("principal_ref")
                    if payload.get("principal_ref") in {"alice", "bob"}
                    else None
                )
                if principal_ref:
                    session_handle = secrets.token_urlsafe(32)
                    self.sessions[session_handle] = ExecutorSession(client, principal_ref)
                    client = None
            result = self._result(
                job,
                "completed",
                started_at,
                started,
                http_status=response.status_code,
                content_type=response.headers.get("content-type"),
                response_excerpt=body_text,
                target_event_id=target_event_id,
                session_handle=session_handle,
                principal_ref=principal_ref,
            )
            return result
        except (httpx.TimeoutException, TimeoutError):
            return self._result(job, "outcome_unknown", started_at, started, error_code="timeout")
        except httpx.RequestError:
            return self._result(job, "failed", started_at, started, error_code="connection_failed")
        except Exception:
            return self._result(job, "failed", started_at, started, error_code="executor_error")
        finally:
            if client is not None and should_close_client:
                await client.aclose()

    @staticmethod
    def _result(
        job: ExecutorJob,
        status: str,
        started_at: str,
        started: float,
        *,
        http_status: int | None = None,
        content_type: str | None = None,
        response_excerpt: str = "",
        target_event_id: UUID | None = None,
        session_handle: str | None = None,
        principal_ref: str | None = None,
        error_code: str | None = None,
    ) -> ExecutorResult:
        elapsed = (time.monotonic() - started) * 1000
        return ExecutorResult(
            job_id=job.job_id,
            action_id=job.action_id,
            request_id=job.request_id,
            status=status,  # type: ignore[arg-type]
            http_status=http_status,
            content_type=content_type,
            response_excerpt=response_excerpt,
            target_event_id=target_event_id,
            session_handle=session_handle,
            principal_ref=principal_ref,  # type: ignore[arg-type]
            error_code=error_code,  # type: ignore[arg-type]
            started_at=started_at,
            finished_at=utc_now(),
            duration_ms=max(0.0, elapsed),
        )

    async def run(self) -> None:
        self.initialize()
        heartbeat = self.spool / "health.json"
        last_health = 0.0
        while True:
            now = time.monotonic()
            if now - self.last_heartbeat_at >= 2.0:
                self._write_heartbeat(
                    heartbeat,
                    "running",
                    read_json(heartbeat).get("target") if heartbeat.exists() else None,
                )
                self.last_heartbeat_at = now
            if now - last_health >= 5.0:
                health_result = await self._check_health()
                self._write_heartbeat(
                    heartbeat,
                    "running",
                    {
                        "status": health_result.status,
                        "http_status": health_result.http_status,
                        "checked_at": health_result.finished_at,
                        "target_id": TARGET_ID,
                    },
                )
                self.last_heartbeat_at = time.monotonic()
                last_health = now
            job_path = next(iter(sorted(self.jobs_dir.glob("*.json"))), None)
            if job_path is None:
                await asyncio.sleep(0.1)
                continue
            processing = self.processing_dir / job_path.name
            try:
                os.replace(job_path, processing)
            except FileNotFoundError:
                continue
            try:
                try:
                    job = ExecutorJob.model_validate(read_json(processing))
                except (ValidationError, ValueError, json.JSONDecodeError):
                    processing.unlink(missing_ok=True)
                    continue
                result_path = self.results_dir / f"{job.job_id}.json"
                if not result_path.exists():
                    result = await self.execute(job)
                    write_json_atomic(result_path, result.model_dump(mode="json"))
            finally:
                processing.unlink(missing_ok=True)

    async def _check_health(self) -> ExecutorResult:
        job = ExecutorJob(
            job_id=uuid4(),
            run_id=uuid4(),
            episode_id=uuid4(),
            step_id=0,
            decision_id=uuid4(),
            action_id=uuid4(),
            request_id=uuid4(),
            idempotency_key=uuid4(),
            action=CheckTargetHealthAction(operation="CHECK_TARGET_HEALTH"),
        )
        return await self.execute(job)

    @staticmethod
    def _write_heartbeat(path: Path, executor_status: str, target: dict[str, Any] | None) -> None:
        write_json_atomic(
            path,
            {
                "executor_status": executor_status,
                "target": target,
                "updated_at": utc_now(),
            },
        )


async def main() -> None:
    executor = RangeExecutor(Path(os.getenv("VIOLET_EXECUTOR_SPOOL", "/spool")))
    try:
        await executor.run()
    finally:
        await executor.close()


if __name__ == "__main__":
    asyncio.run(main())
