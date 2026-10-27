"""Executor pinned to the single SQLi lab HTTP target."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from pydantic import ValidationError

from violet.range.spool import read_json, write_json_atomic
from violet.sqli_lab.contracts import SqliExecutorResult, SqliJob


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_target_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "sqli-target"
        or parsed.port != 8002
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("SQLi executor target must be the fixed sqli-target service on port 8002")
    return "http://sqli-target:8002"


class SqliExecutor:
    def __init__(self, spool: Path | None = None, target_url: str | None = None) -> None:
        state_home = Path(os.getenv("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        self.spool = spool or Path(
            os.getenv("SQLI_EXECUTOR_SPOOL", str(state_home / "violet/sqli-executor"))
        )
        self.target_url = validate_target_url(
            target_url or os.getenv("SQLI_TARGET_BASE_URL") or "http://sqli-target:8002"
        )
        self.jobs = self.spool / "jobs"
        self.processing = self.spool / "processing"
        self.results = self.spool / "results"
        self.counts: dict[str, int] = {}
        self.last_request_at = 0.0

    def initialize(self) -> None:
        for path in (self.jobs, self.processing, self.results):
            path.mkdir(parents=True, exist_ok=True)
        for path in self.processing.glob("*.json"):
            try:
                job = SqliJob.model_validate(read_json(path))
            except Exception:
                continue
            output = self.results / path.name
            if not output.exists():
                result = self._result(
                    job,
                    status="outcome_unknown",
                    actor_ref="guest" if job.purpose != "verifier_legitimate" else "researcher",
                    started_at=_utc_now(),
                    started=time.monotonic(),
                    challenge_id=job.challenge_id,
                    error_code="executor_error",
                )
                write_json_atomic(output, result.model_dump(mode="json"))
            path.unlink(missing_ok=True)

    async def execute(self, job: SqliJob) -> SqliExecutorResult:
        started_at = _utc_now()
        started = time.monotonic()
        key = f"{job.run_id}:{job.episode_id}"
        if self.counts.get(key, 0) >= 120:
            return self._result(
                job,
                status="rejected",
                actor_ref="guest" if job.purpose != "verifier_legitimate" else "researcher",
                started_at=started_at,
                started=started,
                challenge_id=job.challenge_id,
                error_code="request_budget",
            )
        self.counts[key] = self.counts.get(key, 0) + 1
        actor_ref = "researcher" if job.purpose == "verifier_legitimate" else "guest"
        request_body = {"field": job.action.field, "query_value": job.action.compose_value()}
        headers = {
            "X-Range-Run-ID": str(job.run_id),
            "X-Range-Episode-ID": str(job.episode_id),
            "X-Range-Decision-ID": str(job.decision_id),
            "X-Range-Request-ID": str(job.request_id),
            "X-Range-Actor": actor_ref,
            "X-Range-Purpose": job.purpose,
        }
        if job.challenge_id is not None:
            headers["X-Range-Challenge-ID"] = str(job.challenge_id)
        async with httpx.AsyncClient(
            base_url=self.target_url,
            timeout=httpx.Timeout(4.0, connect=2.0),
            follow_redirects=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        ) as client:
            elapsed = time.monotonic() - self.last_request_at
            if self.last_request_at and elapsed < 0.05:
                await asyncio.sleep(0.05 - elapsed)
            self.last_request_at = time.monotonic()
            try:
                challenge_phrase: str | None = None
                if job.challenge_id is not None:
                    secret_response = await client.get(
                        f"/internal/challenges/{job.challenge_id}/phrase"
                    )
                    if secret_response.status_code != 200:
                        return self._result(
                            job,
                            status="outcome_unknown",
                            actor_ref=actor_ref,
                            started_at=started_at,
                            started=started,
                            challenge_id=job.challenge_id,
                            error_code="challenge_unavailable",
                        )
                    secret_body = secret_response.json()
                    secret_value = secret_body.get("phrase") if isinstance(secret_body, dict) else None
                    if not isinstance(secret_value, str) or not secret_value:
                        return self._result(
                            job,
                            status="outcome_unknown",
                            actor_ref=actor_ref,
                            started_at=started_at,
                            started=started,
                            challenge_id=job.challenge_id,
                            error_code="challenge_unavailable",
                        )
                    challenge_phrase = secret_value
                async with asyncio.timeout(job.deadline_seconds):
                    response = await client.post("/api/lookup", json=request_body, headers=headers)
                try:
                    payload = response.json()
                except (ValueError, UnicodeDecodeError):
                    payload = {"detail": "non-json target response"}
                if not isinstance(payload, dict):
                    payload = {"detail": "unexpected target response"}
                if challenge_phrase is not None:
                    disclosed = payload.pop("challenge_phrase", None) == challenge_phrase
                    payload = self._redact_phrase(payload, challenge_phrase)
                    payload["challenge_disclosed"] = disclosed
                target_event = response.headers.get("X-Sqli-Target-Event-ID")
                return self._result(
                    job,
                    status="completed",
                    actor_ref=actor_ref,
                    started_at=started_at,
                    started=started,
                    challenge_id=job.challenge_id,
                    http_status=response.status_code,
                    target_event_id=UUID(target_event) if target_event else None,
                    request={**request_body, "actor_ref": actor_ref},
                    response=payload,
                )
            except (httpx.TimeoutException, TimeoutError):
                return self._result(
                    job,
                    status="outcome_unknown",
                    actor_ref=actor_ref,
                    started_at=started_at,
                    started=started,
                    challenge_id=job.challenge_id,
                    error_code="timeout",
                    request={**request_body, "actor_ref": actor_ref},
                )
            except httpx.RequestError:
                return self._result(
                    job,
                    status="failed",
                    actor_ref=actor_ref,
                    started_at=started_at,
                    started=started,
                    challenge_id=job.challenge_id,
                    error_code="connection_failed",
                    request={**request_body, "actor_ref": actor_ref},
                )
            except Exception:
                return self._result(
                    job,
                    status="failed",
                    actor_ref=actor_ref,
                    started_at=started_at,
                    started=started,
                    challenge_id=job.challenge_id,
                    error_code="executor_error",
                    request={**request_body, "actor_ref": actor_ref},
                )

    @classmethod
    def _redact_phrase(cls, value: object, phrase: str) -> Any:
        if isinstance(value, str):
            return value.replace(phrase, "[redacted]")
        if isinstance(value, list):
            return [cls._redact_phrase(item, phrase) for item in value]
        if isinstance(value, dict):
            return {str(key): cls._redact_phrase(item, phrase) for key, item in value.items()}
        return value

    @staticmethod
    def _result(
        job: SqliJob,
        *,
        status: str,
        actor_ref: str,
        started_at: str,
        started: float,
        challenge_id: UUID | None = None,
        http_status: int | None = None,
        target_event_id: UUID | None = None,
        request: dict[str, object] | None = None,
        response: dict[str, object] | None = None,
        error_code: str | None = None,
    ) -> SqliExecutorResult:
        return SqliExecutorResult(
            job_id=job.job_id,
            run_id=job.run_id,
            episode_id=job.episode_id,
            decision_id=job.decision_id,
            step_id=job.step_id,
            request_id=job.request_id,
            purpose=job.purpose,
            challenge_id=challenge_id,
            status=status,  # type: ignore[arg-type]
            actor_ref=actor_ref,  # type: ignore[arg-type]
            http_status=http_status,
            target_event_id=target_event_id,
            request=request or {},
            response=response or {},
            error_code=error_code,  # type: ignore[arg-type]
            duration_ms=max(0.0, (time.monotonic() - started) * 1000),
            started_at=started_at,
            finished_at=_utc_now(),
        )

    async def _health(self) -> dict[str, object]:
        try:
            async with httpx.AsyncClient(base_url=self.target_url, timeout=2.0) as client:
                response = await client.get("/health")
                return {
                    "status": "available" if response.status_code == 200 else "unavailable",
                    "http_status": response.status_code,
                }
        except httpx.HTTPError:
            return {"status": "unavailable", "http_status": None}

    async def run(self) -> None:
        self.initialize()
        heartbeat = self.spool / "health.json"
        last_health = 0.0
        while True:
            now = time.monotonic()
            if now - last_health >= 3.0:
                target = await self._health()
                write_json_atomic(
                    heartbeat,
                    {
                        "status": "running",
                        "target": target,
                        "updated_at": _utc_now(),
                        "target_id": "sqli_lab_v1",
                    },
                )
                last_health = time.monotonic()
            path = next(iter(sorted(self.jobs.glob("*.json"))), None)
            if path is None:
                await asyncio.sleep(0.1)
                continue
            processing = self.processing / path.name
            try:
                os.replace(path, processing)
            except FileNotFoundError:
                continue
            try:
                try:
                    job = SqliJob.model_validate(read_json(processing))
                except (ValidationError, ValueError):
                    processing.unlink(missing_ok=True)
                    continue
                result_path = self.results / path.name
                if not result_path.exists():
                    result = await self.execute(job)
                    write_json_atomic(result_path, result.model_dump(mode="json"))
            finally:
                processing.unlink(missing_ok=True)


async def _main() -> None:
    await SqliExecutor().run()


if __name__ == "__main__":
    asyncio.run(_main())
