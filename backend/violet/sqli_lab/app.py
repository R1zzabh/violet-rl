"""The isolated, intentionally vulnerable HTTP endpoint for the SQLi lab."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from violet.sqli_lab.contracts import TARGET_VERSION, FieldName
from violet.sqli_lab.control import SqliRuntime, watch_controls
from violet.sqli_lab.database import SqliDatabase

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class LookupBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: FieldName
    query_value: str = Field(min_length=1, max_length=96)

    @field_validator("query_value")
    @classmethod
    def safe_character_set(cls, value: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9_ '@=\-]{1,96}", value):
            raise ValueError("lookup value contains a character outside the lab grammar")
        return value


class ChallengeSecretBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phrase: str = Field(min_length=1, max_length=96)


def create_app(
    database_url: str | None = None,
    evidence_path: Path | None = None,
    control_inbox: Path | None = None,
    control_results: Path | None = None,
) -> FastAPI:
    database = SqliDatabase(database_url)
    runtime = SqliRuntime()
    state_home = Path(os.getenv("XDG_STATE_HOME", str(Path.home() / ".local/state")))
    lab_state = state_home / "violet" / "sqli-lab"
    evidence_path = evidence_path or Path(
        os.getenv("SQLI_TARGET_EVIDENCE", str(lab_state / "evidence/events.jsonl"))
    )
    control_inbox = control_inbox or Path(
        os.getenv("SQLI_CONTROL_INBOX", str(lab_state / "control/inbox"))
    )
    control_results = control_results or Path(
        os.getenv("SQLI_CONTROL_RESULTS", str(lab_state / "control/results"))
    )
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(watch_controls(runtime, control_inbox, control_results))
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            database.close()

    app = FastAPI(title="Violet SQLi Lab", version=TARGET_VERSION, lifespan=lifespan)
    app.state.runtime = runtime
    app.state.database = database

    def append_evidence(item: dict[str, Any]) -> str:
        event_id = str(uuid4())
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"target_event_id": event_id, "occurred_at": datetime.now(UTC).isoformat(), **item}
        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        fd = os.open(evidence_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o640)
        try:
            os.write(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)
        return event_id

    def trace_stage(
        request: Request,
        stage_id: str,
        *,
        status: str,
        started: float | None = None,
        inputs: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Record an execution fact at the point where the target observes it."""
        stages = getattr(request.state, "sqli_trace", None)
        if not isinstance(stages, list):
            stages = []
            request.state.sqli_trace = stages
        stages.append(
            {
                "stage_id": stage_id,
                "status": status,
                "occurred_at": datetime.now(UTC).isoformat(),
                "duration_ms": (
                    round((time.monotonic() - started) * 1000, 3)
                    if started is not None
                    else None
                ),
                "inputs": inputs or {},
                "result": result or {},
            }
        )

    @app.middleware("http")
    async def add_target_event(request: Request, call_next: Any) -> Response:
        started = time.monotonic()
        if request.url.path.startswith("/internal/challenges/"):
            response = cast("Response", await call_next(request))
            response.headers["Cache-Control"] = "no-store"
            return response
        request.state.sqli_trace = []
        trace_stage(
            request,
            "request_received",
            status="observed",
            inputs={
                "method": request.method,
                "path": request.url.path,
                "request_id": request.headers.get("x-range-request-id"),
                "decision_id": request.headers.get("x-range-decision-id"),
            },
        )
        try:
            response = cast("Response", await call_next(request))
        except Exception:
            response = JSONResponse(status_code=500, content={"detail": "lab target error"})
        trace_stage(
            request,
            "http_response",
            status="completed" if response.status_code < 500 else "server_error",
            started=started,
            result={"http_status": response.status_code},
        )
        event_id = append_evidence(
            {
                "run_id": request.headers.get("x-range-run-id"),
                "episode_id": request.headers.get("x-range-episode-id"),
                "decision_id": request.headers.get("x-range-decision-id"),
                "request_id": request.headers.get("x-range-request-id"),
                "purpose": request.headers.get("x-range-purpose", "unspecified"),
                "route_id": "lookup" if request.url.path == "/api/lookup" else "health",
                "method": request.method,
                "actor_ref": request.headers.get("x-range-actor", "unknown"),
                "http_status": response.status_code,
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
                "controls": runtime.snapshot(),
                "trace_schema_id": "violet.server_trace.v1",
                "trace": request.state.sqli_trace,
                **getattr(request.state, "sqli_audit", {}),
            }
        )
        response.headers["X-Sqli-Target-Event-ID"] = event_id
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "target_id": "sqli_lab_v1", "version": TARGET_VERSION}

    @app.get("/api/lookup")
    async def lookup_description() -> dict[str, object]:
        return {"route_id": "lookup", "method": "POST", "target_version": TARGET_VERSION}

    @app.get("/internal/challenges/{challenge_id}/phrase")
    async def get_challenge_phrase(challenge_id: str) -> dict[str, str]:
        """Executor-only volatile read used to redact the phrase before spool writes."""
        phrase = runtime.get_challenge(challenge_id)
        if phrase is None:
            raise HTTPException(status_code=404, detail="challenge is not active")
        return {"phrase": phrase}

    @app.post("/internal/challenges/{challenge_id}")
    async def set_challenge_phrase(challenge_id: str, body: ChallengeSecretBody) -> dict[str, str]:
        runtime.set_challenge(challenge_id, body.phrase)
        return {"challenge_id": challenge_id, "status": "ready"}

    @app.delete("/internal/challenges/{challenge_id}")
    async def clear_challenge_phrase(challenge_id: str) -> dict[str, str]:
        runtime.clear_challenge(challenge_id)
        return {"challenge_id": challenge_id, "status": "cleared"}

    @app.post("/api/lookup")
    async def lookup(body: LookupBody, request: Request) -> JSONResponse:
        route_started = time.monotonic()
        trace_stage(
            request,
            "application_route",
            status="entered",
            started=route_started,
            inputs={"route_id": "lookup", "method": "POST"},
        )
        trace_stage(
            request,
            "request_validation",
            status="passed",
            inputs={"field": body.field, "query_value": body.query_value},
        )
        actor_ref = request.headers.get("x-range-actor", "guest")
        episode_id = request.headers.get("x-range-episode-id", "unscoped")
        challenge_id = request.headers.get("x-range-challenge-id")
        if actor_ref not in {"guest", "researcher"}:
            trace_stage(
                request,
                "authentication_and_access",
                status="rejected",
                inputs={"actor_ref": actor_ref},
                result={"reason": "unknown synthetic actor", "http_status": 400},
            )
            trace_stage(
                request,
                "database_operation",
                status="not_reached",
                result={"reason": "actor validation rejected the request"},
            )
            raise HTTPException(status_code=400, detail="unknown synthetic actor")
        trace_stage(
            request,
            "authentication_and_access",
            status="passed",
            inputs={"actor_ref": actor_ref},
            result={"principal_scope": "synthetic_guest" if actor_ref == "guest" else "researcher"},
        )
        status, reason = runtime.check_request(actor_ref, episode_id)
        trace_stage(
            request,
            "blue_controls",
            status="blocked" if status is not None else "passed",
            inputs={"actor_ref": actor_ref, "controls": runtime.snapshot()},
            result={"http_status": status, "reason": reason},
        )
        if status is not None:
            trace_stage(
                request,
                "database_operation",
                status="not_reached",
                result={"reason": "request stopped by an active server control"},
            )
            request.state.sqli_audit = {
                "field": body.field,
                "query_length": len(body.query_value),
                "query_syntax_signal": "'" in body.query_value and "OR" in body.query_value.upper(),
                "row_count": 0,
                "unauthorized_rows": 0,
                "query_rejected": False,
                "challenge_id": challenge_id,
                "challenge_disclosed": False,
            }
            return JSONResponse(status_code=status, content={"detail": reason, "rows": []})
        database_started = time.monotonic()
        try:
            rows = database.execute_lookup(
                actor_ref=actor_ref,
                field=body.field,
                query_value=body.query_value,
                parameterized=runtime.parameterized,
            )
        except Exception:
            # SQL errors are useful teaching evidence but never expose server internals.
            trace_stage(
                request,
                "database_operation",
                status="rejected",
                started=database_started,
                inputs={
                    "operation": "SELECT synthetic records",
                    "field": body.field,
                    "query_value": body.query_value,
                    "query_mode": "parameterized" if runtime.parameterized else "legacy query construction",
                },
                result={"error": "query rejected", "row_count": 0},
            )
            request.state.sqli_audit = {
                "field": body.field,
                "query_length": len(body.query_value),
                "query_syntax_signal": "'" in body.query_value and "OR" in body.query_value.upper(),
                "row_count": 0,
                "unauthorized_rows": 0,
                "query_rejected": True,
                "challenge_id": challenge_id,
                "challenge_disclosed": False,
            }
            return JSONResponse(status_code=400, content={"detail": "query rejected", "rows": []})
        trace_stage(
            request,
            "database_operation",
            status="completed",
            started=database_started,
            inputs={
                "operation": "SELECT synthetic records",
                "field": body.field,
                "query_value": body.query_value,
                "query_mode": "parameterized" if runtime.parameterized else "legacy query construction",
                "row_limit": 8,
            },
            result={
                "row_count": len(rows),
                "records": [
                    {"record_ref": row.get("record_ref"), "owner_ref": row.get("owner_ref")}
                    for row in rows
                ],
            },
        )
        unauthorized = sum(row["owner_ref"] != actor_ref for row in rows)
        challenge_phrase = (
            runtime.get_challenge(challenge_id)
            if challenge_id and actor_ref == "guest" and unauthorized > 0
            else None
        )
        challenge_disclosed = challenge_phrase is not None
        request.state.sqli_audit = {
            "field": body.field,
            "query_length": len(body.query_value),
            "query_syntax_signal": "'" in body.query_value and "OR" in body.query_value.upper(),
            "row_count": len(rows),
            "unauthorized_rows": unauthorized,
            "query_rejected": False,
            "challenge_id": challenge_id,
            "challenge_disclosed": challenge_disclosed,
        }
        content: dict[str, object] = {
            "rows": rows,
            "row_count": len(rows),
            "target_version": TARGET_VERSION,
            "controls": runtime.snapshot(),
        }
        if challenge_phrase is not None:
            # The fixed executor removes this value in memory before writing its result spool.
            content["challenge_phrase"] = challenge_phrase
        return JSONResponse(
            status_code=200,
            content=content,
        )

    return app


app = create_app()
