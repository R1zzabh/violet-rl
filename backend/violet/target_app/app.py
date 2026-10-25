"""The actual Violet Ledger HTTP target used by the bounded training range."""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from violet.target_app.control import TargetRuntime, watch_control_directory
from violet.target_app.database import TargetDatabase
from violet.target_app.security import authenticate, read_document
from violet.target_app.telemetry import TargetTelemetry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


class LoginBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    credential_ref: Literal["synthetic_alice", "synthetic_bob", "synthetic_invalid"]


def create_app(
    database_url: str | None = None,
    evidence_path: Path | None = None,
    control_inbox: Path | None = None,
    control_results: Path | None = None,
) -> FastAPI:
    resolved_database_url = (
        database_url
        or os.getenv("TARGET_DATABASE_URL")
        or "sqlite:////tmp/violet-ledger-target.sqlite3"
    )
    evidence_path = evidence_path or Path(
        os.getenv("TARGET_EVIDENCE_PATH", "/tmp/violet-ledger-evidence.jsonl")
    )
    control_inbox = control_inbox or Path(
        os.getenv("TARGET_CONTROL_INBOX", "/tmp/violet-target-control/inbox")
    )
    control_results = control_results or Path(
        os.getenv("TARGET_CONTROL_RESULTS", "/tmp/violet-target-control/results")
    )
    database = TargetDatabase(resolved_database_url)
    runtime = TargetRuntime(database)
    telemetry = TargetTelemetry(evidence_path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(watch_control_directory(runtime, control_inbox, control_results))
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            database.engine.dispose()

    target = FastAPI(title="Violet Ledger", version="1.0.0", lifespan=lifespan)
    target.state.runtime = runtime
    target.state.telemetry = telemetry

    @target.middleware("http")
    async def correlate_and_audit(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.monotonic()
        request_id = request.headers.get("x-violet-request-id")
        correlation = {
            "run_id": request.headers.get("x-violet-run-id"),
            "episode_id": request.headers.get("x-violet-episode-id"),
            "step_id": request.headers.get("x-violet-step-id"),
            "decision_id": request.headers.get("x-violet-decision-id"),
            "action_id": request.headers.get("x-violet-action-id"),
            "request_id": request_id,
        }
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(status_code=500, content={"detail": "target error"})
        principal = None
        session_token = request.cookies.get("violet_session")
        if session_token:
            principal = runtime.sessions.get(session_token)
        route_id = (
            "home"
            if request.url.path == "/"
            else (
                "login"
                if request.url.path == "/api/login"
                else "document"
                if request.url.path.startswith("/api/documents/")
                else "health"
                if request.url.path == "/health"
                else "unknown"
            )
        )
        event_id = telemetry.record(
            {
                **correlation,
                "route_id": route_id,
                "method": request.method,
                "status_class": response.status_code // 100,
                "http_status": response.status_code,
                "principal_ref": principal,
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
            }
        )
        response.headers["X-Violet-Target-Event-ID"] = event_id
        return response

    @target.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "target_id": "violet_ledger_v1", "version": "1.0.0"}

    @target.get("/", response_class=HTMLResponse)
    async def home() -> str:
        return """<!doctype html><html><head><title>Violet Ledger</title></head>
<body><main><h1>Violet Ledger</h1><p>Local synthetic records training target.</p>
<p>Use the training console to access Alice and Bob's sample documents.</p></main></body></html>"""

    @target.get("/api/login")
    async def login_probe() -> dict[str, str]:
        return {"route_id": "login", "method": "POST", "purpose": "synthetic account sign-in"}

    @target.post("/api/login")
    async def login(body: LoginBody, response: Response) -> dict[str, str]:
        principal = authenticate(runtime, body.credential_ref)
        token = secrets.token_urlsafe(32)
        runtime.sessions[token] = principal
        response.set_cookie(
            "violet_session", token, httponly=True, samesite="strict", secure=False, path="/"
        )
        return {"authenticated": "true", "principal_ref": principal}

    @target.get("/api/documents/{document_ref}")
    async def get_document(document_ref: str, request: Request) -> dict[str, str]:
        if document_ref not in {"doc_alice", "doc_bob"}:
            raise HTTPException(status_code=404, detail="document not found")
        token = request.cookies.get("violet_session")
        principal = runtime.sessions.get(token) if token else None
        return read_document(runtime, principal, document_ref)

    return target


app = create_app()
