"""Local real-range API backed by the isolated executor and typed control spool."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from violet.persistence.repository import EventRepository
from violet.range.contracts import (
    CheckDocumentAccessAction,
    ControlAcknowledgement,
    ControlCommand,
    ControlRequest,
    ControlVerificationRequest,
    ExecutorJob,
    ExecutorJobRequest,
    ExecutorResult,
)
from violet.range.control_verifier import verify_ownership_control
from violet.range.spool import read_json, write_json_atomic
from violet.range.verifier import verify_document_access

router = APIRouter()
SPOOL = Path(os.getenv("VIOLET_EXECUTOR_SPOOL", "/tmp/violet-executor-spool"))
CONTROL_INBOX = Path(os.getenv("VIOLET_CONTROL_INBOX", "/tmp/violet-target-control/inbox"))
CONTROL_RESULTS = Path(os.getenv("VIOLET_CONTROL_RESULTS", "/tmp/violet-target-control/results"))
TARGET_EVIDENCE = Path(os.getenv("VIOLET_TARGET_EVIDENCE", "/tmp/violet-target-evidence"))
_REPOSITORY: EventRepository | None = None


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    run_id: UUID | None = None
    episode_id: UUID | None = None


def _new_job(request: ExecutorJobRequest) -> ExecutorJob:
    action_id = uuid4()
    return ExecutorJob(
        job_id=uuid4(),
        run_id=request.run_id or uuid4(),
        episode_id=request.episode_id or uuid4(),
        step_id=request.step_id,
        decision_id=request.decision_id or uuid4(),
        action_id=action_id,
        request_id=uuid4(),
        idempotency_key=action_id,
        action=request.action,
    )


def _enqueue(job: ExecutorJob) -> None:
    SPOOL.joinpath("jobs").mkdir(parents=True, exist_ok=True)
    records = SPOOL / "records"
    records.mkdir(parents=True, exist_ok=True)
    write_json_atomic(records / f"{job.job_id}.json", job.model_dump(mode="json"))
    write_json_atomic(SPOOL / "jobs" / f"{job.job_id}.json", job.model_dump(mode="json"))


def _repository() -> EventRepository:
    global _REPOSITORY
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=503, detail="range event store is not configured")
    if _REPOSITORY is None:
        repository = EventRepository(database_url)
        try:
            repository.initialize()
            _reconcile_outboxes(repository)
        except Exception:
            repository.close()
            raise HTTPException(
                status_code=503, detail="range event store is unavailable"
            ) from None
        _REPOSITORY = repository
    return _REPOSITORY


def initialize_range_store() -> None:
    """Initialize the event store and recover committed outbox records at startup."""
    if os.getenv("DATABASE_URL"):
        _repository()


def close_range_store() -> None:
    global _REPOSITORY
    if _REPOSITORY is not None:
        _REPOSITORY.close()
        _REPOSITORY = None


def _reconcile_outboxes(repository: EventRepository) -> None:
    for payload in repository.pending_jobs():
        job = ExecutorJob.model_validate(payload)
        records = SPOOL / "records"
        records.mkdir(parents=True, exist_ok=True)
        write_json_atomic(records / f"{job.job_id}.json", payload)
        result_path = SPOOL / "results" / f"{job.job_id}.json"
        processing_path = SPOOL / "processing" / f"{job.job_id}.json"
        job_path = SPOOL / "jobs" / f"{job.job_id}.json"
        if not result_path.exists() and not processing_path.exists() and not job_path.exists():
            job_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(job_path, payload)
    for payload in repository.pending_controls():
        command = ControlCommand.model_validate(payload)
        acknowledgement = CONTROL_RESULTS / f"{command.command_id}.json"
        command_path = CONTROL_INBOX / f"{command.command_id}.json"
        if not acknowledgement.exists() and not command_path.exists():
            _enqueue_control(command)


def _target_evidence(target_event_id: UUID | None) -> dict[str, Any] | None:
    if target_event_id is None:
        return None
    evidence_path = TARGET_EVIDENCE
    if evidence_path.is_dir() or not evidence_path.suffix:
        evidence_path = evidence_path / "target-access.jsonl"
    if not evidence_path.exists():
        return None
    try:
        with evidence_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("target_event_id") == str(target_event_id):
                    return item
    except OSError:
        return None
    return None


def _sync_executor_result(
    repository: EventRepository, job: ExecutorJob, payload: dict[str, Any] | None = None
) -> None:
    if payload is None:
        payload = repository.get_job(str(job.job_id))
    if payload is None or payload.get("result") is not None:
        return
    result_path = SPOOL / "results" / f"{job.job_id}.json"
    if not result_path.exists():
        return
    try:
        result = ExecutorResult.model_validate(read_json(result_path))
    except Exception:
        return
    finding = None
    if isinstance(job.action, CheckDocumentAccessAction):
        finding = verify_document_access(
            job, result, repository.fixture_seed(str(job.run_id), str(job.episode_id))
        )
    repository.save_execution(
        job,
        result,
        _target_evidence(result.target_event_id),
        finding,
    )


def _sync_completed_jobs(repository: EventRepository, run_id: str | None = None) -> None:
    for payload in repository.pending_jobs():
        try:
            job = ExecutorJob.model_validate(payload)
        except Exception:
            continue
        if run_id is None or str(job.run_id) == run_id:
            _sync_executor_result(repository, job)


def _sync_completed_controls(repository: EventRepository, run_id: str | None = None) -> None:
    for payload in repository.pending_controls():
        try:
            command = ControlCommand.model_validate(payload)
        except Exception:
            continue
        if run_id is not None and str(command.run_id) != run_id:
            continue
        acknowledgement_path = CONTROL_RESULTS / f"{command.command_id}.json"
        if not acknowledgement_path.exists():
            continue
        try:
            acknowledgement = ControlAcknowledgement.model_validate(read_json(acknowledgement_path))
        except Exception:
            continue
        repository.save_control_ack(acknowledgement)


def _enqueue_control(command: ControlCommand) -> None:
    CONTROL_INBOX.mkdir(parents=True, exist_ok=True)
    write_json_atomic(CONTROL_INBOX / f"{command.command_id}.json", command.model_dump(mode="json"))


@router.get("/health")
async def range_health() -> Response:
    health_path = SPOOL / "health.json"
    if not health_path.exists():
        return Response(
            content=json.dumps({"status": "unavailable", "reason": "executor_not_started"}),
            status_code=503,
            media_type="application/json",
        )
    try:
        value = read_json(health_path)
    except (OSError, ValueError):
        raise HTTPException(status_code=503, detail="executor health record is invalid") from None
    try:
        last_updated = datetime.fromisoformat(value["updated_at"])
        age_seconds = (datetime.now(UTC) - last_updated.astimezone(UTC)).total_seconds()
    except (KeyError, TypeError, ValueError):
        raise HTTPException(
            status_code=503, detail="executor health record has no valid timestamp"
        ) from None
    target_state = value.get("target") or {}
    if (
        age_seconds > 15
        or not target_state
        or value.get("executor_status") != "running"
        or target_state.get("status") != "completed"
        or target_state.get("http_status") != 200
    ):
        return Response(content=json.dumps(value), status_code=503, media_type="application/json")
    return Response(content=json.dumps({"status": "ok", **value}), media_type="application/json")


@router.post("/jobs", status_code=202)
async def submit_job(request: ExecutorJobRequest) -> dict[str, Any]:
    job = _new_job(request)
    repository = _repository()
    try:
        repository.submit_job(job)
        _enqueue(job)
    except Exception:
        raise HTTPException(
            status_code=503, detail="range job could not be persisted or queued"
        ) from None
    return {
        "job_id": str(job.job_id),
        "status": "queued",
        "run_id": str(job.run_id),
        "episode_id": str(job.episode_id),
        "decision_id": str(job.decision_id),
        "action_id": str(job.action_id),
        "request_id": str(job.request_id),
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: UUID) -> dict[str, Any]:
    repository = _repository()
    persisted = repository.get_job(str(job_id))
    if persisted is None:
        raise HTTPException(status_code=404, detail="range job not found")
    record_path = SPOOL / "records" / f"{job_id}.json"
    result_path = SPOOL / "results" / f"{job_id}.json"
    if not record_path.exists():
        job = ExecutorJob.model_validate(persisted["payload"])
        record_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(record_path, persisted["payload"])
    else:
        job = ExecutorJob.model_validate(read_json(record_path))
    if persisted.get("result") is None and result_path.exists():
        _sync_executor_result(repository, job, persisted)
        persisted = repository.get_job(str(job_id)) or persisted
    if persisted.get("result") is None:
        _reconcile_outboxes(repository)
        return {"job_id": str(job_id), "status": "queued_or_running"}
    result = ExecutorResult.model_validate(persisted["result"])
    value: dict[str, Any] = {
        "job": job.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
    }
    if isinstance(job.action, CheckDocumentAccessAction):
        value["finding"] = verify_document_access(
            job, result, repository.fixture_seed(str(job.run_id), str(job.episode_id))
        )
    return value


@router.get("/runs/{run_id}/events")
async def get_run_events(
    run_id: UUID,
    after_seq: int = 0,
    limit: int = 500,
) -> dict[str, Any]:
    if after_seq < 0 or limit < 1 or limit > 1000:
        raise HTTPException(status_code=422, detail="invalid event replay cursor or limit")
    repository = _repository()
    _sync_completed_jobs(repository, str(run_id))
    _sync_completed_controls(repository, str(run_id))
    events = repository.events(str(run_id), after_seq=after_seq, limit=limit)
    return {"run_id": str(run_id), "after_seq": after_seq, "events": events}


@router.post("/reset", status_code=202)
async def reset_target(request: ResetRequest) -> dict[str, str]:
    run_id, episode_id = request.run_id or uuid4(), request.episode_id or uuid4()
    command = ControlCommand(
        command_id=uuid4(),
        command="RESET",
        seed=request.seed,
        run_id=run_id,
        episode_id=episode_id,
    )
    try:
        _repository().submit_control(command)
        _enqueue_control(command)
    except Exception:
        raise HTTPException(
            status_code=503, detail="target reset could not be persisted or queued"
        ) from None
    return {
        "command_id": str(command.command_id),
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "status": "queued",
    }


@router.post("/controls", status_code=202)
async def apply_control(request: ControlRequest) -> dict[str, str]:
    run_id, episode_id = request.run_id or uuid4(), request.episode_id or uuid4()
    command = ControlCommand(
        command_id=uuid4(),
        command="APPLY_CONTROL",
        action=request.action,
        principal_ref=request.principal_ref,
        run_id=run_id,
        episode_id=episode_id,
    )
    try:
        _repository().submit_control(command)
        _enqueue_control(command)
    except Exception:
        raise HTTPException(
            status_code=503, detail="target control could not be persisted or queued"
        ) from None
    return {
        "command_id": str(command.command_id),
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "status": "queued",
    }


@router.get("/controls/{command_id}")
async def get_control_ack(command_id: UUID) -> dict[str, Any]:
    repository = _repository()
    persisted = repository.get_control(str(command_id))
    if persisted is None:
        raise HTTPException(status_code=404, detail="target control command not found")
    path = CONTROL_RESULTS / f"{command_id}.json"
    if not path.exists():
        _reconcile_outboxes(repository)
        if persisted.get("acknowledgement") is not None:
            return persisted["acknowledgement"]
        return {"command_id": str(command_id), "status": "queued_or_running"}
    try:
        acknowledgement = ControlAcknowledgement.model_validate(read_json(path))
    except Exception:
        raise HTTPException(status_code=502, detail="target acknowledgement is invalid") from None
    repository.save_control_ack(acknowledgement)
    return acknowledgement.model_dump(mode="json")


@router.post("/controls/{command_id}/verify")
async def verify_control(command_id: UUID, request: ControlVerificationRequest) -> dict[str, Any]:
    repository = _repository()
    _sync_completed_controls(repository)
    control = repository.get_control(str(command_id))
    if control is None:
        raise HTTPException(status_code=404, detail="target control command not found")
    if control.get("acknowledgement") is None:
        raise HTTPException(status_code=409, detail="target control has no acknowledgement")
    command = ControlCommand.model_validate(control["payload"])
    if command.action != "HARDEN_DOCUMENT_OWNERSHIP":
        raise HTTPException(
            status_code=422, detail="this verification requires ownership hardening"
        )
    acknowledgement = ControlAcknowledgement.model_validate(control["acknowledgement"])
    _sync_completed_jobs(repository, str(command.run_id))
    unauthorized_record = repository.get_job(str(request.unauthorized_job_id))
    owner_record = repository.get_job(str(request.owner_job_id))
    if unauthorized_record is None or owner_record is None:
        raise HTTPException(status_code=404, detail="verification job not found")
    if unauthorized_record.get("result") is None or owner_record.get("result") is None:
        raise HTTPException(status_code=409, detail="verification jobs are not complete")
    unauthorized_job = ExecutorJob.model_validate(unauthorized_record["payload"])
    owner_job = ExecutorJob.model_validate(owner_record["payload"])
    unauthorized_result = ExecutorResult.model_validate(unauthorized_record["result"])
    owner_result = ExecutorResult.model_validate(owner_record["result"])
    ack_seq = repository.event_seq(
        str(command.run_id), "target_control_acknowledged", "command_id", str(command_id)
    )
    alice_seq = repository.event_seq(
        str(command.run_id), "action_submitted", "job_id", str(unauthorized_job.job_id)
    )
    bob_seq = repository.event_seq(
        str(command.run_id), "action_submitted", "job_id", str(owner_job.job_id)
    )
    if ack_seq is None or alice_seq is None or bob_seq is None:
        raise HTTPException(status_code=409, detail="control or request events are not persisted")
    if alice_seq <= ack_seq or bob_seq <= ack_seq:
        raise HTTPException(status_code=409, detail="verification requests must follow the control")
    verification = verify_ownership_control(
        command,
        acknowledgement,
        unauthorized_job,
        unauthorized_result,
        owner_job,
        owner_result,
    )
    try:
        repository.record_control_verification(str(command_id), verification)
    except ValueError:
        raise HTTPException(
            status_code=409, detail="control verification could not be stored"
        ) from None
    return verification
