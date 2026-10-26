"""Local real-range API backed by the isolated executor and typed control spool."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from violet.persistence.repository import EventRepository
from violet.range.contracts import (
    AgentDecision,
    BlueAction,
    CheckDocumentAccessAction,
    ControlAcknowledgement,
    ControlCommand,
    ControlRequest,
    ControlVerificationRequest,
    ExecutorJob,
    ExecutorJobRequest,
    ExecutorResult,
    RangeAction,
    RangeObservation,
    TransitionSubmission,
    WaitAction,
)
from violet.range.control_verifier import submission_event_type, verify_ownership_control
from violet.range.detection import detect
from violet.range.observations import (
    BLUE_ACTION_CATALOG_V2,
    FEATURE_ENCODING_SCHEMA_ID,
    RED_ACTION_CATALOG_V2,
    build_blue_observation,
    build_red_observation,
    encode_observation,
)
from violet.range.rewards import blue_reward, red_reward
from violet.range.spool import read_json, write_json_atomic
from violet.range.verifier import verify_document_access
from violet.sqli_lab.contracts import (
    SCENARIO_ID as SQLI_SCENARIO_ID,
)
from violet.sqli_lab.contracts import (
    TARGET_VERSION as SQLI_TARGET_VERSION,
)
from violet.sqli_lab.contracts import (
    SqliControlCommand,
    SqliExecutorResult,
    SqliJob,
    SqliRequestAction,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

router = APIRouter()
# Container-local IPC paths are explicit, isolated mounts in docker-compose.yml.
SPOOL = Path(os.getenv("VIOLET_EXECUTOR_SPOOL", "/tmp/violet-executor-spool"))  # nosec B108
CONTROL_INBOX = Path(os.getenv("VIOLET_CONTROL_INBOX", "/tmp/violet-target-control/inbox"))  # nosec B108
CONTROL_RESULTS = Path(os.getenv("VIOLET_CONTROL_RESULTS", "/tmp/violet-target-control/results"))  # nosec B108
TARGET_EVIDENCE = Path(os.getenv("VIOLET_TARGET_EVIDENCE", "/tmp/violet-target-evidence"))  # nosec B108
SQLI_SPOOL = Path(os.getenv("SQLI_EXECUTOR_SPOOL", "/tmp/sqli-executor-spool"))  # nosec B108
SQLI_CONTROL_INBOX = Path(os.getenv("SQLI_CONTROL_INBOX", "/tmp/sqli-control/inbox"))  # nosec B108
SQLI_CONTROL_RESULTS = Path(os.getenv("SQLI_CONTROL_RESULTS", "/tmp/sqli-control/results"))  # nosec B108
SQLI_TARGET_EVIDENCE = Path(os.getenv("SQLI_TARGET_EVIDENCE", "/tmp/sqli-evidence/events.jsonl"))  # nosec B108
_REPOSITORY: EventRepository | None = None


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    run_id: UUID | None = None
    episode_id: UUID | None = None
    command_id: UUID | None = None


class DecisionSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: AgentDecision
    observation: RangeObservation

    @model_validator(mode="after")
    def validate_pair(self) -> DecisionSubmission:
        if (
            self.decision.run_id != self.observation.run_id
            or self.decision.episode_id != self.observation.episode_id
            or self.decision.step_id != self.observation.step_id
            or self.decision.observation_schema_id != self.observation.schema_id
            or self.decision.legal_action_mask != self.observation.legal_action_mask
        ):
            raise ValueError("decision does not match its recorded observation and legal mask")
        sample = self.decision.policy_sample
        if sample is not None:
            encoded = encode_observation(self.observation)
            if len(sample.encoded_observation) != len(encoded) or any(
                abs(expected - actual) > 1e-5
                for expected, actual in zip(encoded, sample.encoded_observation, strict=True)
            ):
                raise ValueError("policy sample does not match the structured observation")
        if self.decision.role == "red":
            if len(self.decision.legal_action_mask) != len(RED_ACTION_CATALOG_V2):
                raise ValueError("Red decision mask does not match the range action catalog")
            action: RangeAction = TypeAdapter(RangeAction).validate_python(
                self.decision.selected_action
            )
            if action.operation == "DISCOVER_ROUTE":
                catalog_id = f"DISCOVER_ROUTE:{action.route_candidate_id}"
            elif action.operation == "AUTH_TEST":
                catalog_id = f"AUTH_TEST:{action.credential_ref}"
            elif action.operation == "CHECK_DOCUMENT_ACCESS":
                catalog_id = f"CHECK_DOCUMENT_ACCESS:{action.document_ref}"
            elif action.operation == "CHECK_TARGET_HEALTH":
                catalog_id = "CHECK_TARGET_HEALTH"
            else:
                catalog_id = "WAIT"
            if catalog_id not in RED_ACTION_CATALOG_V2:
                raise ValueError("selected Red action is outside the catalog")
            if not self.decision.legal_action_mask[RED_ACTION_CATALOG_V2.index(catalog_id)]:
                raise ValueError("selected Red action is masked as illegal")
            if self.decision.policy_sample is not None and (
                self.decision.policy_sample.encoding_schema_id != FEATURE_ENCODING_SCHEMA_ID
                or self.decision.policy_sample.selected_action_index
                != RED_ACTION_CATALOG_V2.index(catalog_id)
            ):
                raise ValueError("Red sampled action does not match the recorded action")
            if action.operation == "CHECK_DOCUMENT_ACCESS":
                aliases: set[str] = set()

                def collect_aliases(value: Any) -> None:
                    if isinstance(value, dict):
                        alias = value.get("session_alias")
                        if isinstance(alias, str):
                            aliases.add(alias)
                        for nested in value.values():
                            collect_aliases(nested)
                    elif isinstance(value, (list, tuple)):
                        for nested in value:
                            collect_aliases(nested)

                collect_aliases(self.observation.recent_events)
                if not action.session_alias or action.session_alias not in aliases:
                    raise ValueError(
                        "document access requires a session alias present in the observation"
                    )
        else:
            if len(self.decision.legal_action_mask) != len(BLUE_ACTION_CATALOG_V2):
                raise ValueError("Blue decision mask does not match the control action catalog")
            blue_action = BlueAction.model_validate(self.decision.selected_action)
            catalog_id = blue_action.operation
            if blue_action.principal_ref is not None:
                catalog_id = f"{blue_action.operation}:{blue_action.principal_ref}"
            if catalog_id not in BLUE_ACTION_CATALOG_V2:
                raise ValueError("selected Blue action is outside the catalog")
            if not self.decision.legal_action_mask[BLUE_ACTION_CATALOG_V2.index(catalog_id)]:
                raise ValueError("selected Blue action is masked as illegal")
            if self.decision.policy_sample is not None and (
                self.decision.policy_sample.encoding_schema_id != FEATURE_ENCODING_SCHEMA_ID
                or self.decision.policy_sample.selected_action_index
                != BLUE_ACTION_CATALOG_V2.index(catalog_id)
            ):
                raise ValueError("Blue sampled action does not match the recorded action")
        return self


class RewardSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    step_id: int = Field(ge=0, le=500)
    job_id: UUID | None = None
    decision_id: UUID | None = None
    source_target_event_id: UUID | None = None
    control_command_id: UUID | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> RewardSubmission:
        if self.role == "red" and (
            (self.job_id is None) == (self.decision_id is None)
            or self.source_target_event_id
            or self.control_command_id
        ):
            raise ValueError(
                "Red reward requires either a completed job or a persisted WAIT decision"
            )
        if self.role == "blue" and (self.job_id is not None or self.decision_id is None):
            raise ValueError("Blue reward requires its persisted decision ID")
        if self.role not in {"red", "blue"}:
            raise ValueError("role must be red or blue")
        return self


class TrainingRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training_run_id: UUID | None = None
    checkpoint_run_id: UUID | None = None
    role: Literal["red", "blue", "joint"]
    mode: Literal["train", "evaluate"] = "train"
    run_kind: Literal["teaching", "research"] = "teaching"
    scenario_id: Literal["violet_ledger_v1", "sqli_lab_v1"] = "violet_ledger_v1"
    episodes: int = Field(default=1, ge=1, le=10000)
    max_steps: int = Field(default=12, ge=1, le=12)
    seed: int = Field(default=17, ge=0, le=2**31 - 1)
    opponent: Literal["scripted", "random", "none"] = "scripted"
    learning_rate: float = Field(default=0.0003, gt=0, le=0.01)

    @model_validator(mode="after")
    def require_evaluation_checkpoint(self) -> TrainingRunRequest:
        if self.mode == "train" and self.role != "joint":
            raise ValueError("training updates Red and Blue together at each episode boundary")
        if self.scenario_id == SQLI_SCENARIO_ID and self.role != "joint":
            raise ValueError(
                "the SQLi learning lab trains and evaluates the joint Red/Blue PPO pair"
            )
        if self.scenario_id == SQLI_SCENARIO_ID and self.opponent != "none":
            raise ValueError("scripted and random policies are excluded from SQLi classroom runs")
        if self.scenario_id != SQLI_SCENARIO_ID and self.opponent == "none":
            raise ValueError("the Violet Ledger scenario requires an explicitly labeled comparator")
        if self.mode == "evaluate" and self.checkpoint_run_id is None:
            raise ValueError("evaluation requires a completed compatible checkpoint run ID")
        if self.run_kind == "research" and self.mode == "train" and self.episodes < 10000:
            raise ValueError("research training requires at least 10,000 episodes per seed")
        if self.run_kind == "research" and self.mode == "evaluate" and self.episodes < 100:
            raise ValueError("research evaluation requires at least 100 paired held-out episodes")
        return self


class WorkerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_id: str = Field(min_length=1, max_length=64, pattern="^[a-zA-Z0-9._-]+$")
    status: str = Field(pattern="^(running|completed|stopped|failed)$")
    result: dict[str, Any] | None = None


class TrainingUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_id: str = Field(min_length=1, max_length=64, pattern="^[a-zA-Z0-9._-]+$")
    role: Literal["red", "blue"] | None = None
    status: str = Field(default="running", pattern="^running$")
    result: dict[str, Any]


class EpisodeFinish(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = Field(pattern="^(completed|incomplete|failed)$")
    reason: str | None = Field(default=None, max_length=128)


class SqliEventSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: UUID
    run_id: UUID
    episode_id: UUID
    seed: int = Field(ge=0, le=2**31 - 1)
    event_type: Literal[
        "sqli_episode_started",
        "sqli_red_decision",
        "sqli_target_response",
        "sqli_blue_decision",
        "sqli_control_acknowledged",
        "sqli_verification",
        "sqli_reward",
        "sqli_transition",
        "sqli_policy_probe",
        "sqli_ppo_update",
        "sqli_episode_summary",
    ]
    payload: dict[str, Any]

    @model_validator(mode="after")
    def payload_is_bounded(self) -> SqliEventSubmission:
        if len(json.dumps(self.payload, separators=(",", ":"))) > 48_000:
            raise ValueError("event payload exceeds the 48 KB lab event limit")
        return self


class SqliJobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: UUID
    episode_id: UUID
    decision_id: UUID
    step_id: int = Field(ge=0, le=40)
    action: SqliRequestAction
    purpose: Literal["policy", "verifier_attack", "verifier_legitimate"] = "policy"


class SqliControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: UUID
    episode_id: UUID
    decision_id: UUID | None = None
    action: Literal["RESET", "PARAMETERIZE_QUERY", "RATE_LIMIT_GUEST", "ISOLATE_GUEST_LOOKUP"]


def _require_trainer(request: Any) -> None:
    expected = os.getenv("VIOLET_TRAINER_TOKEN")
    supplied = request.headers.get("x-violet-trainer-token")
    if not expected or supplied != expected:
        raise HTTPException(status_code=403, detail="trainer worker is not authorized")


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
        idempotency_key=request.idempotency_key or action_id,
        purpose=request.purpose,
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
                if isinstance(item, dict) and item.get("target_event_id") == str(target_event_id):
                    return item
    except OSError:
        return None
    return None


def _sqli_telemetry(result: SqliExecutorResult) -> dict[str, object]:
    """Join target-owned telemetry only after checking the executor correlation IDs."""
    if result.target_event_id is None:
        return {"evidence_status": "missing"}
    path = SQLI_TARGET_EVIDENCE
    if path.is_dir() or not path.suffix:
        path = path / "events.jsonl"
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict) or item.get("target_event_id") != str(
                    result.target_event_id
                ):
                    continue
                if (
                    item.get("run_id") != str(result.run_id)
                    or item.get("episode_id") != str(result.episode_id)
                    or item.get("request_id") != str(result.request_id)
                ):
                    return {"evidence_status": "correlation_mismatch"}
                allowed = (
                    "route_id",
                    "method",
                    "actor_ref",
                    "http_status",
                    "duration_ms",
                    "field",
                    "query_length",
                    "query_syntax_signal",
                    "row_count",
                    "query_rejected",
                    "controls",
                )
                return {
                    "evidence_status": "verified",
                    **{key: item[key] for key in allowed if key in item},
                }
    except OSError:
        pass
    return {"evidence_status": "missing"}


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
    target_evidence = _target_evidence(result.target_event_id)
    finding = None
    if isinstance(job.action, CheckDocumentAccessAction):
        finding = verify_document_access(
            job,
            result,
            repository.fixture_seed(str(job.run_id), str(job.episode_id)),
            target_evidence,
        )
    repository.save_execution(
        job,
        result,
        target_evidence,
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


@router.get("/target/status")
async def target_status() -> dict[str, Any]:
    """Expose target health and acknowledged controls without fixture or session data."""
    health_path = SPOOL / "health.json"
    try:
        health_data = read_json(health_path) if health_path.exists() else {}
    except (OSError, ValueError):
        health_data = {}
    health: dict[str, Any] = health_data if isinstance(health_data, dict) else {}
    raw_target = health.get("target")
    target: dict[str, Any] = raw_target if isinstance(raw_target, dict) else {}
    state = _repository().latest_target_control_state()
    age_seconds = None
    if health.get("updated_at"):
        try:
            updated_at = datetime.fromisoformat(health["updated_at"])
            age_seconds = max(0.0, (datetime.now(UTC) - updated_at.astimezone(UTC)).total_seconds())
        except (TypeError, ValueError):
            age_seconds = None
    available = bool(
        health.get("executor_status") == "running"
        and target.get("status") == "completed"
        and target.get("http_status") == 200
        and age_seconds is not None
        and age_seconds <= 15
    )
    return {
        "schema_id": "violet.target_status.v1",
        "target_id": "violet_ledger_v1",
        "target_version": "violet_ledger_v1",
        "status": "available" if available else "unavailable",
        "last_health_check_age_seconds": age_seconds,
        "state_version": state["state_version"],
        "active_controls": state["controls"],
    }


@router.post("/jobs", status_code=202)
async def submit_job(request: ExecutorJobRequest) -> dict[str, Any]:
    if isinstance(request.action, WaitAction):
        raise HTTPException(
            status_code=422, detail="WAIT is a no-request decision and cannot be queued"
        )
    repository = _repository()
    if request.idempotency_key is not None:
        existing = repository.get_job_by_idempotency(str(request.idempotency_key))
        if existing is not None:
            persisted_job = ExecutorJob.model_validate(existing["payload"])
            if (
                persisted_job.action.model_dump(mode="json")
                != request.action.model_dump(mode="json")
                or persisted_job.purpose != request.purpose
                or (request.run_id is not None and str(persisted_job.run_id) != str(request.run_id))
                or (
                    request.episode_id is not None
                    and str(persisted_job.episode_id) != str(request.episode_id)
                )
                or persisted_job.step_id != request.step_id
                or (
                    request.decision_id is not None
                    and persisted_job.decision_id != request.decision_id
                )
            ):
                raise HTTPException(
                    status_code=409, detail="idempotency key was reused for different job content"
                )
            return {
                "job_id": str(persisted_job.job_id),
                "status": existing["status"],
                "run_id": str(persisted_job.run_id),
                "episode_id": str(persisted_job.episode_id),
                "decision_id": str(persisted_job.decision_id),
                "action_id": str(persisted_job.action_id),
                "request_id": str(persisted_job.request_id),
            }
    if request.purpose == "policy" and request.decision_id is None:
        raise HTTPException(status_code=409, detail="policy jobs require a persisted decision")
    if request.decision_id is not None:
        decision = repository.get_decision(str(request.decision_id))
        if (
            decision is None
            or decision["role"] != "red"
            or decision["selected_action"] != request.action.model_dump(mode="json")
            or (request.run_id is not None and decision["run_id"] != str(request.run_id))
            or (
                request.episode_id is not None and decision["episode_id"] != str(request.episode_id)
            )
            or decision["step_id"] != request.step_id
        ):
            raise HTTPException(status_code=409, detail="job must match a persisted Red decision")
    job = _new_job(request)
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
    public_result = result.model_dump(mode="json")
    value: dict[str, Any] = {
        "job": job.model_dump(mode="json"),
        "result": public_result,
    }
    if isinstance(job.action, CheckDocumentAccessAction):
        value["finding"] = verify_document_access(
            job,
            result,
            repository.fixture_seed(str(job.run_id), str(job.episode_id)),
            _target_evidence(result.target_event_id),
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


@router.get("/runs/{run_id}/stream")
async def stream_run_events(
    run_id: UUID, request: Request, after_seq: int = 0
) -> StreamingResponse:
    """Replay committed events, then stream only later committed sequence numbers."""
    if after_seq < 0:
        raise HTTPException(status_code=422, detail="invalid event replay cursor")
    last_event_id = request.headers.get("last-event-id")
    if last_event_id:
        try:
            after_seq = max(after_seq, int(last_event_id))
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Last-Event-ID must be an integer sequence"
            ) from None
    repository = _repository()

    async def committed_events() -> AsyncIterator[str]:
        cursor = after_seq
        while not await request.is_disconnected():
            _sync_completed_jobs(repository, str(run_id))
            _sync_completed_controls(repository, str(run_id))
            batch = repository.events(str(run_id), after_seq=cursor, limit=500)
            if batch:
                for event in batch:
                    cursor = int(event["seq"])
                    yield f"id: {cursor}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"
            else:
                yield ": committed-event heartbeat\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        committed_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.get("/runs/{run_id}/episodes")
async def get_run_episodes(run_id: UUID, limit: int = 100) -> dict[str, Any]:
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    return {"run_id": str(run_id), "items": _repository().list_episodes(str(run_id), limit)}


@router.get("/runs/{run_id}/episodes/{episode_id}/timeline")
async def get_episode_timeline(
    run_id: UUID,
    episode_id: UUID,
    after_seq: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    if after_seq < 0 or not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="invalid episode timeline cursor or limit")
    repository = _repository()
    if not repository.episode_exists(str(run_id), str(episode_id)):
        raise HTTPException(status_code=404, detail="range episode not found")
    items = repository.episode_events_page(
        str(run_id), str(episode_id), after_seq=after_seq, limit=limit
    )
    return {
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "after_seq": after_seq,
        "next_after_seq": items[-1]["seq"] if items else after_seq,
        "items": items,
    }


@router.get("/runs/{run_id}/sqli/summary")
async def get_sqli_run_summary(
    run_id: UUID, after_seq: int = 0, limit: int = 10000
) -> dict[str, Any]:
    if after_seq < 0 or not 1 <= limit <= 50000:
        raise HTTPException(status_code=422, detail="invalid summary cursor or limit")
    repository = _repository()
    run = repository.get_run(str(run_id))
    if run is None or run.get("target_version") != SQLI_TARGET_VERSION:
        raise HTTPException(status_code=404, detail="SQLi lab run not found")
    items = repository.lab_summary_events(str(run_id), after_seq, limit)
    return {
        "run_id": str(run_id),
        "after_seq": after_seq,
        "next_after_seq": items[-1]["seq"] if items else after_seq,
        "items": items,
    }


@router.post("/training-runs", status_code=202)
async def create_training_run(request: TrainingRunRequest) -> dict[str, Any]:
    if request.mode == "evaluate" and request.checkpoint_run_id is not None:
        source = _repository().get_training_run(str(request.checkpoint_run_id))
        if (
            source is None
            or source["status"] != "completed"
            or source["role"] != request.role
            or not (source.get("result") or {}).get("checkpoint_sha256")
        ):
            raise HTTPException(
                status_code=422,
                detail="evaluation checkpoint must belong to a completed run for the same role",
            )
        source_result = source.get("result") or {} if source else {}
        source_config = source.get("config") or {} if source else {}
        if source_config.get("scenario_id", "violet_ledger_v1") != request.scenario_id:
            raise HTTPException(
                status_code=422,
                detail="evaluation checkpoint belongs to an incompatible target scenario",
            )
        if request.run_kind == "research" and (
            source_config.get("run_kind") != "research"
            or int(source_result.get("completed_training_episodes", 0)) < 10000
        ):
            raise HTTPException(
                status_code=422,
                detail="research evaluation requires a research checkpoint with 10,000 completed training episodes",
            )
    training_run_id = str(request.training_run_id or uuid4())
    sqli = request.scenario_id == SQLI_SCENARIO_ID
    config = {
        "episodes": request.episodes,
        "run_kind": request.run_kind,
        "scenario_id": request.scenario_id,
        "max_steps": request.max_steps,
        "seed": request.seed,
        "opponent": "none" if sqli else request.opponent,
        "learning_rate": request.learning_rate,
        "checkpoint_run_id": str(request.checkpoint_run_id) if request.checkpoint_run_id else None,
        "target_version": SQLI_TARGET_VERSION if sqli else "violet_ledger_v1",
        "observation_schema_ids": (
            {role: f"{role}.sqli.range_observation.v1" for role in ("red", "blue")}
            if request.role == "joint"
            else {request.role: f"{request.role}.sqli.range_observation.v1"}
        )
        if sqli
        else (
            [f"{role}.range_observation.v2" for role in ("red", "blue")]
            if request.role == "joint"
            else [f"{request.role}.range_observation.v2"]
        ),
        "action_catalog_schema_ids": {
            "red": "sqli.red_action_grammar.v1",
            "blue": "sqli.blue_action_catalog.v1",
        }
        if sqli
        else (
            {role: f"{role}.action_catalog.v2" for role in ("red", "blue")}
            if request.role == "joint"
            else {request.role: f"{request.role}.action_catalog.v2"}
        ),
        "observation_schema_id": "sqli.range_observation.v1" if sqli else None,
        "action_schema_id": "sqli.composed_action.v1" if sqli else None,
        "reward_schema_id": "sqli.reward.v1" if sqli else "violet.range.reward.v2",
        "algorithm": ("factorized_red_categorical_ppo" if sqli else "masked_categorical_ppo")
        if request.mode == "train"
        else "policy_evaluation",
    }
    try:
        return _repository().create_training_run(
            training_run_id, request.role, request.mode, config
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.get("/training-runs")
async def list_training_runs(limit: int = 50) -> dict[str, Any]:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    return {"items": _repository().list_training_runs(limit)}


@router.get("/training-runs/{training_run_id}")
async def get_training_run(training_run_id: UUID) -> dict[str, Any]:
    record = _repository().get_training_run(str(training_run_id))
    if record is None:
        raise HTTPException(status_code=404, detail="training run not found")
    return record


@router.get("/training-runs/{training_run_id}/updates")
async def get_training_updates(training_run_id: UUID, role: str | None = None) -> dict[str, Any]:
    if role is not None and role not in {"red", "blue"}:
        raise HTTPException(status_code=422, detail="role must be red or blue")
    repository = _repository()
    if repository.get_training_run(str(training_run_id)) is None:
        raise HTTPException(status_code=404, detail="training run not found")
    return {
        "training_run_id": str(training_run_id),
        "items": repository.training_updates(str(training_run_id), role),
    }


@router.get("/sqli/target/status")
async def sqli_target_status() -> dict[str, Any]:
    health_path = SQLI_SPOOL / "health.json"
    if not health_path.exists():
        return {
            "scenario_id": SQLI_SCENARIO_ID,
            "target_version": SQLI_TARGET_VERSION,
            "status": "unavailable",
            "reason": "sqli_executor_not_started",
        }
    try:
        health = read_json(health_path)
        last_updated = datetime.fromisoformat(str(health["updated_at"]))
        age = max(0.0, (datetime.now(UTC) - last_updated.astimezone(UTC)).total_seconds())
        target = health.get("target") or {}
    except (OSError, KeyError, TypeError, ValueError):
        return {
            "scenario_id": SQLI_SCENARIO_ID,
            "target_version": SQLI_TARGET_VERSION,
            "status": "unavailable",
            "reason": "invalid_executor_health",
        }
    available = age <= 15 and isinstance(target, dict) and target.get("status") == "available"
    return {
        "scenario_id": SQLI_SCENARIO_ID,
        "target_version": SQLI_TARGET_VERSION,
        "status": "available" if available else "unavailable",
        "last_health_check_age_seconds": round(age, 3),
        "executor": health.get("status"),
        "target": target,
    }


@router.post("/sqli/events", status_code=201)
async def record_sqli_event(submission: SqliEventSubmission, request: Request) -> dict[str, Any]:
    _require_trainer(request)
    try:
        _repository().record_lab_event(
            run_id=str(submission.run_id),
            episode_id=str(submission.episode_id),
            seed=submission.seed,
            event_id=str(submission.event_id),
            event_type=submission.event_type,
            payload=submission.payload,
            scenario_id=SQLI_SCENARIO_ID,
            target_version=SQLI_TARGET_VERSION,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return {"event_id": str(submission.event_id), "status": "persisted"}


@router.post("/sqli/jobs", status_code=202)
async def submit_sqli_job(submission: SqliJobSubmission, request: Request) -> dict[str, Any]:
    _require_trainer(request)
    try:
        composed_value = submission.action.compose_value()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    job = SqliJob(
        job_id=uuid4(),
        run_id=submission.run_id,
        episode_id=submission.episode_id,
        decision_id=submission.decision_id,
        step_id=submission.step_id,
        request_id=uuid4(),
        action=submission.action,
        purpose=submission.purpose,
    )
    SQLI_SPOOL.joinpath("jobs").mkdir(parents=True, exist_ok=True)
    write_json_atomic(SQLI_SPOOL / "jobs" / f"{job.job_id}.json", job.model_dump(mode="json"))
    _repository().record_lab_event(
        run_id=str(job.run_id),
        episode_id=str(job.episode_id),
        seed=0,
        event_id=str(job.job_id),
        event_type="sqli_target_response",
        payload={
            "phase": "request_submitted",
            "job_id": str(job.job_id),
            "decision_id": str(job.decision_id),
            "request_id": str(job.request_id),
            "step_id": job.step_id,
            "purpose": job.purpose,
            "target_id": "sqli_lab_v1",
            "route_id": "lookup",
            "components": submission.action.components(),
            "composed_query_value": composed_value,
            "action_schema_id": submission.action.schema_id,
        },
        scenario_id=SQLI_SCENARIO_ID,
        target_version=SQLI_TARGET_VERSION,
    )
    return {"job_id": str(job.job_id), "request_id": str(job.request_id), "status": "queued"}


@router.get("/sqli/jobs/{job_id}")
async def get_sqli_job(job_id: UUID, request: Request) -> dict[str, Any]:
    _require_trainer(request)
    result_path = SQLI_SPOOL / "results" / f"{job_id}.json"
    if not result_path.exists():
        if (SQLI_SPOOL / "jobs" / f"{job_id}.json").exists() or (
            SQLI_SPOOL / "processing" / f"{job_id}.json"
        ).exists():
            return {"job_id": str(job_id), "status": "queued"}
        return {"job_id": str(job_id), "status": "unknown"}
    try:
        result = SqliExecutorResult.model_validate(read_json(result_path))
    except Exception:
        raise HTTPException(status_code=503, detail="SQLi executor result is invalid") from None
    telemetry = _sqli_telemetry(result)
    return result.model_copy(update={"telemetry": telemetry}).model_dump(mode="json")


@router.post("/sqli/controls", status_code=202)
async def submit_sqli_control(submission: SqliControlRequest, request: Request) -> dict[str, Any]:
    _require_trainer(request)
    command = SqliControlCommand(
        command_id=uuid4(),
        run_id=submission.run_id,
        episode_id=submission.episode_id,
        decision_id=submission.decision_id,
        action=submission.action,
    )
    SQLI_CONTROL_INBOX.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        SQLI_CONTROL_INBOX / f"{command.command_id}.json", command.model_dump(mode="json")
    )
    _repository().record_lab_event(
        run_id=str(command.run_id),
        episode_id=str(command.episode_id),
        seed=0,
        event_id=str(command.command_id),
        event_type="sqli_blue_decision",
        payload={
            "phase": "control_submitted",
            "command_id": str(command.command_id),
            "decision_id": str(command.decision_id) if command.decision_id else None,
            "action": command.action,
            "action_schema_id": "sqli.blue_action_catalog.v1",
        },
        scenario_id=SQLI_SCENARIO_ID,
        target_version=SQLI_TARGET_VERSION,
    )
    return {"command_id": str(command.command_id), "status": "queued"}


@router.get("/sqli/controls/{command_id}")
async def get_sqli_control(command_id: UUID, request: Request) -> dict[str, Any]:
    _require_trainer(request)
    result_path = SQLI_CONTROL_RESULTS / f"{command_id}.json"
    if not result_path.exists():
        return {"command_id": str(command_id), "status": "queued"}
    try:
        return read_json(result_path)
    except (OSError, ValueError):
        raise HTTPException(
            status_code=503, detail="SQLi control acknowledgement is invalid"
        ) from None


@router.post("/training-runs/{training_run_id}/stop")
async def stop_training_run(training_run_id: UUID) -> dict[str, Any]:
    record = _repository().request_training_stop(str(training_run_id))
    if record is None:
        raise HTTPException(status_code=404, detail="training run not found")
    return record


@router.post("/training-runs/claim")
async def claim_training_run(request: Request) -> dict[str, Any]:
    _require_trainer(request)
    worker_id = request.headers.get("x-violet-worker-id", "")
    if not worker_id or len(worker_id) > 64:
        raise HTTPException(status_code=422, detail="valid worker ID is required")
    return {"job": _repository().claim_training_run(worker_id)}


@router.post("/training-runs/{training_run_id}/worker-update")
async def training_worker_update(
    training_run_id: UUID, update: WorkerUpdate, request: Request
) -> dict[str, Any]:
    _require_trainer(request)
    try:
        record = _repository().update_training_run(
            str(training_run_id), update.worker_id, update.status, update.result
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    if record is None:
        raise HTTPException(status_code=404, detail="training run not found")
    return record


@router.post("/training-runs/{training_run_id}/heartbeat")
async def training_worker_heartbeat(training_run_id: UUID, request: Request) -> dict[str, Any]:
    _require_trainer(request)
    worker_id = request.headers.get("x-violet-worker-id", "")
    if not worker_id or len(worker_id) > 64:
        raise HTTPException(status_code=422, detail="valid worker ID is required")
    try:
        record = _repository().heartbeat_training_run(str(training_run_id), worker_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    if record is None:
        raise HTTPException(status_code=404, detail="training run not found")
    return record


@router.post("/training-runs/{training_run_id}/updates", status_code=201)
async def record_training_update(
    training_run_id: UUID, update: TrainingUpdate, request: Request
) -> dict[str, Any]:
    _require_trainer(request)
    current = _repository().get_training_run(str(training_run_id))
    if current is None:
        raise HTTPException(status_code=404, detail="training run not found")
    if current["worker_id"] != update.worker_id or current["status"] != "running":
        raise HTTPException(status_code=409, detail="training run is not owned by this worker")
    if update.role is not None and current["role"] not in {"joint", update.role}:
        raise HTTPException(status_code=409, detail="update role does not match the training run")
    update_role = update.role or current["role"]
    try:
        index = int(update.result["update_index"])
        global_step = int(update.result["global_step"])
        metrics = update.result["metrics"]
        digest = update.result.get("checkpoint_sha256")
        if index < 1 or global_step < 0 or not isinstance(metrics, dict):
            raise ValueError("invalid training update values")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
            raise ValueError("checkpoint digest must be SHA-256")
        _repository().record_training_update(
            str(training_run_id), update_role, index, global_step, metrics, digest
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return {
        "training_run_id": str(training_run_id),
        "status": "update_recorded",
        "update_index": index,
    }


@router.post("/runs/{run_id}/episodes/{episode_id}/finish")
async def finish_episode(run_id: UUID, episode_id: UUID, request: EpisodeFinish) -> dict[str, Any]:
    try:
        _repository().finish_episode(str(run_id), str(episode_id), request.status, request.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return {"episode_id": str(episode_id), "status": request.status}


@router.get("/runs/{run_id}/observations/{role}", response_model=RangeObservation)
async def get_observation(
    run_id: UUID,
    role: str,
    episode_id: UUID,
    step_id: int = 0,
) -> RangeObservation:
    if role not in {"red", "blue"} or not 0 <= step_id <= 500:
        raise HTTPException(status_code=422, detail="invalid role or step")
    repository = _repository()
    if not repository.episode_exists(str(run_id), str(episode_id)):
        raise HTTPException(status_code=404, detail="range episode not found")
    episode_events = repository.episode_events(str(run_id), str(episode_id))
    submitted = sum(
        event["event_type"] in {"action_submitted", "evaluator_probe_submitted"}
        for event in episode_events
    )
    # Keep four requests available for independent defensive verification probes.
    remaining = max(0, 36 - submitted)
    current_target_state = await target_status()
    current_status = str(current_target_state["status"])
    if role == "blue":
        return build_blue_observation(
            run_id=run_id,
            episode_id=episode_id,
            step_id=step_id,
            events=episode_events,
            remaining_requests=remaining,
            target_status=current_status,
            active_controls=current_target_state["active_controls"],
        )
    return build_red_observation(
        run_id=run_id,
        episode_id=episode_id,
        step_id=step_id,
        events=episode_events,
        remaining_requests=remaining,
        target_status=current_status,
    )


@router.post("/decisions", status_code=201)
async def record_decision(submission: DecisionSubmission) -> dict[str, Any]:
    """Persist exact role observation and typed catalog decision before execution."""
    try:
        repository = _repository()
        existing = repository.get_decision(str(submission.decision.decision_id))
        if existing is None:
            canonical = await get_observation(
                submission.decision.run_id,
                submission.decision.role,
                submission.decision.episode_id,
                submission.decision.step_id,
            )
            if canonical.model_dump(mode="json") != submission.observation.model_dump(mode="json"):
                raise ValueError(
                    "decision observation is stale or differs from the API-built role view"
                )
        repository.record_decision(submission.decision, submission.observation)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except Exception:
        raise HTTPException(
            status_code=503, detail="agent decision could not be persisted"
        ) from None
    return {
        "decision_id": str(submission.decision.decision_id),
        "status": "decision_recorded",
        "next_action": "submit_job"
        if submission.decision.role == "red"
        else "apply_control_or_monitor",
    }


@router.post("/runs/{run_id}/episodes/{episode_id}/transitions", status_code=201)
async def record_transition(
    run_id: UUID, episode_id: UUID, request: TransitionSubmission
) -> dict[str, Any]:
    """Persist a complete sampled transition after its next role view is known."""
    repository = _repository()
    decision = repository.get_decision(str(request.decision_id))
    if (
        decision is None
        or decision["run_id"] != str(run_id)
        or decision["episode_id"] != str(episode_id)
        or decision["policy_sample"] is None
    ):
        raise HTTPException(status_code=409, detail="transition requires a sampled policy decision")
    sample = decision["policy_sample"]
    if sample.get("mode") != request.mode:
        raise HTTPException(
            status_code=409, detail="transition mode differs from its policy sample"
        )
    next_observation = request.next_observation
    if (
        next_observation.run_id != run_id
        or next_observation.episode_id != episode_id
        or next_observation.schema_id != decision["observation_schema_id"]
        or next_observation.step_id != decision["step_id"] + 1
    ):
        raise HTTPException(status_code=409, detail="next observation does not follow the decision")
    try:
        canonical = await get_observation(
            run_id, decision["role"], episode_id, next_observation.step_id
        )
    except HTTPException:
        raise
    if canonical.model_dump(mode="json") != next_observation.model_dump(mode="json"):
        raise HTTPException(status_code=409, detail="next observation is stale or non-canonical")
    reward = repository.get_reward(
        str(run_id), str(episode_id), int(decision["step_id"]), str(decision["role"])
    )
    if reward is None or reward.get("decision_id") != str(request.decision_id):
        raise HTTPException(status_code=409, detail="transition requires its persisted reward")

    events = repository.episode_events(str(run_id), str(episode_id))
    operation = str(decision["selected_action"].get("operation", ""))
    execution_evidence: dict[str, Any] = {}
    if decision["role"] == "red":
        if operation == "WAIT":
            execution_status = "not_executed"
            accepted = True
        else:
            result_event = next(
                (
                    event
                    for event in events
                    if event["event_type"] == "executor_result"
                    and event["payload"].get("decision_id") == str(request.decision_id)
                ),
                None,
            )
            if result_event is None:
                raise HTTPException(status_code=409, detail="transition executor result is missing")
            outcome = result_event["payload"].get("result", {})
            status = str(outcome.get("status", "failed"))
            execution_status = status
            accepted = status == "completed"
            execution_evidence = {
                "executor_result_event_id": result_event["event_id"],
                "job_id": result_event["payload"].get("job_id"),
                "action_id": result_event["payload"].get("action_id"),
                "request_id": result_event["payload"].get("request_id"),
                "target_event_id": outcome.get("target_event_id"),
                "http_status": outcome.get("http_status"),
            }
    elif operation == "MONITOR":
        execution_status, accepted = "not_executed", True
    else:
        submitted_control_event = next(
            (
                event
                for event in events
                if event["event_type"] == "target_control_submitted"
                and (event["payload"].get("command") or {}).get("decision_id")
                == str(request.decision_id)
            ),
            None,
        )
        command_id = (
            submitted_control_event["payload"].get("command_id")
            if submitted_control_event
            else None
        )
        acknowledgement_event = next(
            (
                event
                for event in events
                if event["event_type"] == "target_control_acknowledged"
                and event["payload"].get("command_id") == command_id
            ),
            None,
        )
        if acknowledgement_event is None:
            raise HTTPException(
                status_code=409, detail="transition control acknowledgement is missing"
            )
        acknowledgement = acknowledgement_event["payload"]
        execution_status = str(acknowledgement.get("status", "failed"))
        accepted = execution_status == "applied"
        execution_evidence = {
            "control_submitted_event_id": submitted_control_event["event_id"]
            if submitted_control_event
            else None,
            "control_acknowledgement_event_id": acknowledgement_event["event_id"],
            "command_id": command_id,
            "state_version": acknowledgement.get("state_version"),
        }

    payload = {
        "schema_id": "violet.agent_transition.v2",
        "decision_id": str(request.decision_id),
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "step_id": decision["step_id"],
        "role": decision["role"],
        "observation_schema_id": decision["observation_schema_id"],
        "feature_encoding_schema_id": sample["encoding_schema_id"],
        "observation": decision["observation"],
        "encoded_observation": sample["encoded_observation"],
        "action_mask": decision["legal_action_mask"],
        "action_distribution": sample["action_distribution"],
        "selected_action_index": sample["selected_action_index"],
        "selected_action": decision["selected_action"],
        "selected_action_probability": sample["selected_action_probability"],
        "old_log_probability": sample["old_log_probability"],
        "value_estimate": sample["value_estimate"],
        "policy_version": sample["policy_version"],
        "entropy": sample["entropy"],
        "execution_status": execution_status,
        "execution_evidence": execution_evidence,
        "accepted": accepted,
        "reward": reward,
        "next_observation": next_observation.model_dump(mode="json"),
        "next_value": request.next_value,
        "terminated": request.terminated,
        "truncated": request.truncated,
        "mode": request.mode,
    }
    try:
        repository.record_transition(
            str(run_id), str(episode_id), str(request.decision_id), payload
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return payload


@router.post("/runs/{run_id}/episodes/{episode_id}/detections")
async def evaluate_detections(run_id: UUID, episode_id: UUID) -> dict[str, Any]:
    """Run the versioned Blue rules on sanitized persisted target telemetry."""
    repository = _repository()
    events = repository.episode_events(str(run_id), str(episode_id))
    alerts = detect(events)
    saved = 0
    for alert in alerts:
        try:
            saved += int(repository.record_detection(str(run_id), str(episode_id), alert))
        except ValueError:
            raise HTTPException(
                status_code=409, detail="detection evidence is not persisted"
            ) from None
    return {"schema_id": "violet.detection.v1", "alerts": alerts, "persisted_count": saved}


@router.post("/runs/{run_id}/episodes/{episode_id}/rewards", status_code=201)
async def record_reward(
    run_id: UUID, episode_id: UUID, request: RewardSubmission
) -> dict[str, Any]:
    """Derive rewards from persisted verifier/detection/control evidence, never client labels."""
    repository = _repository()
    events = repository.episode_events(str(run_id), str(episode_id))
    if request.role == "red":
        if request.decision_id is not None:
            decision = repository.get_decision(str(request.decision_id))
            if (
                decision is None
                or decision["role"] != "red"
                or decision["run_id"] != str(run_id)
                or decision["episode_id"] != str(episode_id)
                or decision["step_id"] != request.step_id
                or decision["selected_action"].get("operation") != "WAIT"
            ):
                raise HTTPException(
                    status_code=409, detail="no matching persisted Red WAIT decision"
                )
            reward = red_reward(None)
            job_record = None
        else:
            job_record = repository.get_job(str(request.job_id)) if request.job_id else None
        if job_record is None or job_record.get("result") is None:
            if request.decision_id is None:
                raise HTTPException(
                    status_code=409, detail="Red reward requires a completed persisted job"
                )
        else:
            job = ExecutorJob.model_validate(job_record["payload"])
            if str(job.run_id) != str(run_id) or str(job.episode_id) != str(episode_id):
                raise HTTPException(status_code=409, detail="job belongs to another run or episode")
            result_event = next(
                (
                    event
                    for event in events
                    if event["event_type"] == "executor_result"
                    and event["payload"].get("job_id") == str(job.job_id)
                ),
                None,
            )
            finding = result_event["payload"].get("finding") if result_event else None
            reward = red_reward(finding)
            reward["job_id"] = str(job.job_id)
            reward["decision_id"] = str(job.decision_id)
        if request.decision_id is not None:
            reward["decision_id"] = str(request.decision_id)
    else:
        decision = repository.get_decision(str(request.decision_id))
        if (
            decision is None
            or decision["role"] != "blue"
            or decision["run_id"] != str(run_id)
            or decision["episode_id"] != str(episode_id)
            or decision["step_id"] != request.step_id
        ):
            raise HTTPException(
                status_code=409, detail="Blue reward requires its persisted decision"
            )
        alert = None
        evaluator_label = None
        if request.source_target_event_id:
            source_id = str(request.source_target_event_id)
            alert = next(
                (
                    event["payload"]
                    for event in events
                    if event["event_type"] == "detection_suspected"
                    and event["payload"].get("source_target_event_id") == source_id
                ),
                None,
            )
            if alert is None:
                raise HTTPException(
                    status_code=409, detail="Blue reward requires a persisted detection"
                )
            evaluator_label = any(
                event["event_type"] == "executor_result"
                and (event["payload"].get("finding") or {}).get("status") == "confirmed"
                and source_id in ((event["payload"].get("finding") or {}).get("evidence") or [])
                for event in events
            )
        control_verified = False
        denied_legitimate_request = False
        if request.control_command_id:
            verification = next(
                (
                    event["payload"]
                    for event in events
                    if event["event_type"] == "control_verification"
                    and event["payload"].get("command_id") == str(request.control_command_id)
                ),
                None,
            )
            control_verified = bool(verification and verification.get("status") == "verified")
            checks = verification.get("checks", {}) if isinstance(verification, dict) else {}
            denied_legitimate_request = bool(
                checks.get("alice_owner_read_served") is False
                or checks.get("bob_owner_read_served") is False
            )
        reward = blue_reward(
            detection=alert,
            evaluator_label=evaluator_label,
            control_verified=control_verified,
            denied_legitimate_request=denied_legitimate_request,
        )
        reward["decision_id"] = str(request.decision_id)
        if request.source_target_event_id:
            reward["source_target_event_id"] = str(request.source_target_event_id)
        if request.control_command_id:
            reward["control_command_id"] = str(request.control_command_id)
    try:
        repository.record_reward(str(run_id), str(episode_id), request.step_id, reward)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return reward


@router.post("/reset", status_code=202)
async def reset_target(request: ResetRequest) -> dict[str, str]:
    run_id, episode_id = request.run_id or uuid4(), request.episode_id or uuid4()
    command = ControlCommand(
        command_id=request.command_id or uuid4(),
        command="RESET",
        seed=request.seed,
        run_id=run_id,
        episode_id=episode_id,
    )
    try:
        _repository().submit_control(command)
        _enqueue_control(command)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
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
    if request.purpose == "policy" and request.decision_id is None:
        raise HTTPException(
            status_code=409, detail="policy controls require a persisted Blue decision"
        )
    if request.decision_id is not None:
        repository = _repository()
        decision = repository.get_decision(str(request.decision_id))
        try:
            selected_control = (
                BlueAction.model_validate(decision["selected_action"]).model_dump(mode="json")
                if decision
                else None
            )
        except Exception:
            selected_control = None
        if (
            decision is None
            or decision["role"] != "blue"
            or selected_control
            != {"operation": request.action, "principal_ref": request.principal_ref}
            or (request.run_id is not None and decision["run_id"] != str(request.run_id))
            or (
                request.episode_id is not None and decision["episode_id"] != str(request.episode_id)
            )
        ):
            raise HTTPException(
                status_code=409, detail="control must match a persisted Blue decision"
            )
    run_id, episode_id = request.run_id or uuid4(), request.episode_id or uuid4()
    command = ControlCommand(
        command_id=request.command_id or uuid4(),
        command="APPLY_CONTROL",
        action=request.action,
        principal_ref=request.principal_ref,
        run_id=run_id,
        episode_id=episode_id,
        decision_id=request.decision_id,
        purpose=request.purpose,
    )
    try:
        _repository().submit_control(command)
        _enqueue_control(command)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
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
            return ControlAcknowledgement.model_validate(persisted["acknowledgement"]).model_dump(
                mode="json"
            )
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
    alice_owner_record = repository.get_job(str(request.alice_owner_job_id))
    owner_record = repository.get_job(str(request.owner_job_id))
    if unauthorized_record is None or alice_owner_record is None or owner_record is None:
        raise HTTPException(status_code=404, detail="verification job not found")
    if any(
        record.get("result") is None
        for record in (unauthorized_record, alice_owner_record, owner_record)
    ):
        raise HTTPException(status_code=409, detail="verification jobs are not complete")
    unauthorized_job = ExecutorJob.model_validate(unauthorized_record["payload"])
    alice_owner_job = ExecutorJob.model_validate(alice_owner_record["payload"])
    owner_job = ExecutorJob.model_validate(owner_record["payload"])
    unauthorized_result = ExecutorResult.model_validate(unauthorized_record["result"])
    alice_owner_result = ExecutorResult.model_validate(alice_owner_record["result"])
    owner_result = ExecutorResult.model_validate(owner_record["result"])
    if not all(
        repository.has_target_evidence(
            str(result.target_event_id) if result.target_event_id else None,
            str(result.request_id),
            str(job.run_id),
            str(job.episode_id),
            str(job.action_id),
        )
        for job, result in (
            (unauthorized_job, unauthorized_result),
            (alice_owner_job, alice_owner_result),
            (owner_job, owner_result),
        )
    ):
        raise HTTPException(
            status_code=409, detail="verification target evidence is missing or mismatched"
        )
    ack_seq = repository.event_seq(
        str(command.run_id), "target_control_acknowledged", "command_id", str(command_id)
    )
    alice_seq = repository.event_seq(
        str(command.run_id),
        submission_event_type(unauthorized_job.purpose),
        "job_id",
        str(unauthorized_job.job_id),
    )
    bob_seq = repository.event_seq(
        str(command.run_id),
        submission_event_type(owner_job.purpose),
        "job_id",
        str(owner_job.job_id),
    )
    alice_owner_seq = repository.event_seq(
        str(command.run_id),
        submission_event_type(alice_owner_job.purpose),
        "job_id",
        str(alice_owner_job.job_id),
    )
    if ack_seq is None or alice_seq is None or bob_seq is None or alice_owner_seq is None:
        raise HTTPException(status_code=409, detail="control or request events are not persisted")
    if alice_seq <= ack_seq or alice_owner_seq <= ack_seq or bob_seq <= ack_seq:
        raise HTTPException(status_code=409, detail="verification requests must follow the control")
    verification = verify_ownership_control(
        command,
        acknowledgement,
        unauthorized_job,
        unauthorized_result,
        alice_owner_job,
        alice_owner_result,
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
