"""Local real-range API backed by the isolated executor and typed control spool."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import random
import secrets
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

import httpx
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
    INTERACTION_PROTOCOL_SCHEMA as SQLI_INTERACTION_PROTOCOL_SCHEMA,
)
from violet.sqli_lab.contracts import (
    LEGACY_INTERACTION_PROTOCOL_SCHEMA as SQLI_LEGACY_PROTOCOL_SCHEMA,
)
from violet.sqli_lab.contracts import (
    REWARD_SCHEMA as SQLI_REWARD_SCHEMA,
)
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
SQLI_CHALLENGE_GATEWAY_URL = os.getenv(
    "SQLI_CHALLENGE_GATEWAY_URL", "http://sqli-challenge-gateway:8003"
).rstrip("/")
_REPOSITORY: EventRepository | None = None
SQLI_CHALLENGE_SESSIONS: dict[str, dict[str, Any]] = {}
SQLI_CHALLENGE_LOCK = asyncio.Lock()
DEMO_CAMPAIGN_LOCK = asyncio.Lock()
PILOT_LOCK = asyncio.Lock()
EXPERIMENT_LOCK = asyncio.Lock()
SQLI_CHALLENGE_MAX_SESSIONS = 16
SQLI_CHALLENGE_RETENTION_SECONDS = 24 * 60 * 60
SQLI_CHALLENGE_ACTIVE_TTL_SECONDS = 20 * 60
SQLI_EVIDENCE_CACHE_SIZE = 2048
SQLI_EVENT_MAX_BYTES = 48_000
SQLI_PPO_EVENT_MAX_BYTES = 128_000
_SQLI_EVIDENCE_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


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
    max_steps: int = Field(default=12, ge=1, le=40)
    seed: int = Field(default=17, ge=0, le=2**31 - 1)
    opponent: Literal["scripted", "random", "none"] = "scripted"
    learning_rate: float = Field(default=0.0003, gt=0, le=0.01)
    campaign_label: Literal[
        "live_demo_v1",
        "retained_baseline_v1",
        "classroom_evaluation_v2",
        "balanced_pilot_v1",
        "balanced_pilot_evaluation_v1",
        "learning_evaluation_v1",
        "exposure_experiment_v1",
        "exposure_experiment_evaluation_v1",
    ] | None = None
    checkpoint_variant: Literal["initial", "final"] = "final"
    learning_evaluation_id: UUID | None = None
    learning_evaluation_seeds: list[int] | None = None
    experiment_id: UUID | None = None
    experiment_branch: Literal["control", "increased"] | None = None
    vulnerable_configuration_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    experiment_episode_seeds: list[int] | None = None
    experiment_evaluation_matches: int | None = Field(default=None, ge=5, le=100)
    baseline_training_run_id: UUID | None = None
    baseline_checkpoint_run_id: UUID | None = None
    pilot_campaign_id: UUID | None = None
    pilot_started_at: str | None = None
    pilot_training_deadline: str | None = None
    pilot_evaluation_deadline: str | None = None
    pilot_training_episodes_per_seed: int | None = Field(default=None, ge=1, le=100)
    pilot_evaluation_seeds: list[int] | None = None

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
        if self.scenario_id != SQLI_SCENARIO_ID and self.max_steps > 12:
            raise ValueError("the Violet Ledger episode limit remains 12 decisions")
        if self.scenario_id != SQLI_SCENARIO_ID and self.opponent == "none":
            raise ValueError("the Violet Ledger scenario requires an explicitly labeled comparator")
        if self.campaign_label is not None and self.scenario_id != SQLI_SCENARIO_ID:
            raise ValueError("classroom campaign labels are only valid for the SQLi lab")
        legacy_40_round = self.campaign_label in {
            "live_demo_v1", "retained_baseline_v1", "classroom_evaluation_v2"
        }
        if legacy_40_round and self.max_steps != 40:
            raise ValueError("archived SQLi classroom campaigns retain their 40-round limit")
        if self.campaign_label in {"balanced_pilot_v1", "balanced_pilot_evaluation_v1"} and self.max_steps != 32:
            raise ValueError("new SQLi policy runs use a maximum of 32 rounds per episode")
        if self.campaign_label in {
            "learning_evaluation_v1", "exposure_experiment_v1", "exposure_experiment_evaluation_v1"
        } and self.max_steps > 32:
            raise ValueError("new SQLi policy runs use a maximum of 32 rounds per episode")
        if self.scenario_id == SQLI_SCENARIO_ID and not legacy_40_round and self.max_steps > 32:
            raise ValueError("new SQLi runs use a maximum of 32 rounds per episode")
        if self.campaign_label in {"live_demo_v1", "retained_baseline_v1", "classroom_evaluation_v2"} and self.episodes != 40:
            raise ValueError("legacy SQLi classroom campaigns use exactly 40 requested episodes")
        if self.campaign_label == "balanced_pilot_v1" and (
            self.mode != "train" or self.episodes != 50 or self.seed not in {17, 18, 19}
        ):
            raise ValueError("balanced pilot training uses 50 episodes on seeds 17, 18, and 19")
        if self.campaign_label == "balanced_pilot_evaluation_v1" and (
            self.mode != "evaluate" or self.episodes != 15 or self.pilot_campaign_id is None
        ):
            raise ValueError("balanced pilot evaluation uses 15 held-out matches per campaign seed")
        if self.campaign_label == "classroom_evaluation_v2" and self.mode != "evaluate":
            raise ValueError("the frozen classroom evaluation must use a checkpoint")
        if self.campaign_label == "retained_baseline_v1" and self.mode != "train":
            raise ValueError("the retained baseline continuation must train from its checkpoint")
        if self.campaign_label == "balanced_pilot_v1" and self.checkpoint_run_id is not None:
            raise ValueError("balanced pilot policies must start from fresh initialization")
        if self.campaign_label == "balanced_pilot_evaluation_v1" and (
            self.pilot_evaluation_seeds is None
            or len(self.pilot_evaluation_seeds) != self.episodes
            or len(set(self.pilot_evaluation_seeds)) != self.episodes
        ):
            raise ValueError("balanced pilot evaluation requires 15 unique held-out seeds")
        if self.campaign_label == "learning_evaluation_v1" and (
            self.mode != "evaluate"
            or self.checkpoint_run_id is None
            or self.learning_evaluation_id is None
            or self.learning_evaluation_seeds is None
            or len(self.learning_evaluation_seeds) != self.episodes
            or len(set(self.learning_evaluation_seeds)) != self.episodes
        ):
            raise ValueError("learning checkpoint evaluation requires a paired seed schedule")
        if self.campaign_label == "exposure_experiment_v1" and (
            self.mode != "train"
            or self.checkpoint_run_id is None
            or self.experiment_id is None
            or self.experiment_branch is None
            or self.vulnerable_configuration_probability is None
            or self.experiment_episode_seeds is None
            or len(self.experiment_episode_seeds) != self.episodes
            or len(set(self.experiment_episode_seeds)) != self.episodes
            or self.experiment_evaluation_matches is None
        ):
            raise ValueError("exposure experiment branches require equal seeded training budgets")
        if self.campaign_label == "exposure_experiment_evaluation_v1" and (
            self.mode != "evaluate"
            or self.checkpoint_run_id is None
            or self.experiment_id is None
            or self.experiment_branch is None
            or self.experiment_episode_seeds is None
            or len(self.experiment_episode_seeds) != self.episodes
        ):
            raise ValueError("exposure experiment evaluation requires the matched test schedule")
        if self.campaign_label in {"balanced_pilot_v1", "balanced_pilot_evaluation_v1"} and (
            self.pilot_campaign_id is None
        ):
            raise ValueError("balanced pilot runs require a campaign ID")
        if self.mode == "evaluate" and self.checkpoint_run_id is None:
            raise ValueError("evaluation requires a completed compatible checkpoint run ID")
        if self.run_kind == "research" and self.mode == "train" and self.episodes < 10000:
            raise ValueError("research training requires at least 10,000 episodes per seed")
        if self.run_kind == "research" and self.mode == "evaluate" and self.episodes < 100:
            raise ValueError("research evaluation requires at least 100 paired held-out episodes")
        return self


class PilotStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LearningReportStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training_run_id: UUID
    evaluation_id: UUID
    matches: int = Field(default=20, ge=5, le=100)


class ExposureExperimentStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_training_run_id: UUID
    experiment_id: UUID
    episodes_per_branch: int = Field(default=30, ge=5, le=500)
    seed: int = Field(default=1701, ge=0, le=2**31 - 1)
    control_vulnerability_probability: float = Field(default=0.35, ge=0.0, le=1.0)
    increased_vulnerability_probability: float = Field(default=0.85, ge=0.0, le=1.0)
    evaluation_matches: int = Field(default=20, ge=5, le=100)


class ExposureExperimentEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluation_matches: int = Field(default=20, ge=5, le=100)


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
        "sqli_episode_outcome",
    ]
    payload: dict[str, Any]

    @model_validator(mode="after")
    def payload_is_bounded(self) -> SqliEventSubmission:
        payload_bytes = len(
            json.dumps(self.payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        limit = (
            SQLI_PPO_EVENT_MAX_BYTES
            if self.event_type == "sqli_ppo_update"
            else SQLI_EVENT_MAX_BYTES
        )
        if payload_bytes > limit:
            limit_kb = limit // 1000
            raise ValueError(f"event payload exceeds the {limit_kb} KB lab event limit")
        return self


class SqliJobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: UUID
    episode_id: UUID
    decision_id: UUID
    step_id: int = Field(ge=0, le=40)
    action: SqliRequestAction
    purpose: Literal["policy", "verifier_attack", "verifier_legitimate"] = "policy"
    challenge_id: UUID | None = None


class SqliControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: UUID
    episode_id: UUID
    decision_id: UUID | None = None
    action: Literal["RESET", "PARAMETERIZE_QUERY", "RATE_LIMIT_GUEST", "ISOLATE_GUEST_LOOKUP"]
    vulnerable_configuration_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    fixture_seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class SqliChallengeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    secret_phrase: str = Field(min_length=1, max_length=96)

    @model_validator(mode="after")
    def phrase_is_not_blank(self) -> SqliChallengeRequest:
        if not self.secret_phrase.strip():
            raise ValueError("enter a non-empty disposable test phrase")
        return self


class SqliChallengeFinish(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["completed", "incomplete", "failed"]
    outcome: dict[str, Any]


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
    item = _find_sqli_target_event(path, str(result.target_event_id))
    if item is None:
        return {"evidence_status": "missing"}
    if (
        item.get("run_id") != str(result.run_id)
        or item.get("episode_id") != str(result.episode_id)
        or item.get("request_id") != str(result.request_id)
        or item.get("challenge_id")
        != (str(result.challenge_id) if result.challenge_id else None)
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
        "unauthorized_rows",
        "query_rejected",
        "challenge_id",
        "challenge_disclosed",
        "controls",
        "purpose",
        "trace_schema_id",
        "trace",
    )
    return {
        "evidence_status": "verified",
        **{key: item[key] for key in allowed if key in item},
    }


@router.get("/server/traces/{target_event_id}")
async def get_server_trace(target_event_id: UUID) -> dict[str, Any]:
    """Return one target-owned request trace from the read-only evidence spool."""
    path = SQLI_TARGET_EVIDENCE
    if path.is_dir() or not path.suffix:
        path = path / "events.jsonl"
    evidence = _find_sqli_target_event(path, str(target_event_id))
    if evidence is None:
        raise HTTPException(status_code=404, detail="server trace evidence was not found")
    trace = evidence.get("trace")
    if not isinstance(trace, list):
        raise HTTPException(status_code=404, detail="server trace stages were not recorded")
    return {
        "schema_id": evidence.get("trace_schema_id", "violet.server_trace.v1"),
        "target_event_id": evidence.get("target_event_id"),
        "run_id": evidence.get("run_id"),
        "episode_id": evidence.get("episode_id"),
        "decision_id": evidence.get("decision_id"),
        "request_id": evidence.get("request_id"),
        "purpose": evidence.get("purpose", "unspecified"),
        "route_id": evidence.get("route_id"),
        "method": evidence.get("method"),
        "http_status": evidence.get("http_status"),
        "occurred_at": evidence.get("occurred_at"),
        "duration_ms": evidence.get("duration_ms"),
        "trace": trace,
        "evidence": {
            "evidence_status": "target_recorded",
            "target_version": SQLI_TARGET_VERSION,
        },
    }


def _find_sqli_target_event(path: Path, target_event_id: str) -> dict[str, Any] | None:
    cache_key = f"{path}:{target_event_id}"
    cached = _SQLI_EVIDENCE_CACHE.get(cache_key)
    if cached is not None:
        _SQLI_EVIDENCE_CACHE.move_to_end(cache_key)
        return cached
    try:
        with path.open("rb") as handle:
            position = handle.seek(0, os.SEEK_END)
            remaining_tail_bytes = 4 * 1024 * 1024
            partial = b""
            while position > 0 and remaining_tail_bytes > 0:
                chunk_start = max(0, position - 64 * 1024)
                handle.seek(chunk_start)
                data = handle.read(position - chunk_start) + partial
                lines = data.split(b"\n")
                if chunk_start > 0:
                    partial = lines[0]
                    lines = lines[1:]
                else:
                    partial = b""
                for line in reversed(lines):
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if isinstance(item, dict) and item.get("target_event_id") == target_event_id:
                        _SQLI_EVIDENCE_CACHE[cache_key] = item
                        _SQLI_EVIDENCE_CACHE.move_to_end(cache_key)
                        while len(_SQLI_EVIDENCE_CACHE) > SQLI_EVIDENCE_CACHE_SIZE:
                            _SQLI_EVIDENCE_CACHE.popitem(last=False)
                        return item
                remaining_tail_bytes -= position - chunk_start
                position = chunk_start
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
async def get_run_episodes(
    run_id: UUID,
    after_index: int | None = None,
    before_index: int | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    if (
        (after_index is not None and after_index < -1)
        or (before_index is not None and before_index < 0)
        or (after_index is not None and before_index is not None)
        or not 1 <= limit <= 500
    ):
        raise HTTPException(status_code=422, detail="invalid episode cursor or limit")
    page = _repository().list_episodes(
        str(run_id),
        limit + 1,
        after_index=after_index,
        before_index=before_index,
    )
    items = page[:limit] if after_index is not None else page[-limit:]
    has_more = len(page) > limit
    older = has_more if after_index is None else True
    newer = has_more if after_index is not None else before_index is not None
    return {
        "run_id": str(run_id),
        "after_index": after_index,
        "before_index": before_index,
        "next_after_index": items[-1]["episode_index"] if items else after_index,
        "next_before_index": items[0]["episode_index"] if items else before_index,
        "has_more": has_more,
        "has_older": older,
        "has_newer": newer,
        "items": items,
    }


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
    page = repository.episode_events_page(
        str(run_id), str(episode_id), after_seq=after_seq, limit=limit + 1
    )
    items = page[:limit]
    return {
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "after_seq": after_seq,
        "next_after_seq": items[-1]["seq"] if items else after_seq,
        "has_more": len(page) > limit,
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
    page = repository.lab_summary_events(str(run_id), after_seq, limit + 1)
    items = page[:limit]
    return {
        "run_id": str(run_id),
        "after_seq": after_seq,
        "next_after_seq": items[-1]["seq"] if items else after_seq,
        "has_more": len(page) > limit,
        "items": items,
    }


def _checkpoint_record_metadata(
    record: dict[str, Any], *, allow_running: bool = False
) -> dict[str, Any] | None:
    """Check persisted checkpoint identity and policy-version metadata.

    The trainer verifies artifact bytes before loading them. This API-side
    check makes sure the run has a digest and its policy versions agree with
    the latest accepted Red and Blue update records.
    """
    config = record.get("config") or {}
    result = record.get("result") or {}
    status = record.get("status")
    statuses = {"completed", "stopped", "running"} if allow_running else {"completed", "stopped"}
    if status not in statuses:
        return None
    if (
        record.get("role") != "joint"
        or config.get("scenario_id") != SQLI_SCENARIO_ID
        or config.get("target_version") != SQLI_TARGET_VERSION
        or config.get("reward_schema_id") not in {"sqli.reward.v1", "sqli.reward.v2", SQLI_REWARD_SCHEMA}
    ):
        return None

    updates = _repository().training_updates(str(record.get("training_run_id")))
    by_role = {role: [item for item in updates if item.get("role") == role] for role in ("red", "blue")}
    if not by_role["red"] or not by_role["blue"]:
        return None

    saved_versions = result.get("policy_versions")
    if not isinstance(saved_versions, dict):
        saved_versions = {}
    metadata = result.get("checkpoint_metadata")
    protocol_schema_id = config.get(
        "interaction_protocol_schema_id",
        metadata.get("interaction_protocol_schema_id", SQLI_LEGACY_PROTOCOL_SCHEMA)
        if isinstance(metadata, dict)
        else SQLI_LEGACY_PROTOCOL_SCHEMA,
    )
    ppo_provenance = result.get("ppo_settings")
    if isinstance(metadata, dict):
        if (
            metadata.get("schema_id") not in {"sqli.checkpoint_metadata.v1", "sqli.checkpoint_metadata.v2", "sqli.checkpoint_metadata.v3"}
            or metadata.get("training_run_id") != record.get("training_run_id")
            or metadata.get("reward_schema_id") != config.get("reward_schema_id")
            or metadata.get("target_version") != SQLI_TARGET_VERSION
            or metadata.get("optimization_passes") != 4
            or (
                metadata.get("schema_id") in {"sqli.checkpoint_metadata.v2", "sqli.checkpoint_metadata.v3"}
                and metadata.get("interaction_protocol_schema_id") != protocol_schema_id
            )
            or (
                metadata.get("schema_id") == "sqli.checkpoint_metadata.v3"
                and (
                    metadata.get("model_architecture") != "factorized_gru_actor_critic.v1"
                    or ppo_provenance is None
                    or ppo_provenance.get("actor_critic") != "factorized_gru_actor_critic"
                )
            )
        ):
            return None
        metadata_provenance = metadata.get("ppo_settings")
        if metadata_provenance is not None:
            if (
                not isinstance(metadata_provenance, dict)
                or metadata_provenance.get("optimization_passes") != 4
                or (ppo_provenance is not None and metadata_provenance != ppo_provenance)
            ):
                return None
            ppo_provenance = metadata_provenance
        recorded_updates = result.get("completed_training_episodes")
        metadata_updates = metadata.get("completed_training_updates")
        if (
            not isinstance(metadata_updates, int)
            or metadata_updates < 0
            or (isinstance(recorded_updates, int) and recorded_updates != metadata_updates)
            or metadata.get("campaign_revision") != config.get("campaign_revision")
        ):
            return None
        saved_versions = metadata.get("policy_versions") or saved_versions

    policy_versions: dict[str, int] = {}
    latest_digests: dict[str, str] = {}
    for role in ("red", "blue"):
        latest = max(
            by_role[role],
            key=lambda item: (item.get("update_index", 0), item.get("created_at", "")),
        )
        metrics = latest.get("metrics") or {}
        version = metrics.get("policy_version_after")
        if not isinstance(version, (int, float)) or int(version) < 1:
            return None
        if saved_versions and int(saved_versions.get(role, -1)) != int(version):
            return None
        policy_versions[role] = int(version)
        digest = latest.get("checkpoint_sha256")
        if isinstance(digest, str) and len(digest) == 64:
            latest_digests[role] = digest

    if saved_versions and any(role not in saved_versions for role in ("red", "blue")):
        return None
    digest = result.get("checkpoint_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        if not allow_running:
            return None
        if len(latest_digests) != 2 or latest_digests["red"] != latest_digests["blue"]:
            return None
        digest = latest_digests["red"]
    elif status in {"running", "completed"} and (
        latest_digests.get("red") != digest or latest_digests.get("blue") != digest
    ):
        return None
    if isinstance(metadata, dict) and metadata.get("checkpoint_sha256") not in {None, digest}:
        return None

    completed_updates = result.get("completed_training_episodes")
    if isinstance(metadata, dict):
        completed_updates = metadata.get("completed_training_updates", completed_updates)
    if not isinstance(completed_updates, int):
        completed_updates = min(
            len({item.get("update_index") for item in by_role[role] if item.get("update_index")})
            for role in ("red", "blue")
        )
    if completed_updates < 1:
        return None
    if isinstance(metadata, dict) and completed_updates != metadata.get("completed_training_updates"):
        return None
    return {
        "training_run_id": record["training_run_id"],
        "status": status,
        "checkpoint_sha256": digest,
        "completed_training_episodes": completed_updates,
        "policy_versions": policy_versions,
        "reward_schema_id": config["reward_schema_id"],
        "interaction_protocol_schema_id": protocol_schema_id,
        "target_version": SQLI_TARGET_VERSION,
        "checkpoint_metadata": metadata,
        "ppo_settings": ppo_provenance if isinstance(ppo_provenance, dict) else None,
    }


@router.post("/training-runs", status_code=202)
async def create_training_run(request: TrainingRunRequest) -> dict[str, Any]:
    checkpoint_metadata: dict[str, Any] | None = None
    source_checkpoint_digest: str | None = None
    source_checkpoint_versions: dict[str, int] | None = None
    if request.mode == "evaluate" and request.checkpoint_run_id is not None:
        source = _repository().get_training_run(str(request.checkpoint_run_id))
        if source is None or source.get("role") != request.role:
            raise HTTPException(
                status_code=422,
                detail="evaluation checkpoint must belong to a compatible run for the same role",
            )
        source_result = source.get("result") or {} if source else {}
        source_config = source.get("config") or {} if source else {}
        if source_config.get("scenario_id", "violet_ledger_v1") != request.scenario_id:
            raise HTTPException(
                status_code=422,
                detail="evaluation checkpoint belongs to an incompatible target scenario",
            )
        if request.scenario_id == SQLI_SCENARIO_ID:
            checkpoint_metadata = _checkpoint_record_metadata(source)
            if checkpoint_metadata is None:
                raise HTTPException(
                    status_code=422,
                    detail="evaluation checkpoint digest and policy metadata could not be verified",
                )
            if checkpoint_metadata["interaction_protocol_schema_id"] != SQLI_INTERACTION_PROTOCOL_SCHEMA:
                raise HTTPException(
                    status_code=422,
                    detail="evaluation requires a checkpoint trained with the current Blue-control-first protocol",
                )
            if request.checkpoint_variant == "initial":
                if request.campaign_label != "learning_evaluation_v1":
                    raise HTTPException(status_code=422, detail="initial checkpoint evaluation is reserved for learning reports")
                initial_digest = source_result.get("initial_checkpoint_sha256")
                initial_versions = source_result.get("initial_policy_versions")
                if (
                    not isinstance(initial_digest, str)
                    or len(initial_digest) != 64
                    or not isinstance(initial_versions, dict)
                    or any(not isinstance(initial_versions.get(role), int) for role in ("red", "blue"))
                ):
                    raise HTTPException(status_code=422, detail="the source run has no verifiable initial checkpoint")
                source_checkpoint_digest = initial_digest
                source_checkpoint_versions = {
                    role: int(initial_versions[role]) for role in ("red", "blue")
                }
            else:
                source_checkpoint_digest = checkpoint_metadata["checkpoint_sha256"]
                source_checkpoint_versions = checkpoint_metadata["policy_versions"]
        elif (
            source.get("status") != "completed"
            or not (source_result.get("checkpoint_sha256") or "")
        ):
            raise HTTPException(
                status_code=422,
                detail="evaluation checkpoint must come from a completed training run",
            )
        if request.run_kind == "research" and (
            source_config.get("run_kind") != "research"
            or int(source_result.get("completed_training_episodes", 0)) < 10000
        ):
            raise HTTPException(
                status_code=422,
                detail="research evaluation requires a research checkpoint with 10,000 completed training episodes",
            )
        if request.campaign_label == "classroom_evaluation_v2" and (
            source_config.get("campaign_revision") != "red_syntax_progress_ppo4_v2"
            or checkpoint_metadata is None
            or checkpoint_metadata["reward_schema_id"] != "sqli.reward.v2"
            or checkpoint_metadata["completed_training_episodes"] < 40
        ):
            raise HTTPException(
                status_code=422,
                detail="classroom evaluation requires the verified 40-update revised checkpoint",
            )
        if request.campaign_label == "balanced_pilot_evaluation_v1" and (
            source_config.get("campaign_label") != "balanced_pilot_v1"
            or source_config.get("pilot_campaign_id") != str(request.pilot_campaign_id)
            or checkpoint_metadata is None
            or checkpoint_metadata["reward_schema_id"] != SQLI_REWARD_SCHEMA
            or checkpoint_metadata["completed_training_episodes"] < 1
        ):
            raise HTTPException(
                status_code=422,
                detail="pilot evaluation requires a verified fresh checkpoint from the same campaign",
            )
    elif request.mode == "train" and request.scenario_id == SQLI_SCENARIO_ID and request.checkpoint_run_id:
        source = _repository().get_training_run(str(request.checkpoint_run_id))
        if source is None:
            raise HTTPException(status_code=422, detail="training continuation checkpoint was not found")
        checkpoint_metadata = _checkpoint_record_metadata(source)
        if checkpoint_metadata is None:
            raise HTTPException(
                status_code=422,
                detail="training continuation checkpoint digest and policy metadata could not be verified",
            )
        source_checkpoint_digest = checkpoint_metadata["checkpoint_sha256"]
        source_checkpoint_versions = checkpoint_metadata["policy_versions"]
        if (source.get("config") or {}).get("scenario_id") != SQLI_SCENARIO_ID:
            raise HTTPException(status_code=422, detail="training checkpoint belongs to an incompatible target")
        if checkpoint_metadata["interaction_protocol_schema_id"] != SQLI_INTERACTION_PROTOCOL_SCHEMA:
            raise HTTPException(status_code=422, detail="training cannot continue a Red-first checkpoint")
        if request.campaign_label == "retained_baseline_v1":
            if checkpoint_metadata["reward_schema_id"] != "sqli.reward.v1":
                raise HTTPException(status_code=422, detail="retained baseline continuation requires the v1 reward checkpoint")
        elif checkpoint_metadata["reward_schema_id"] not in {
            "sqli.reward.v1", "sqli.reward.v2", SQLI_REWARD_SCHEMA
        }:
            raise HTTPException(status_code=422, detail="training checkpoint reward contract is incompatible")
    training_run_id = str(request.training_run_id or uuid4())
    sqli = request.scenario_id == SQLI_SCENARIO_ID
    campaign_revision = {
        "live_demo_v1": "red_syntax_progress_ppo4_v2",
        "retained_baseline_v1": "retained_baseline_reward_v1_ppo4",
        "classroom_evaluation_v2": "frozen_evaluation_40_v1",
        "balanced_pilot_v1": "balanced_adversarial_gru_ppo4_v4",
        "balanced_pilot_evaluation_v1": "balanced_adversarial_eval_v1",
        "learning_evaluation_v1": "frozen_checkpoint_comparison_gru_v1",
        "exposure_experiment_v1": "exposure_adaptation_gru_v1",
        "exposure_experiment_evaluation_v1": "exposure_held_out_gru_v1",
    }.get(request.campaign_label or "")
    reward_schema_id = (
        "sqli.reward.v1"
        if sqli and request.campaign_label == "retained_baseline_v1"
        else "sqli.reward.v2"
        if sqli and request.campaign_label in {"live_demo_v1", "classroom_evaluation_v2"}
        else SQLI_REWARD_SCHEMA
        if sqli
        else "violet.range.reward.v2"
    )
    config = {
        "episodes": request.episodes,
        "run_kind": request.run_kind,
        "scenario_id": request.scenario_id,
        "max_steps": request.max_steps,
        "seed": request.seed,
        "opponent": "none" if sqli else request.opponent,
        "learning_rate": request.learning_rate,
        "campaign_label": request.campaign_label,
        "campaign_revision": campaign_revision,
        "interaction_protocol_schema_id": SQLI_INTERACTION_PROTOCOL_SCHEMA if sqli else None,
        "pilot_campaign_id": str(request.pilot_campaign_id) if request.pilot_campaign_id else None,
        "pilot_started_at": request.pilot_started_at,
        "pilot_training_deadline": request.pilot_training_deadline,
        "pilot_evaluation_deadline": request.pilot_evaluation_deadline,
        "pilot_training_episodes_per_seed": request.pilot_training_episodes_per_seed,
        "pilot_training_seed_schedule": (
            [(request.seed + index) % (2**31 - 1) for index in range(request.episodes)]
            if request.campaign_label == "balanced_pilot_v1"
            else None
        ),
        "pilot_evaluation_seeds": request.pilot_evaluation_seeds,
        "checkpoint_variant": request.checkpoint_variant,
        "learning_evaluation_id": str(request.learning_evaluation_id) if request.learning_evaluation_id else None,
        "learning_evaluation_seeds": request.learning_evaluation_seeds,
        "experiment_id": str(request.experiment_id) if request.experiment_id else None,
        "experiment_branch": request.experiment_branch,
        "vulnerable_configuration_probability": request.vulnerable_configuration_probability,
        "experiment_episode_seeds": request.experiment_episode_seeds,
        "experiment_evaluation_matches": request.experiment_evaluation_matches,
        "checkpoint_run_id": str(request.checkpoint_run_id) if request.checkpoint_run_id else None,
        "baseline_training_run_id": (
            str(request.baseline_training_run_id) if request.baseline_training_run_id else None
        ),
        "baseline_checkpoint_run_id": (
            str(request.baseline_checkpoint_run_id) if request.baseline_checkpoint_run_id else None
        ),
        "source_checkpoint_sha256": source_checkpoint_digest,
        "source_policy_versions": source_checkpoint_versions,
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
        "reward_schema_id": reward_schema_id,
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


@router.post("/sqli/demo-campaign", status_code=410)
async def start_demo_campaign(restart: bool = False) -> dict[str, Any]:
    """Retire the page-load demo in favor of an explicitly started pilot."""
    del restart
    raise HTTPException(
        status_code=410,
        detail="The single-seed demo is retired. Start an explicit three-seed pilot at /sqli/pilots.",
    )


def _is_sqli_campaign(record: dict[str, Any], label: str) -> bool:
    config = record.get("config") or {}
    return config.get("campaign_label") == label and config.get("scenario_id") == SQLI_SCENARIO_ID


def _retained_campaign_baseline(all_runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Resolve the original v1 checkpoint even after continuation restarts."""
    by_id = {str(item.get("training_run_id")): item for item in all_runs}
    for item in all_runs:
        config = item.get("config") or {}
        linked_id = config.get("baseline_checkpoint_run_id") or config.get("baseline_training_run_id")
        linked = by_id.get(str(linked_id)) if linked_id else None
        if (
            linked
            and (linked.get("config") or {}).get("reward_schema_id") == "sqli.reward.v1"
            and _checkpoint_record_metadata(linked) is not None
        ):
            return linked
    return next(
        (
            item
            for item in all_runs
            if item.get("status") in {"stopped", "completed"}
            and (item.get("config") or {}).get("scenario_id") == SQLI_SCENARIO_ID
            and (item.get("config") or {}).get("reward_schema_id") == "sqli.reward.v1"
            and _checkpoint_record_metadata(item) is not None
        ),
        None,
    )


def _latest_available_sqli_checkpoint() -> dict[str, Any] | None:
    for item in _all_training_runs():
        config = item.get("config") or {}
        if (
            config.get("campaign_label") != "balanced_pilot_v1"
            or config.get("campaign_revision") not in {"balanced_adversarial_ppo4_v3", "balanced_adversarial_gru_ppo4_v4"}
            or config.get("scenario_id") != SQLI_SCENARIO_ID
            or config.get("episodes") != 50
            or config.get("interaction_protocol_schema_id") != SQLI_INTERACTION_PROTOCOL_SCHEMA
            or item.get("mode") != "train"
        ):
            continue
        checkpoint = _checkpoint_record_metadata(item, allow_running=True)
        if (
            checkpoint is None
            or checkpoint["reward_schema_id"] != SQLI_REWARD_SCHEMA
            or checkpoint["interaction_protocol_schema_id"] != SQLI_INTERACTION_PROTOCOL_SCHEMA
            or checkpoint["completed_training_episodes"] < 50
        ):
            continue
        updates = _repository().training_updates(str(item["training_run_id"]))
        role_indices = {
            role: {
                int(update["update_index"])
                for update in updates
                if update.get("role") == role and isinstance(update.get("update_index"), int)
            }
            for role in ("red", "blue")
        }
        if any(not set(range(1, 51)).issubset(role_indices[role]) for role in ("red", "blue")):
            continue
        final_rows = {
            role: next(
                update
                for update in reversed(updates)
                if update.get("role") == role and update.get("update_index") == 50
            )
            for role in ("red", "blue")
        }
        if any(
            final_rows[role].get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
            or int((final_rows[role].get("metrics") or {}).get("optimizer_steps", 0)) != 4
            for role in ("red", "blue")
        ):
            continue
        return {**item, "checkpoint_metadata": checkpoint}
    return None


def _latest_completed_sqli_checkpoint() -> dict[str, Any] | None:
    """Compatibility alias for callers that used the old completed-run name."""
    return _latest_available_sqli_checkpoint()


@router.get("/sqli/checkpoint")
async def latest_sqli_checkpoint() -> dict[str, Any] | None:
    """Return compact checkpoint metadata without the campaign's episode array."""
    checkpoint = _latest_available_sqli_checkpoint()
    if checkpoint is None:
        return None
    return {
        "training_run_id": checkpoint["training_run_id"],
        "status": checkpoint["status"],
        "created_at": checkpoint.get("created_at"),
        "config": checkpoint.get("config") or {},
        "result": {
            "completed_training_episodes": (checkpoint.get("checkpoint_metadata") or {}).get("completed_training_episodes"),
            "valid_ppo_updated_episodes": (checkpoint.get("checkpoint_metadata") or {}).get("completed_training_episodes"),
            "checkpoint_sha256": (checkpoint.get("checkpoint_metadata") or {}).get("checkpoint_sha256"),
            "policy_versions": (checkpoint.get("checkpoint_metadata") or {}).get("policy_versions"),
            "checkpoint_metadata": (checkpoint.get("checkpoint_metadata") or {}).get("checkpoint_metadata"),
        },
    }


def _challenge_public(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "challenge_id": session["challenge_id"],
        "run_id": session["run_id"],
        "episode_id": session["episode_id"],
        "training_run_id": session["training_run_id"],
        "status": session["status"],
        "created_at": session["created_at"],
        "result": session.get("result"),
    }


@router.post("/sqli/challenges", status_code=202)
async def create_sqli_challenge(challenge: SqliChallengeRequest) -> dict[str, Any]:
    checkpoint = _latest_completed_sqli_checkpoint()
    if checkpoint is None:
        raise HTTPException(
            status_code=409,
            detail="complete 50 verified balanced-pilot updates before starting a challenge",
        )
    health_path = SQLI_SPOOL / "health.json"
    if not health_path.exists():
        raise HTTPException(status_code=503, detail="the isolated SQLi target is unavailable")
    try:
        health_record = read_json(health_path)
        health_updated = datetime.fromisoformat(str(health_record["updated_at"]))
        target = health_record.get("target") or {}
        target_available = (
            (datetime.now(UTC) - health_updated.astimezone(UTC)).total_seconds() <= 15
            and isinstance(target, dict)
            and target.get("status") == "available"
        )
    except (OSError, KeyError, TypeError, ValueError):
        target_available = False
    if not target_available:
        raise HTTPException(status_code=503, detail="the isolated SQLi target is unavailable")
    challenge_id = uuid4()
    session = {
        "challenge_id": str(challenge_id),
        "run_id": str(challenge_id),
        "episode_id": str(uuid4()),
        "training_run_id": checkpoint["training_run_id"],
        "checkpoint_sha256": checkpoint["checkpoint_metadata"]["checkpoint_sha256"],
        "checkpoint_policy_versions": checkpoint["checkpoint_metadata"]["policy_versions"],
        "checkpoint_metadata": checkpoint["checkpoint_metadata"].get("checkpoint_metadata"),
        "seed": secrets.randbelow(2**31 - 100_000),
        "max_steps": 32,
        "status": "preparing",
        "created_at": datetime.now(UTC).isoformat(),
        "result": None,
    }
    expired_ids: list[str] = []
    active_session_exists = False
    async with SQLI_CHALLENGE_LOCK:
        now = datetime.now(UTC)
        for old_id, old_session in list(SQLI_CHALLENGE_SESSIONS.items()):
            try:
                created = datetime.fromisoformat(str(old_session["created_at"]))
                age_seconds = (now - created.astimezone(UTC)).total_seconds()
            except (KeyError, TypeError, ValueError):
                age_seconds = SQLI_CHALLENGE_RETENTION_SECONDS + 1
            if (
                old_session.get("status") in {"queued", "preparing", "running"}
                and age_seconds > SQLI_CHALLENGE_ACTIVE_TTL_SECONDS
            ):
                old_session["status"] = "incomplete"
                old_session["result"] = {"reason": "challenge_session_expired"}
                expired_ids.append(old_id)
            if (
                age_seconds > SQLI_CHALLENGE_RETENTION_SECONDS
                and old_session.get("status") not in {"queued", "preparing", "running"}
            ):
                SQLI_CHALLENGE_SESSIONS.pop(old_id, None)
        active_session_exists = any(
            item.get("status") in {"queued", "preparing", "running"}
            for item in SQLI_CHALLENGE_SESSIONS.values()
        )
        if not active_session_exists and len(SQLI_CHALLENGE_SESSIONS) >= SQLI_CHALLENGE_MAX_SESSIONS:
            oldest_id = next(iter(SQLI_CHALLENGE_SESSIONS))
            SQLI_CHALLENGE_SESSIONS.pop(oldest_id, None)
        if not active_session_exists:
            SQLI_CHALLENGE_SESSIONS[session["challenge_id"]] = session
    for expired_id in expired_ids:
        await _clear_target_challenge(expired_id)
    if active_session_exists:
        raise HTTPException(status_code=409, detail="a challenge session is already active")
    try:
        async with httpx.AsyncClient(base_url=SQLI_CHALLENGE_GATEWAY_URL, timeout=5.0) as client:
            response = await client.post(
                f"/challenge/{challenge_id}", json={"phrase": challenge.secret_phrase}
            )
            response.raise_for_status()
            acknowledgement = response.json()
    except (httpx.HTTPError, ValueError):
        acknowledgement = None
    if not isinstance(acknowledgement, dict) or acknowledgement.get("status") != "ready":
        async with SQLI_CHALLENGE_LOCK:
            session["status"] = "failed"
            session["result"] = {"reason": "target_challenge_setup_failed"}
        await _clear_target_challenge(str(challenge_id))
        raise HTTPException(status_code=503, detail="the isolated target could not prepare this challenge")
    async with SQLI_CHALLENGE_LOCK:
        session["status"] = "queued"
    return {**_challenge_public(session), "max_steps": 32}


@router.get("/sqli/challenges/{challenge_id}")
async def get_sqli_challenge(challenge_id: UUID) -> dict[str, Any]:
    expired = False
    async with SQLI_CHALLENGE_LOCK:
        session = SQLI_CHALLENGE_SESSIONS.get(str(challenge_id))
        if session is not None and _challenge_session_expired(session):
            session["status"] = "incomplete"
            session["result"] = {"reason": "challenge_session_expired"}
            expired = True
    if session is not None:
        if expired:
            await _clear_target_challenge(str(challenge_id))
        return _challenge_public(session)
    run = _repository().get_run(str(challenge_id))
    if run is None or run.get("target_version") != SQLI_TARGET_VERSION:
        raise HTTPException(status_code=404, detail="challenge session not found")
    events = _repository().lab_summary_events(str(challenge_id), limit=50000)
    outcome = next(
        (
            item["payload"]
            for item in reversed(events)
            if item["event_type"] == "sqli_episode_outcome"
            and item.get("payload", {}).get("phase") == "challenge"
        ),
        None,
    )
    if outcome is not None:
        status = outcome.get("status")
        if status not in {"completed", "incomplete", "failed"}:
            status = "incomplete"
        return {
            "challenge_id": str(challenge_id),
            "run_id": str(challenge_id),
            "episode_id": outcome.get("episode_id"),
            "training_run_id": outcome.get("checkpoint_run_id"),
            "status": status,
            "created_at": None,
            "result": outcome,
        }
    return {
        "challenge_id": str(challenge_id),
        "run_id": str(challenge_id),
        "episode_id": None,
        "training_run_id": None,
        "status": "running",
        "created_at": None,
        "result": None,
    }


@router.post("/sqli/challenges/claim")
async def claim_sqli_challenge(request: Request) -> dict[str, Any]:
    _require_trainer(request)
    async with SQLI_CHALLENGE_LOCK:
        session = next(
            (
                item
                for item in SQLI_CHALLENGE_SESSIONS.values()
                if item["status"] == "queued"
            ),
            None,
        )
        if session is None:
            return {"job": None}
        session["status"] = "running"
        return {
            "job": {
                "challenge_id": session["challenge_id"],
                "run_id": session["run_id"],
                "episode_id": session["episode_id"],
                "training_run_id": session["training_run_id"],
                "checkpoint_sha256": session["checkpoint_sha256"],
                "checkpoint_policy_versions": session["checkpoint_policy_versions"],
                "checkpoint_metadata": session["checkpoint_metadata"],
                "seed": session["seed"],
                "max_steps": session["max_steps"],
            }
        }


@router.post("/sqli/challenges/{challenge_id}/finish")
async def finish_sqli_challenge(
    challenge_id: UUID, submission: SqliChallengeFinish, request: Request
) -> dict[str, Any]:
    _require_trainer(request)
    async with SQLI_CHALLENGE_LOCK:
        session = SQLI_CHALLENGE_SESSIONS.get(str(challenge_id))
        if session is None:
            raise HTTPException(status_code=404, detail="challenge session not found")
        safe_outcome = {
            "schema_id": "sqli.episode_outcome.v2",
            **submission.outcome,
            "phase": "challenge",
            "episode_id": str(session["episode_id"]),
            "checkpoint_run_id": str(session["training_run_id"]),
        }
        existing = _repository().lab_summary_events(str(challenge_id), limit=50000)
        has_outcome = any(
            item["event_type"] == "sqli_episode_outcome"
            and item.get("payload", {}).get("phase") == "challenge"
            for item in existing
        )
        if not has_outcome:
            try:
                _repository().record_lab_event(
                    run_id=str(challenge_id),
                    episode_id=str(session["episode_id"]),
                    seed=int(session["seed"]),
                    event_id=str(uuid4()),
                    event_type="sqli_episode_outcome",
                    payload=safe_outcome,
                    scenario_id=SQLI_SCENARIO_ID,
                    target_version=SQLI_TARGET_VERSION,
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from None
        session["status"] = submission.status
        session["result"] = safe_outcome
    cleared = await _clear_target_challenge(str(challenge_id))
    session["cleanup_status"] = "cleared" if cleared else "target_ttl_cleanup_pending"
    return _challenge_public(session)


def _challenge_session_expired(session: dict[str, Any]) -> bool:
    if session.get("status") not in {"queued", "preparing", "running"}:
        return False
    try:
        created = datetime.fromisoformat(str(session["created_at"]))
    except (KeyError, TypeError, ValueError):
        return True
    age = (datetime.now(UTC) - created.astimezone(UTC)).total_seconds()
    return age > SQLI_CHALLENGE_ACTIVE_TTL_SECONDS


async def _clear_target_challenge(challenge_id: str) -> bool:
    try:
        async with httpx.AsyncClient(base_url=SQLI_CHALLENGE_GATEWAY_URL, timeout=5.0) as client:
            response = await client.delete(f"/challenge/{challenge_id}")
            response.raise_for_status()
            return True
    except httpx.HTTPError:
        return False


def _all_training_runs() -> list[dict[str, Any]]:
    repository = _repository()
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = repository.list_training_runs(100, offset)
        records.extend(page)
        if len(page) < 100:
            return records
        offset += len(page)


def _pilot_evaluation_seed_schedule(training_seed: int, count: int = 15) -> list[int]:
    """Create a campaign-unique seed block outside the training seed range."""
    first = 1_000_000_000 + training_seed * 1_000
    return [first + index for index in range(count)]


def _pilot_wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * ((rate * (1 - rate) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _pilot_records(campaign_id: UUID) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    campaign_key = str(campaign_id)
    records = [
        item
        for item in _all_training_runs()
        if (item.get("config") or {}).get("pilot_campaign_id") == campaign_key
    ]
    training = [
        item
        for item in records
        if (item.get("config") or {}).get("campaign_label") == "balanced_pilot_v1"
    ]
    evaluation = [
        item
        for item in records
        if (item.get("config") or {}).get("campaign_label") == "balanced_pilot_evaluation_v1"
    ]
    return sorted(training, key=lambda item: int((item.get("config") or {}).get("seed", 0))), sorted(
        evaluation, key=lambda item: int((item.get("config") or {}).get("seed", 0))
    )


def _pilot_status(campaign_id: UUID) -> dict[str, Any]:
    training, evaluation = _pilot_records(campaign_id)
    if not training:
        raise HTTPException(status_code=404, detail="pilot campaign not found")

    active_states = {"queued", "running"}
    training_active = any(item.get("status") in active_states for item in training)
    evaluation_active = any(item.get("status") in active_states for item in evaluation)
    terminal = {"completed", "stopped", "failed"}
    training_terminal = len(training) == 3 and all(item.get("status") in terminal for item in training)
    checkpointable = [
        item for item in training if _checkpoint_record_metadata(item) is not None
    ]
    if training_active or len(training) < 3:
        status = "training"
    elif evaluation_active:
        status = "evaluation"
    elif evaluation and all(item.get("status") in terminal for item in evaluation):
        status = "completed" if len(evaluation) == 3 and all(
            item.get("status") == "completed" for item in evaluation
        ) else "incomplete"
    elif training_terminal and not evaluation:
        status = "ready_for_evaluation" if checkpointable else "incomplete"
    else:
        status = "incomplete"

    training_rows = []
    for item in training:
        config = item.get("config") or {}
        result = item.get("result") or {}
        checkpoint = _checkpoint_record_metadata(item)
        training_rows.append(
            {
                "seed": config.get("seed"),
                "training_run_id": item.get("training_run_id"),
                "status": item.get("status"),
                "stop_requested": bool(item.get("stop_requested")),
                "requested_episodes": int(config.get("episodes", 50)),
                "completed_episodes": int(result.get("completed_training_episodes", 0)),
                "attempted_episodes": int(result.get("episode_summary_count", 0)),
                "policy_versions": result.get("policy_versions"),
                "checkpoint_sha256": (checkpoint or {}).get("checkpoint_sha256"),
                "checkpoint_available": checkpoint is not None,
            }
        )

    eval_by_seed = {
        int((item.get("config") or {}).get("seed", -1)): item for item in evaluation
    }
    per_seed: list[dict[str, Any]] = []
    pooled_completed = pooled_red = pooled_blue = pooled_disclosures = 0
    prevention_weighted = availability_weighted = exposure_weighted = 0.0
    for item in training:
        config = item.get("config") or {}
        seed = int(config.get("seed", 0))
        eval_run = eval_by_seed.get(seed)
        eval_result = ((eval_run or {}).get("result") or {}).get("evaluation") or {}
        completed = int(eval_result.get("completed_matches", 0))
        red_wins = int(eval_result.get("red_wins", 0))
        blue_wins = int(eval_result.get("blue_wins", 0))
        disclosures = int(eval_result.get("verified_disclosures", 0))
        pooled_completed += completed
        pooled_red += red_wins
        pooled_blue += blue_wins
        pooled_disclosures += disclosures
        for source, accumulator in (
            ("verified_prevention_rate", "prevention"),
            ("legitimate_service_availability", "availability"),
            ("verified_exposure_rate", "exposure"),
        ):
            value = eval_result.get(source)
            if isinstance(value, (int, float)):
                if accumulator == "prevention":
                    prevention_weighted += float(value) * completed
                elif accumulator == "availability":
                    availability_weighted += float(value) * completed
                else:
                    exposure_weighted += float(value) * completed
        per_seed.append(
            {
                "training_seed": seed,
                "training_run_id": item.get("training_run_id"),
                "training_status": item.get("status"),
                "completed_training_episodes": int(
                    (item.get("result") or {}).get("completed_training_episodes", 0)
                ),
                "evaluation_run_id": (eval_run or {}).get("training_run_id"),
                "evaluation_status": (eval_run or {}).get("status", "not_started"),
                "evaluation_seeds": (eval_run or {}).get("config", {}).get("pilot_evaluation_seeds", []),
                "requested_matches": int((eval_run or {}).get("config", {}).get("episodes", 15)),
                "completed_matches": completed,
                "incomplete_matches": max(0, int((eval_run or {}).get("config", {}).get("episodes", 15)) - completed),
                "red_wins": red_wins,
                "blue_wins": blue_wins,
                "undecided_matches": int(eval_result.get("undecided_matches", 0)),
                "red_win_rate": red_wins / completed if completed else None,
                "blue_win_rate": blue_wins / completed if completed else None,
                "red_win_rate_wilson_95": _pilot_wilson(red_wins, completed),
                "blue_win_rate_wilson_95": _pilot_wilson(blue_wins, completed),
                "verified_disclosures": disclosures,
                "verified_exposure_rate": eval_result.get("verified_exposure_rate"),
                "verified_prevention_rate": eval_result.get("verified_prevention_rate"),
                "legitimate_service_availability": eval_result.get("legitimate_service_availability"),
            }
        )

    requested_matches = 45
    red_rate = pooled_red / pooled_completed if pooled_completed else None
    blue_rate = pooled_blue / pooled_completed if pooled_completed else None
    complete_evaluation = pooled_completed == requested_matches
    within_balance = (
        red_rate is not None
        and blue_rate is not None
        and red_rate >= 0.2
        and blue_rate >= 0.2
        and red_rate <= 0.8
        and blue_rate <= 0.8
    )
    if complete_evaluation and within_balance:
        balance_verdict = "balanced"
    elif not complete_evaluation:
        balance_verdict = "incomplete"
    elif red_rate is not None and red_rate > 0.8:
        balance_verdict = "red_dominant"
    elif blue_rate is not None and blue_rate > 0.8:
        balance_verdict = "blue_dominant"
    else:
        balance_verdict = "unbalanced"
    start_config = (training[0].get("config") or {})
    return {
        "campaign_id": str(campaign_id),
        "status": status,
        "started_at": start_config.get("pilot_started_at"),
        "training_deadline": start_config.get("pilot_training_deadline"),
        "evaluation_deadline": start_config.get("pilot_evaluation_deadline"),
        "interaction_protocol_schema_id": SQLI_INTERACTION_PROTOCOL_SCHEMA,
        "reward_schema_id": SQLI_REWARD_SCHEMA,
        "training": {
            "requested_per_seed": 50,
            "planned_seeds": [17, 18, 19],
            "completed_updates": sum(item["completed_episodes"] for item in training_rows),
            "requested_updates": 150,
            "runs": training_rows,
        },
        "evaluation": {
            "requested_matches": requested_matches,
            "completed_matches": pooled_completed,
            "incomplete_matches": max(0, requested_matches - pooled_completed),
            "attempted_matches": sum(
                int(((item.get("result") or {}).get("evaluation") or {}).get("attempted_matches", 0))
                for item in evaluation
            ),
            "red_wins": pooled_red,
            "blue_wins": pooled_blue,
            "undecided_matches": max(0, pooled_completed - pooled_red - pooled_blue),
            "red_win_rate": red_rate,
            "blue_win_rate": blue_rate,
            "red_win_rate_wilson_95": _pilot_wilson(pooled_red, pooled_completed),
            "blue_win_rate_wilson_95": _pilot_wilson(pooled_blue, pooled_completed),
            "verified_disclosures": pooled_disclosures,
            "verified_exposure_rate": exposure_weighted / pooled_completed if pooled_completed else None,
            "verified_prevention_rate": prevention_weighted / pooled_completed if pooled_completed else None,
            "legitimate_service_availability": availability_weighted / pooled_completed if pooled_completed else None,
            "per_seed": per_seed,
            "balanced": complete_evaluation and within_balance,
            "balance_verdict": balance_verdict,
            "balance_rule": "each side wins at least 20% and neither side exceeds 80% of completed held-out matches",
            "uncertainty_method": "Wilson score 95% intervals for pooled and per-seed win rates",
        },
    }


def _sqli_target_available() -> bool:
    health_path = SQLI_SPOOL / "health.json"
    if not health_path.exists():
        return False
    try:
        health = read_json(health_path)
        updated = datetime.fromisoformat(str(health["updated_at"]))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        target = health.get("target") or {}
        return (
            (datetime.now(UTC) - updated.astimezone(UTC)).total_seconds() <= 15
            and isinstance(target, dict)
            and target.get("status") == "available"
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


@router.post("/sqli/pilots", status_code=202)
async def start_balanced_pilot(_request: PilotStartRequest) -> dict[str, Any]:
    """Start three fresh paired policies for the agreed 17/18/19 pilot."""
    async with PILOT_LOCK:
        active = [
            item
            for item in _all_training_runs()
            if (item.get("config") or {}).get("campaign_label") in {
                "balanced_pilot_v1",
                "balanced_pilot_evaluation_v1",
            }
            and item.get("status") in {"queued", "running"}
        ]
        if active:
            raise HTTPException(
                status_code=409,
                detail=f"pilot run {active[0].get('training_run_id')} is already active",
            )
        if not _sqli_target_available():
            raise HTTPException(status_code=503, detail="the isolated SQLi target is unavailable")

        now = datetime.now(UTC)
        campaign_id = uuid4()
        started_at = now.isoformat()
        training_deadline = (now + timedelta(hours=3)).isoformat()
        evaluation_deadline = (now + timedelta(hours=4)).isoformat()
        created: list[dict[str, Any]] = []
        try:
            for seed in (17, 18, 19):
                created.append(
                    await create_training_run(
                        TrainingRunRequest(
                            role="joint",
                            mode="train",
                            run_kind="teaching",
                            scenario_id="sqli_lab_v1",
                            episodes=50,
                            max_steps=32,
                            seed=seed,
                            opponent="none",
                            campaign_label="balanced_pilot_v1",
                            pilot_campaign_id=campaign_id,
                            pilot_started_at=started_at,
                            pilot_training_deadline=training_deadline,
                            pilot_evaluation_deadline=evaluation_deadline,
                            pilot_training_episodes_per_seed=50,
                        )
                    )
                )
        except Exception:
            for item in created:
                if item.get("status") in {"queued", "running"}:
                    _repository().request_training_stop(str(item["training_run_id"]))
            raise
        return _pilot_status(campaign_id)


@router.get("/sqli/pilots/{campaign_id}")
async def get_balanced_pilot(campaign_id: UUID) -> dict[str, Any]:
    return _pilot_status(campaign_id)


@router.post("/sqli/pilots/{campaign_id}/evaluation", status_code=202)
async def start_balanced_pilot_evaluation(campaign_id: UUID) -> dict[str, Any]:
    """Queue one frozen held-out run for each seed with a verified checkpoint."""
    async with PILOT_LOCK:
        training, existing_evaluations = _pilot_records(campaign_id)
        if not training:
            raise HTTPException(status_code=404, detail="pilot campaign not found")
        if any(item.get("status") in {"queued", "running"} for item in training):
            raise HTTPException(status_code=409, detail="pilot training has not reached an episode boundary")
        if not _sqli_target_available():
            raise HTTPException(status_code=503, detail="the isolated SQLi target is unavailable")

        deadline = (training[0].get("config") or {}).get("pilot_evaluation_deadline")
        created: list[dict[str, Any]] = []
        existing_seeds = {
            int((item.get("config") or {}).get("seed", -1)) for item in existing_evaluations
        }
        for source in training:
            config = source.get("config") or {}
            seed = int(config["seed"])
            if seed in existing_seeds:
                continue
            checkpoint = _checkpoint_record_metadata(source)
            if checkpoint is None:
                continue
            evaluation_seeds = _pilot_evaluation_seed_schedule(seed)
            created.append(
                await create_training_run(
                    TrainingRunRequest(
                        role="joint",
                        mode="evaluate",
                        run_kind="teaching",
                        scenario_id="sqli_lab_v1",
                        episodes=15,
                        max_steps=32,
                        seed=seed,
                        opponent="none",
                        checkpoint_run_id=UUID(str(source["training_run_id"])),
                        campaign_label="balanced_pilot_evaluation_v1",
                        pilot_campaign_id=campaign_id,
                        pilot_started_at=(training[0].get("config") or {}).get("pilot_started_at"),
                        pilot_evaluation_deadline=deadline,
                        pilot_training_episodes_per_seed=50,
                        pilot_evaluation_seeds=evaluation_seeds,
                    )
                )
            )
        if not created and not existing_evaluations:
            raise HTTPException(
                status_code=409,
                detail="no seed produced a verifiable checkpoint; the pilot cannot be evaluated",
            )
        return _pilot_status(campaign_id)


def _seed_schedule(namespace_id: UUID, count: int, *, lower: int = 1_500_000_000) -> list[int]:
    """Create deterministic, high-range held-out seeds independent of training seeds."""
    modulus = 2**31 - 1
    start = lower + namespace_id.int % max(1, modulus - lower - count)
    return [(start + index) % modulus for index in range(count)]


def _run_episode_summaries(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = (record.get("result") or {}).get("episodes")
    return [item for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []


def _metric_map(record: dict[str, Any], key: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for item in _run_episode_summaries(record):
        seed = item.get("seed")
        raw = item.get(key)
        if item.get("status") != "completed" or not isinstance(seed, int):
            continue
        if isinstance(raw, bool):
            values[seed] = float(raw)
        elif isinstance(raw, (int, float)):
            values[seed] = float(raw)
    return values


def _paired_difference(initial: dict[int, float], later: dict[int, float], seed: int) -> dict[str, Any]:
    paired = [later[key] - initial[key] for key in sorted(initial.keys() & later.keys())]
    if not paired:
        return {"matched_fixtures": 0, "mean_change": None, "bootstrap_ci_95": None}
    mean = sum(paired) / len(paired)
    if len(paired) < 2:
        interval = None
    else:
        rng = random.Random(seed)
        draws = sorted(
            sum(paired[rng.randrange(len(paired))] for _ in paired) / len(paired)
            for _ in range(2000)
        )
        interval = [draws[49], draws[1949]]
    return {"matched_fixtures": len(paired), "mean_change": mean, "bootstrap_ci_95": interval}


def _learning_report(evaluation_id: UUID) -> dict[str, Any]:
    evaluation_key = str(evaluation_id)
    records = [
        item
        for item in _all_training_runs()
        if (item.get("config") or {}).get("learning_evaluation_id") == evaluation_key
    ]
    if not records:
        raise HTTPException(status_code=404, detail="learning evaluation was not found")
    source_id = str((records[0].get("config") or {}).get("checkpoint_run_id") or "")
    by_variant = {
        str((item.get("config") or {}).get("checkpoint_variant")): item
        for item in records
    }
    initial = by_variant.get("initial")
    later = by_variant.get("final")
    report_status = (
        "running"
        if any(item.get("status") in {"queued", "running"} for item in records)
        else "completed"
        if initial and later and all(item.get("status") == "completed" for item in (initial, later))
        else "incomplete"
    )

    def summarize(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        result = record.get("result") or {}
        evaluation = result.get("evaluation") or {}
        return {
            "run_id": record.get("training_run_id"),
            "status": record.get("status"),
            "checkpoint_variant": (record.get("config") or {}).get("checkpoint_variant"),
            "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
            "policy_versions": evaluation.get("policy_versions"),
            "requested_matches": evaluation.get("requested_matches"),
            "completed_matches": evaluation.get("completed_matches"),
            "incomplete_matches": evaluation.get("incomplete_matches"),
            "verified_red_success_rate": evaluation.get("verified_exposure_rate"),
            "red_success_wilson_95": _pilot_wilson(
                int(evaluation.get("verified_disclosures", 0)),
                int(evaluation.get("completed_matches", 0)),
            ),
            "verified_blue_prevention_rate": evaluation.get("verified_prevention_rate"),
            "legitimate_service_availability": evaluation.get("legitimate_service_availability"),
            "evaluation_seeds": (record.get("config") or {}).get("learning_evaluation_seeds", []),
            "frozen": evaluation.get("pair_frozen") is True,
            "scripted_opponents": evaluation.get("scripted_opponents"),
        }

    changes: dict[str, Any] = {}
    if initial and later:
        metric_keys = {
            "verified_red_success_rate": "verified_exposure",
            "verified_blue_prevention_rate": "verified_prevention_rate",
            "legitimate_service_availability": "legitimate_query_availability",
        }
        for report_key, episode_key in metric_keys.items():
            changes[report_key] = _paired_difference(
                _metric_map(initial, episode_key),
                _metric_map(later, episode_key),
                evaluation_id.int ^ sum(report_key.encode("utf-8")),
            )
    config = (records[0].get("config") or {})
    initial_eval = ((initial or {}).get("result") or {}).get("evaluation") or {}
    later_eval = ((later or {}).get("result") or {}).get("evaluation") or {}
    return {
        "schema_id": "violet.learning_report.v1",
        "evaluation_id": evaluation_key,
        "status": report_status,
        "source_training_run_id": source_id,
        "matched_seed_schedule": config.get("learning_evaluation_seeds", []),
        "test_distribution": {
            "target_version": SQLI_TARGET_VERSION,
            "scenario_id": SQLI_SCENARIO_ID,
            "vulnerable_configuration_probability": 0.5,
            "description": "The same held-out, seeded server configurations are used for both frozen checkpoint pairs.",
        },
        "initial": summarize(initial),
        "later": summarize(later),
        "paired_changes": changes,
        "family_results": {
            "sql_injection": {
                "status": report_status,
                "initial": initial_eval,
                "later": later_eval,
            },
            "unauthorized_record_access": {"status": "not_evaluated_in_sqli_scenario"},
            "permission_or_workflow_abuse": {"status": "not_available_in_current_target"},
        },
        "claim_basis": "Matched frozen held-out outcome metrics only; training loss and action-probability changes are not treated as performance improvement.",
    }


@router.post("/learning-reports", status_code=202)
async def start_learning_report(request: LearningReportStartRequest) -> dict[str, Any]:
    async with EXPERIMENT_LOCK:
        existing = [
            item for item in _all_training_runs()
            if (item.get("config") or {}).get("learning_evaluation_id") == str(request.evaluation_id)
        ]
        if existing:
            return _learning_report(request.evaluation_id)
        source = _repository().get_training_run(str(request.training_run_id))
        if source is None or source.get("role") != "joint" or source.get("mode") != "train":
            raise HTTPException(status_code=404, detail="joint training run was not found")
        checkpoint = _checkpoint_record_metadata(source)
        result = source.get("result") or {}
        initial_digest = result.get("initial_checkpoint_sha256")
        initial_versions = result.get("initial_policy_versions")
        if (
            checkpoint is None
            or source.get("status") not in {"completed", "stopped"}
            or not isinstance(initial_digest, str)
            or len(initial_digest) != 64
            or not isinstance(initial_versions, dict)
        ):
            raise HTTPException(status_code=409, detail="source run needs verified initial and final checkpoints")
        if not _sqli_target_available():
            raise HTTPException(status_code=503, detail="the isolated SQLi target is unavailable")
        config = source.get("config") or {}
        training_seeds = {
            (int(config.get("seed", 0)) + index) % (2**31 - 1)
            for index in range(int(config.get("episodes", 0)))
        }
        namespace = request.evaluation_id
        base = _seed_schedule(namespace, request.matches)
        for offset in range(100):
            if not training_seeds.intersection(base):
                break
            namespace = UUID(int=(namespace.int + 1) % (2**128))
            base = _seed_schedule(namespace, request.matches)
        else:
            raise HTTPException(status_code=409, detail="could not construct a disjoint evaluation seed schedule")
        created: list[dict[str, Any]] = []
        for variant in ("initial", "final"):
            created.append(
                await create_training_run(
                    TrainingRunRequest(
                        role="joint",
                        mode="evaluate",
                        run_kind="teaching",
                        scenario_id="sqli_lab_v1",
                        episodes=request.matches,
                        max_steps=min(32, int(config.get("max_steps", 32))),
                        seed=base[0],
                        opponent="none",
                        checkpoint_run_id=request.training_run_id,
                        campaign_label="learning_evaluation_v1",
                        checkpoint_variant=variant,
                        learning_evaluation_id=request.evaluation_id,
                        learning_evaluation_seeds=base,
                        vulnerable_configuration_probability=0.5,
                    )
                )
            )
        return _learning_report(request.evaluation_id)


@router.get("/learning-reports/{evaluation_id}")
async def get_learning_report(evaluation_id: UUID) -> dict[str, Any]:
    return _learning_report(evaluation_id)


def _exposure_experiment_report(experiment_id: UUID) -> dict[str, Any]:
    rows = [
        item for item in _all_training_runs()
        if (item.get("config") or {}).get("experiment_id") == str(experiment_id)
    ]
    if not rows:
        raise HTTPException(status_code=404, detail="exposure experiment was not found")
    branches: dict[str, dict[str, Any] | None] = {"control": None, "increased": None}
    evaluations: dict[str, dict[str, Any] | None] = {"control": None, "increased": None}
    for item in rows:
        config = item.get("config") or {}
        branch = config.get("experiment_branch")
        if branch not in branches:
            continue
        target = evaluations if config.get("campaign_label") == "exposure_experiment_evaluation_v1" else branches
        target[str(branch)] = item
    training_active = any(item and item.get("status") in {"queued", "running"} for item in branches.values())
    evaluation_active = any(item and item.get("status") in {"queued", "running"} for item in evaluations.values())
    if training_active or any(item is None for item in branches.values()):
        status = "training"
    elif evaluation_active:
        status = "evaluation"
    elif all(evaluations.values()) and all(item and item.get("status") == "completed" for item in evaluations.values()):
        status = "completed"
    elif all(item and item.get("status") in {"completed", "stopped", "failed"} for item in branches.values()):
        status = "ready_for_evaluation" if not any(evaluations.values()) else "incomplete"
    else:
        status = "incomplete"

    def training_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        config = record.get("config") or {}
        result = record.get("result") or {}
        summaries = _run_episode_summaries(record)
        configured = [
            bool((item.get("initial_server_configuration") or {}).get("vulnerable_configuration"))
            for item in summaries
            if isinstance(item.get("initial_server_configuration"), dict)
        ]
        completed = [item for item in summaries if item.get("status") == "completed"]
        def mean(key: str) -> float | None:
            found = [float(item[key]) for item in completed if isinstance(item.get(key), (int, float, bool))]
            return sum(found) / len(found) if found else None
        return {
            "run_id": record.get("training_run_id"),
            "status": record.get("status"),
            "requested_updates": config.get("episodes"),
            "completed_updates": result.get("completed_training_episodes", 0),
            "checkpoint_sha256": result.get("checkpoint_sha256"),
            "policy_versions": result.get("policy_versions"),
            "configured_vulnerability_probability": config.get("vulnerable_configuration_probability"),
            "realized_vulnerable_configuration_frequency": sum(configured) / len(configured) if configured else None,
            "episodes_with_target_configuration": len(configured),
            "verified_red_success_rate": mean("verified_exposure"),
            "verified_blue_prevention_rate": mean("verified_prevention_rate"),
            "legitimate_service_availability": mean("legitimate_query_availability"),
        }

    def evaluation_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        result = record.get("result") or {}
        return {
            "run_id": record.get("training_run_id"),
            "status": record.get("status"),
            **(result.get("evaluation") or {}),
            "evaluation_seeds": (record.get("config") or {}).get("experiment_episode_seeds", []),
            "target_test_probability": (record.get("config") or {}).get("vulnerable_configuration_probability"),
        }

    first_config = (branches.get("control") or {}).get("config") or {}
    return {
        "schema_id": "violet.exposure_experiment.v1",
        "experiment_id": str(experiment_id),
        "status": status,
        "family": "sql_injection",
        "source_training_run_id": first_config.get("checkpoint_run_id"),
        "equal_training_budget": bool(
            branches["control"] and branches["increased"]
            and (branches["control"].get("config") or {}).get("episodes")
            == (branches["increased"].get("config") or {}).get("episodes")
        ),
        "matched_training_seeds": first_config.get("experiment_episode_seeds", []),
        "training": {key: training_summary(value) for key, value in branches.items()},
        "held_out_evaluation": {
            "test_distribution_unchanged": True,
            "same_seed_schedule": bool(
                evaluations["control"] and evaluations["increased"]
                and (evaluations["control"].get("config") or {}).get("experiment_episode_seeds")
                == (evaluations["increased"].get("config") or {}).get("experiment_episode_seeds")
            ),
            "branches": {key: evaluation_summary(value) for key, value in evaluations.items()},
        },
        "interpretation": "Both branches start from the same source checkpoint and use equal episode budgets. Only the training probability for the SQL injection vulnerable server configuration changes; agents continue choosing all actions. Frozen evaluation uses the same fixtures at the same 0.5 test probability.",
    }


@router.post("/experiments/exposure", status_code=202)
async def start_exposure_experiment(request: ExposureExperimentStartRequest) -> dict[str, Any]:
    if request.increased_vulnerability_probability <= request.control_vulnerability_probability:
        raise HTTPException(status_code=422, detail="increased exposure probability must exceed the control branch")
    async with EXPERIMENT_LOCK:
        existing = [
            item for item in _all_training_runs()
            if (item.get("config") or {}).get("experiment_id") == str(request.experiment_id)
        ]
        if existing:
            return _exposure_experiment_report(request.experiment_id)
        source = _repository().get_training_run(str(request.source_training_run_id))
        if source is None or source.get("role") != "joint" or source.get("mode") != "train":
            raise HTTPException(status_code=404, detail="joint source training run was not found")
        checkpoint = _checkpoint_record_metadata(source)
        if checkpoint is None or source.get("status") not in {"completed", "stopped"}:
            raise HTTPException(status_code=409, detail="experiment source needs a verified completed checkpoint")
        if not _sqli_target_available():
            raise HTTPException(status_code=503, detail="the isolated SQLi target is unavailable")
        config = source.get("config") or {}
        training_seeds = [(request.seed + index) % (2**31 - 1) for index in range(request.episodes_per_branch)]
        created: list[dict[str, Any]] = []
        for branch, probability in (
            ("control", request.control_vulnerability_probability),
            ("increased", request.increased_vulnerability_probability),
        ):
            created.append(
                await create_training_run(
                    TrainingRunRequest(
                        role="joint",
                        mode="train",
                        run_kind="teaching",
                        scenario_id="sqli_lab_v1",
                        episodes=request.episodes_per_branch,
                        max_steps=min(32, int(config.get("max_steps", 32))),
                        seed=request.seed,
                        opponent="none",
                        checkpoint_run_id=request.source_training_run_id,
                        campaign_label="exposure_experiment_v1",
                        experiment_id=request.experiment_id,
                        experiment_branch=branch,
                        vulnerable_configuration_probability=probability,
                        experiment_episode_seeds=training_seeds,
                        experiment_evaluation_matches=request.evaluation_matches,
                    )
                )
            )
        return _exposure_experiment_report(request.experiment_id)


@router.get("/experiments/{experiment_id}")
async def get_exposure_experiment(experiment_id: UUID) -> dict[str, Any]:
    return _exposure_experiment_report(experiment_id)


@router.post("/experiments/{experiment_id}/evaluation", status_code=202)
async def start_exposure_experiment_evaluation(
    experiment_id: UUID,
    request: ExposureExperimentEvaluationRequest,
) -> dict[str, Any]:
    async with EXPERIMENT_LOCK:
        all_runs = [
            item for item in _all_training_runs()
            if (item.get("config") or {}).get("experiment_id") == str(experiment_id)
        ]
        training = {
            str((item.get("config") or {}).get("experiment_branch")): item
            for item in all_runs
            if (item.get("config") or {}).get("campaign_label") == "exposure_experiment_v1"
        }
        existing_evaluation = [
            item for item in all_runs
            if (item.get("config") or {}).get("campaign_label") == "exposure_experiment_evaluation_v1"
        ]
        if len(training) != 2:
            raise HTTPException(status_code=404, detail="both exposure training branches were not found")
        if existing_evaluation:
            return _exposure_experiment_report(experiment_id)
        if any(item.get("status") not in {"completed", "stopped"} for item in training.values()):
            raise HTTPException(status_code=409, detail="both training branches must finish before held-out evaluation")
        updates = [int((item.get("result") or {}).get("completed_training_episodes", 0)) for item in training.values()]
        requested_updates = [int((item.get("config") or {}).get("episodes", 0)) for item in training.values()]
        if updates[0] != updates[1] or updates[0] != requested_updates[0] or requested_updates[0] != requested_updates[1]:
            raise HTTPException(status_code=409, detail="both branches must complete the same requested update budget")
        if not _sqli_target_available():
            raise HTTPException(status_code=503, detail="the isolated SQLi target is unavailable")
        seeds = _seed_schedule(experiment_id, request.evaluation_matches, lower=1_800_000_000)
        created: list[dict[str, Any]] = []
        for branch in ("control", "increased"):
            source = training[branch]
            source_config = source.get("config") or {}
            created.append(
                await create_training_run(
                    TrainingRunRequest(
                        role="joint",
                        mode="evaluate",
                        run_kind="teaching",
                        scenario_id="sqli_lab_v1",
                        episodes=request.evaluation_matches,
                        max_steps=min(32, int(source_config.get("max_steps", 32))),
                        seed=seeds[0],
                        opponent="none",
                        checkpoint_run_id=UUID(str(source["training_run_id"])),
                        campaign_label="exposure_experiment_evaluation_v1",
                        experiment_id=experiment_id,
                        experiment_branch=branch,
                        vulnerable_configuration_probability=0.5,
                        experiment_episode_seeds=seeds,
                        experiment_evaluation_matches=request.evaluation_matches,
                    )
                )
            )
        return _exposure_experiment_report(experiment_id)


def _decode_sqli_episode_cursor(cursor: str | None) -> tuple[datetime | None, str | None]:
    if cursor is None:
        return None, None
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        payload = json.loads(raw)
        created_at = datetime.fromisoformat(payload["created_at"])
        episode_id = str(UUID(payload["episode_id"]))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return created_at.astimezone(UTC), episode_id
    except (binascii.Error, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=422, detail="invalid SQLi episode cursor") from None


def _encode_sqli_episode_cursor(item: dict[str, Any]) -> str:
    value = json.dumps(
        {"created_at": item["created_at"], "episode_id": item["episode_id"]},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


@router.get("/sqli/episodes")
async def list_sqli_episodes(
    cursor: str | None = None,
    limit: int = 50,
    run_id: UUID | None = None,
    phase: str | None = None,
    status: str | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    """Browse SQLi episodes across every persisted run without target traffic."""
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    if phase is not None and phase not in {"training", "evaluation", "challenge", "other"}:
        raise HTTPException(status_code=422, detail="invalid SQLi episode phase")
    if status is not None and status not in {"running", "completed", "incomplete", "failed"}:
        raise HTTPException(status_code=422, detail="invalid SQLi episode status")
    if outcome is not None and outcome not in {"red", "blue", "tie", "no_winner", "unknown"}:
        raise HTTPException(status_code=422, detail="invalid SQLi episode outcome")

    cursor_created_at, cursor_episode_id = _decode_sqli_episode_cursor(cursor)
    page = _repository().list_sqli_episodes(
        target_version=SQLI_TARGET_VERSION,
        limit=limit + 1,
        cursor_created_at=cursor_created_at,
        cursor_episode_id=cursor_episode_id,
        run_id=str(run_id) if run_id else None,
        phase=phase,
        status=status,
        outcome=outcome,
    )
    has_more = len(page) > limit
    items = page[:limit]
    return {
        "cursor": cursor,
        "next_cursor": _encode_sqli_episode_cursor(items[-1]) if has_more and items else None,
        "has_more": has_more,
        "items": items,
    }


@router.get("/training-runs")
async def list_training_runs(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be non-negative")
    page = _repository().list_training_runs(limit + 1, offset)
    return {
        "items": page[:limit],
        "next_offset": offset + min(limit, len(page)),
        "has_more": len(page) > limit,
    }


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
        challenge_id=submission.challenge_id,
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
            "challenge_id": str(job.challenge_id) if job.challenge_id else None,
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
        vulnerable_configuration_probability=submission.vulnerable_configuration_probability,
        fixture_seed=submission.fixture_seed,
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
