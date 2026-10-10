"""Versioned, bounded contracts shared by the API and isolated executor."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias
from uuid import UUID  # noqa: TC003 - Pydantic needs the runtime UUID class for schema validation.

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier: TypeAlias = Annotated[
    str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
]
SessionHandle: TypeAlias = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{32,96}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DiscoverRouteAction(StrictModel):
    operation: Literal["DISCOVER_ROUTE"]
    route_candidate_id: Literal["home", "login", "document"]


class AuthTestAction(StrictModel):
    operation: Literal["AUTH_TEST"]
    credential_ref: Literal["synthetic_alice", "synthetic_bob", "synthetic_invalid"]


class CheckDocumentAccessAction(StrictModel):
    operation: Literal["CHECK_DOCUMENT_ACCESS"]
    document_ref: Literal["doc_alice", "doc_bob"]
    session_handle: SessionHandle | None = None


class CheckTargetHealthAction(StrictModel):
    operation: Literal["CHECK_TARGET_HEALTH"]


RangeAction: TypeAlias = Annotated[
    DiscoverRouteAction | AuthTestAction | CheckDocumentAccessAction | CheckTargetHealthAction,
    Field(discriminator="operation"),
]


class ExecutorJob(StrictModel):
    schema_id: Literal["violet.executor_job.v1"] = "violet.executor_job.v1"
    job_id: UUID
    run_id: UUID
    episode_id: UUID
    step_id: int = Field(ge=0, le=500)
    decision_id: UUID
    action_id: UUID
    request_id: UUID
    target_id: Literal["violet_ledger_v1"] = "violet_ledger_v1"
    idempotency_key: UUID
    deadline_seconds: float = Field(default=5.0, gt=0, le=5.0)
    action: RangeAction


class ExecutorJobRequest(StrictModel):
    action: RangeAction
    run_id: UUID | None = None
    episode_id: UUID | None = None
    step_id: int = Field(default=0, ge=0, le=500)
    decision_id: UUID | None = None


class ExecutorResult(StrictModel):
    schema_id: Literal["violet.executor_result.v1"] = "violet.executor_result.v1"
    job_id: UUID
    action_id: UUID
    request_id: UUID
    status: Literal["completed", "rejected", "failed", "outcome_unknown"]
    http_status: int | None = Field(default=None, ge=100, le=599)
    content_type: str | None = None
    response_excerpt: str = Field(default="", max_length=2048)
    target_event_id: UUID | None = None
    session_handle: SessionHandle | None = None
    principal_ref: Literal["alice", "bob"] | None = None
    error_code: (
        Literal[
            "unknown_session",
            "timeout",
            "connection_failed",
            "invalid_job",
            "executor_error",
            "request_budget_exhausted",
            "episode_deadline_exceeded",
        ]
        | None
    ) = None
    started_at: str
    finished_at: str
    duration_ms: float = Field(ge=0)


class ControlRequest(StrictModel):
    action: Literal["HARDEN_DOCUMENT_OWNERSHIP", "RATE_LIMIT_ACTOR", "BLOCK_ACTOR"]
    principal_ref: Literal["alice", "bob"] | None = None
    run_id: UUID | None = None
    episode_id: UUID | None = None

    @model_validator(mode="after")
    def validate_principal(self) -> ControlRequest:
        if self.action in {"RATE_LIMIT_ACTOR", "BLOCK_ACTOR"} and self.principal_ref is None:
            raise ValueError("this control requires a principal reference")
        if self.action == "HARDEN_DOCUMENT_OWNERSHIP" and self.principal_ref is not None:
            raise ValueError("ownership hardening does not accept a principal reference")
        return self


class ControlVerificationRequest(StrictModel):
    unauthorized_job_id: UUID
    owner_job_id: UUID


class ControlCommand(StrictModel):
    schema_id: Literal["violet.target_control.v1"] = "violet.target_control.v1"
    command_id: UUID
    command: Literal["APPLY_CONTROL", "RESET"]
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    action: Literal["HARDEN_DOCUMENT_OWNERSHIP", "RATE_LIMIT_ACTOR", "BLOCK_ACTOR"] | None = None
    principal_ref: Literal["alice", "bob"] | None = None
    run_id: UUID | None = None
    episode_id: UUID | None = None

    @model_validator(mode="after")
    def validate_command(self) -> ControlCommand:
        if self.command == "RESET":
            if self.seed is None or self.action is not None or self.principal_ref is not None:
                raise ValueError("reset requires a seed and cannot include a control action")
        else:
            if self.seed is not None or self.action is None:
                raise ValueError("control application requires an action and cannot include a seed")
            if self.action in {"RATE_LIMIT_ACTOR", "BLOCK_ACTOR"} and self.principal_ref is None:
                raise ValueError("this control requires a principal reference")
            if self.action == "HARDEN_DOCUMENT_OWNERSHIP" and self.principal_ref is not None:
                raise ValueError("ownership hardening does not accept a principal reference")
        return self


class ControlAcknowledgement(StrictModel):
    schema_id: Literal["violet.target_control_ack.v1"] = "violet.target_control_ack.v1"
    command_id: UUID
    run_id: UUID | None = None
    episode_id: UUID | None = None
    status: Literal["applied", "rejected", "failed"]
    target_version: Literal["violet_ledger_v1"] = "violet_ledger_v1"
    state_version: int = Field(ge=0)
    controls: dict[str, object]
    fixture_seed: int
    completed_at: str
    error_code: str | None = None
