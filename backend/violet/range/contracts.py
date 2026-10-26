"""Versioned, bounded contracts shared by the API and isolated executor."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias
from uuid import UUID  # noqa: TC003 - Pydantic needs the runtime UUID class for schema validation.

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier: TypeAlias = Annotated[
    str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RangeObservation(StrictModel):
    """Auditable role view supplied to a policy, with ground truth excluded."""

    schema_id: Literal["red.range_observation.v2", "blue.range_observation.v2"]
    run_id: UUID
    episode_id: UUID
    step_id: int = Field(ge=0, le=500)
    target_status: Literal["available", "unavailable", "unknown"] = "unknown"
    remaining_requests: int = Field(ge=0, le=40)
    discovered_routes: tuple[Literal["home", "login", "document"], ...] = ()
    recent_events: tuple[dict[str, object], ...] = Field(default=(), max_length=64)
    active_controls: dict[str, object] = Field(default_factory=dict)
    source_event_ids: tuple[UUID, ...] = Field(default=(), max_length=256)
    legal_action_mask: tuple[bool, ...] = Field(default=(), max_length=32)


class PolicySample(StrictModel):
    """Complete categorical-policy sample needed to replay a PPO decision."""

    encoding_schema_id: Literal["violet.range.features.v2"]
    encoded_observation: tuple[float, ...] = Field(min_length=1, max_length=256)
    action_distribution: tuple[float, ...] = Field(min_length=1, max_length=32)
    selected_action_index: int = Field(ge=0, le=31)
    selected_action_probability: float = Field(gt=0.0, le=1.0)
    old_log_probability: float = Field(ge=-100.0, le=0.0)
    value_estimate: float = Field(ge=-1e6, le=1e6)
    entropy: float = Field(ge=0.0, le=10.0)
    policy_version: int = Field(ge=0)
    mode: Literal["train", "evaluate"]

    @model_validator(mode="after")
    def validate_distribution(self) -> PolicySample:
        if not 0 <= self.selected_action_index < len(self.action_distribution):
            raise ValueError("selected action index is outside the policy distribution")
        if any(not 0.0 <= probability <= 1.0 for probability in self.action_distribution):
            raise ValueError("action probabilities must be between zero and one")
        if abs(sum(self.action_distribution) - 1.0) > 1e-5:
            raise ValueError("action probabilities must sum to one")
        if (
            abs(
                self.action_distribution[self.selected_action_index]
                - self.selected_action_probability
            )
            > 1e-5
        ):
            raise ValueError("selected action probability does not match the distribution")
        return self


class AgentDecision(StrictModel):
    schema_id: Literal["violet.agent_decision.v1"] = "violet.agent_decision.v1"
    decision_id: UUID
    run_id: UUID
    episode_id: UUID
    step_id: int = Field(ge=0, le=500)
    role: Literal["red", "blue"]
    policy_id: Identifier
    observation_schema_id: Literal["red.range_observation.v2", "blue.range_observation.v2"]
    legal_action_mask: tuple[bool, ...] = Field(min_length=1, max_length=32)
    selected_action: dict[str, object]
    inference_ms: float = Field(ge=0, le=60000)
    policy_sample: PolicySample | None = None

    @model_validator(mode="after")
    def role_observation_matches(self) -> AgentDecision:
        expected = f"{self.role}.range_observation.v2"
        if self.observation_schema_id != expected:
            raise ValueError("observation schema does not match policy role")
        if self.policy_sample is not None:
            if len(self.policy_sample.action_distribution) != len(self.legal_action_mask):
                raise ValueError("policy distribution does not match the action catalog")
            if not self.legal_action_mask[self.policy_sample.selected_action_index]:
                raise ValueError("sampled action is masked as illegal")
            if any(
                probability > 1e-8 and not allowed
                for probability, allowed in zip(
                    self.policy_sample.action_distribution, self.legal_action_mask, strict=True
                )
            ):
                raise ValueError("policy distribution assigns probability to a masked action")
        return self


class TransitionSubmission(StrictModel):
    """Episode-boundary link from a sampled action to its next policy view."""

    decision_id: UUID
    next_observation: RangeObservation
    next_value: float = Field(ge=-1e6, le=1e6)
    terminated: bool = False
    truncated: bool = False
    mode: Literal["train", "evaluate"]

    @model_validator(mode="after")
    def validate_terminal_flags(self) -> TransitionSubmission:
        if self.terminated and self.truncated:
            raise ValueError("transition cannot be both terminated and truncated")
        return self


class BlueAction(StrictModel):
    """Finite defensive action catalog; no caller-controlled target parameters."""

    operation: Literal["MONITOR", "HARDEN_DOCUMENT_OWNERSHIP", "RATE_LIMIT_ACTOR", "BLOCK_ACTOR"]
    principal_ref: Literal["alice", "bob"] | None = None

    @model_validator(mode="after")
    def validate_principal(self) -> BlueAction:
        if self.operation in {"RATE_LIMIT_ACTOR", "BLOCK_ACTOR"} and self.principal_ref is None:
            raise ValueError("this action requires an observed synthetic principal")
        if (
            self.operation in {"MONITOR", "HARDEN_DOCUMENT_OWNERSHIP"}
            and self.principal_ref is not None
        ):
            raise ValueError("this action does not accept a principal reference")
        return self


class DiscoverRouteAction(StrictModel):
    operation: Literal["DISCOVER_ROUTE"]
    route_candidate_id: Literal["home", "login", "document"]


class AuthTestAction(StrictModel):
    operation: Literal["AUTH_TEST"]
    credential_ref: Literal["synthetic_alice", "synthetic_bob", "synthetic_invalid"]


class CheckDocumentAccessAction(StrictModel):
    operation: Literal["CHECK_DOCUMENT_ACCESS"]
    document_ref: Literal["doc_alice", "doc_bob"]
    session_alias: Identifier | None = None

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_session_handle(cls, value: object) -> object:
        if isinstance(value, dict) and "session_handle" in value:
            compatible = dict(value)
            compatible.pop("session_handle", None)
            return compatible
        return value


class CheckTargetHealthAction(StrictModel):
    operation: Literal["CHECK_TARGET_HEALTH"]


class WaitAction(StrictModel):
    operation: Literal["WAIT"]


RangeAction: TypeAlias = Annotated[
    DiscoverRouteAction
    | AuthTestAction
    | CheckDocumentAccessAction
    | CheckTargetHealthAction
    | WaitAction,
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
    purpose: Literal["policy", "evaluator", "manual"] = "policy"
    deadline_seconds: float = Field(default=5.0, gt=0, le=5.0)
    action: RangeAction


class ExecutorJobRequest(StrictModel):
    action: RangeAction
    run_id: UUID | None = None
    episode_id: UUID | None = None
    step_id: int = Field(default=0, ge=0, le=500)
    decision_id: UUID | None = None
    idempotency_key: UUID | None = None
    purpose: Literal["policy", "evaluator", "manual"] = "policy"


class ExecutorResult(StrictModel):
    schema_id: Literal["violet.executor_result.v1"] = "violet.executor_result.v1"
    job_id: UUID
    action_id: UUID
    request_id: UUID
    status: Literal["completed", "rejected", "failed", "outcome_unknown"]
    method: Literal["GET", "POST"] | None = None
    route_catalog_id: Literal["home", "login", "document", "health"] | None = None
    response_headers: dict[str, str] = Field(default_factory=dict, max_length=8)
    http_status: int | None = Field(default=None, ge=100, le=599)
    content_type: str | None = None
    response_excerpt: str = Field(default="", max_length=2048)
    target_event_id: UUID | None = None
    session_alias: Identifier | None = None
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
            "executor_restarted_in_flight",
        ]
        | None
    ) = None
    started_at: str
    finished_at: str
    duration_ms: float = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_session_handle(cls, value: object) -> object:
        if isinstance(value, dict) and "session_handle" in value:
            compatible = dict(value)
            compatible.pop("session_handle", None)
            return compatible
        return value

    @model_validator(mode="after")
    def validate_safe_response_headers(self) -> ExecutorResult:
        allowed = {
            "content-type",
            "cache-control",
            "x-violet-target-event-id",
            "x-violet-target-version",
        }
        if any(
            key.lower() not in allowed or len(value) > 256
            for key, value in self.response_headers.items()
        ):
            raise ValueError("response headers contain an unsupported or oversized field")
        return self


class ControlRequest(StrictModel):
    action: Literal["HARDEN_DOCUMENT_OWNERSHIP", "RATE_LIMIT_ACTOR", "BLOCK_ACTOR"]
    principal_ref: Literal["alice", "bob"] | None = None
    run_id: UUID | None = None
    episode_id: UUID | None = None
    decision_id: UUID | None = None
    command_id: UUID | None = None
    purpose: Literal["policy", "evaluator"] = "policy"

    @model_validator(mode="after")
    def validate_principal(self) -> ControlRequest:
        if self.action in {"RATE_LIMIT_ACTOR", "BLOCK_ACTOR"} and self.principal_ref is None:
            raise ValueError("this control requires a principal reference")
        if self.action == "HARDEN_DOCUMENT_OWNERSHIP" and self.principal_ref is not None:
            raise ValueError("ownership hardening does not accept a principal reference")
        return self


class ControlVerificationRequest(StrictModel):
    unauthorized_job_id: UUID
    alice_owner_job_id: UUID
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
    decision_id: UUID | None = None
    purpose: Literal["policy", "evaluator"] = "policy"

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
