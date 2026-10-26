"""Small, versioned action grammar and target control contracts for the SQLi lab."""

from __future__ import annotations

from typing import Literal
from uuid import UUID  # noqa: TC003 - Pydantic needs the runtime UUID class for schema validation.

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


FieldName = Literal["handle", "department", "city", "email"]
BaseTerm = Literal["rhea", "alice", "bob", "engineering", "finance"]
Connector = Literal["none", "OR", "AND"]
Predicate = Literal["none", "tautology", "contradiction"]
ControlAction = Literal["MONITOR", "PARAMETERIZE_QUERY", "RATE_LIMIT_GUEST", "ISOLATE_GUEST_LOOKUP"]

OBSERVATION_SCHEMA = "sqli.range_observation.v1"
ACTION_SCHEMA = "sqli.composed_action.v1"
REWARD_SCHEMA = "sqli.reward.v1"
FEATURE_SCHEMA = "sqli.range_features.v1"
SCENARIO_ID = "sqli_lab_v1"
TARGET_VERSION = "sqli_target_v1"
RED_COMPONENTS = {
    "field": ("handle", "department", "city", "email"),
    "base_term": ("rhea", "alice", "bob", "engineering", "finance"),
    "close_quote": ("no", "yes"),
    "connector": ("none", "OR", "AND"),
    "predicate": ("none", "tautology", "contradiction"),
    "comment": ("no", "yes"),
}
BLUE_ACTIONS: tuple[ControlAction, ...] = (
    "MONITOR",
    "PARAMETERIZE_QUERY",
    "RATE_LIMIT_GUEST",
    "ISOLATE_GUEST_LOOKUP",
)


class SqliObservation(StrictModel):
    schema_id: Literal["red.sqli.range_observation.v1", "blue.sqli.range_observation.v1"]
    run_id: UUID
    episode_id: UUID
    step_id: int = Field(ge=0, le=40)
    features: tuple[float, ...] = Field(min_length=1, max_length=32)
    telemetry: dict[str, object] = Field(default_factory=dict)
    remaining_requests: int = Field(ge=0, le=40)
    legal_action_mask: tuple[bool, ...] = Field(min_length=1, max_length=32)


class SqliRequestAction(StrictModel):
    schema_id: Literal["sqli.composed_action.v1"] = "sqli.composed_action.v1"
    field: FieldName
    base_term: BaseTerm
    close_quote: bool
    connector: Connector
    predicate: Predicate
    comment: bool

    def compose_value(self) -> str:
        """Compose a bounded input from grammar parts, never from a payload catalog."""
        value: str = self.base_term
        if self.close_quote:
            value += "'"
        if self.connector != "none":
            value += f" {self.connector}"
            if self.predicate != "none":
                value += " 1=1" if self.predicate == "tautology" else " 1=0"
        if self.comment:
            value += " -- "
        if len(value) > 96:
            raise ValueError("composed lookup input exceeds the 96 character limit")
        return value

    def components(self) -> dict[str, str | bool]:
        return {
            "field": self.field,
            "base_term": self.base_term,
            "close_quote": self.close_quote,
            "connector": self.connector,
            "predicate": self.predicate,
            "comment": self.comment,
        }


class SqliJob(StrictModel):
    schema_id: Literal["sqli.executor_job.v1"] = "sqli.executor_job.v1"
    job_id: UUID
    run_id: UUID
    episode_id: UUID
    decision_id: UUID
    step_id: int = Field(ge=0, le=40)
    request_id: UUID
    action: SqliRequestAction
    purpose: Literal["policy", "verifier_attack", "verifier_legitimate"]
    deadline_seconds: float = Field(default=4.0, gt=0, le=4.0)


class SqliExecutorResult(StrictModel):
    schema_id: Literal["sqli.executor_result.v1"] = "sqli.executor_result.v1"
    job_id: UUID
    run_id: UUID
    episode_id: UUID
    decision_id: UUID
    step_id: int = Field(ge=0, le=40)
    request_id: UUID
    purpose: Literal["policy", "verifier_attack", "verifier_legitimate"]
    status: Literal["completed", "failed", "outcome_unknown", "rejected"]
    actor_ref: Literal["guest", "researcher"]
    method: Literal["POST"] = "POST"
    route_id: Literal["lookup"] = "lookup"
    http_status: int | None = Field(default=None, ge=100, le=599)
    target_event_id: UUID | None = None
    request: dict[str, object]
    response: dict[str, object] = Field(default_factory=dict)
    telemetry: dict[str, object] = Field(default_factory=dict)
    error_code: (
        Literal["timeout", "connection_failed", "executor_error", "request_budget"] | None
    ) = None
    duration_ms: float = Field(ge=0)
    started_at: str
    finished_at: str


class SqliControlCommand(StrictModel):
    schema_id: Literal["sqli.control.v1"] = "sqli.control.v1"
    command_id: UUID
    run_id: UUID
    episode_id: UUID
    decision_id: UUID | None = None
    action: Literal["RESET", "PARAMETERIZE_QUERY", "RATE_LIMIT_GUEST", "ISOLATE_GUEST_LOOKUP"]


class SqliControlAcknowledgement(StrictModel):
    schema_id: Literal["sqli.control_ack.v1"] = "sqli.control_ack.v1"
    command_id: UUID
    run_id: UUID
    episode_id: UUID
    status: Literal["applied", "failed", "rejected"]
    state_version: int = Field(ge=0)
    controls: dict[str, object] = Field(default_factory=dict)
    completed_at: str
    error_code: str | None = None
