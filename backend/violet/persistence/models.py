"""SQLAlchemy records for the canonical real-range event timeline."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "range_runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="real_range")
    target_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="running")
    config: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    next_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_event_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")


class Episode(Base):
    __tablename__ = "range_episodes"
    __table_args__ = (UniqueConstraint("run_id", "episode_index"),)

    episode_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("range_runs.run_id"), nullable=False)
    episode_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="running")
    config: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class RunEvent(Base):
    __tablename__ = "range_run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "seq"),
        Index("ix_range_events_episode_seq", "episode_id", "seq"),
        Index("ix_range_events_type_time", "run_id", "event_type", "occurred_at"),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("range_runs.run_id"), nullable=False)
    episode_id: Mapped[str | None] = mapped_column(
        ForeignKey("range_episodes.episode_id"), nullable=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    previous_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class ExecutorJobRecord(Base):
    __tablename__ = "range_executor_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("range_runs.run_id"), nullable=False)
    episode_id: Mapped[str] = mapped_column(ForeignKey("range_episodes.episode_id"), nullable=False)
    action_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    idempotency_key: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="queued")
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    result: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ControlRecord(Base):
    __tablename__ = "range_control_commands"

    command_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("range_runs.run_id"), nullable=False)
    episode_id: Mapped[str] = mapped_column(ForeignKey("range_episodes.episode_id"), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="queued")
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    acknowledgement: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class TargetEvidence(Base):
    __tablename__ = "range_target_evidence"
    __table_args__ = (
        Index("ix_range_evidence_request", "request_id"),
        Index("ix_range_evidence_episode_time", "episode_id", "occurred_at"),
    )

    target_event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("range_runs.run_id"), nullable=False)
    episode_id: Mapped[str] = mapped_column(ForeignKey("range_episodes.episode_id"), nullable=False)
    action_id: Mapped[str] = mapped_column(String(36), nullable=False)
    request_id: Mapped[str] = mapped_column(String(36), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)


class AgentDecisionRecord(Base):
    """Immutable, sanitized policy input/output committed before its action."""

    __tablename__ = "range_decisions"
    __table_args__ = (
        UniqueConstraint("episode_id", "step_id", "role"),
        Index("ix_range_decisions_policy", "policy_id"),
    )

    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("range_runs.run_id"), nullable=False)
    episode_id: Mapped[str] = mapped_column(ForeignKey("range_episodes.episode_id"), nullable=False)
    step_id: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(8), nullable=False)
    policy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    observation_schema_id: Mapped[str] = mapped_column(String(64), nullable=False)
    observation: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    legal_action_mask: Mapped[list[bool]] = mapped_column(JSON, nullable=False)
    selected_action: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    policy_sample: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    inference_ms: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class AgentTransitionRecord(Base):
    """Immutable, evidence-linked transition joined to its sampled decision."""

    __tablename__ = "range_transitions"
    __table_args__ = (Index("ix_range_transitions_episode", "episode_id", "step_id", "role"),)

    decision_id: Mapped[str] = mapped_column(
        ForeignKey("range_decisions.decision_id"), primary_key=True
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("range_runs.run_id"), nullable=False)
    episode_id: Mapped[str] = mapped_column(ForeignKey("range_episodes.episode_id"), nullable=False)
    step_id: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(8), nullable=False)
    accepted: Mapped[bool] = mapped_column(nullable=False)
    execution_status: Mapped[str] = mapped_column(String(24), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class RangeDetectionRecord(Base):
    __tablename__ = "range_detections"

    detection_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("range_runs.run_id"), nullable=False)
    episode_id: Mapped[str] = mapped_column(ForeignKey("range_episodes.episode_id"), nullable=False)
    source_target_event_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    detector_version: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class RangeRewardRecord(Base):
    __tablename__ = "range_rewards"
    __table_args__ = (UniqueConstraint("episode_id", "step_id", "role"),)

    reward_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("range_runs.run_id"), nullable=False)
    episode_id: Mapped[str] = mapped_column(ForeignKey("range_episodes.episode_id"), nullable=False)
    step_id: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(8), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class TrainingRunRecord(Base):
    """Durable control-plane state for range training and evaluation jobs."""

    __tablename__ = "range_training_runs"

    training_run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    role: Mapped[str] = mapped_column(String(8), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="queued", index=True)
    config: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stop_requested: Mapped[bool] = mapped_column(default=False, nullable=False)
    result: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class TrainingUpdateRecord(Base):
    """Metrics/checkpoint provenance emitted after a PPO optimization update."""

    __tablename__ = "range_training_updates"
    __table_args__ = (
        UniqueConstraint(
            "training_run_id", "role", "update_index", name="uq_range_update_role_index"
        ),
    )

    update_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("range_training_runs.training_run_id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(8), nullable=False)
    update_index: Mapped[int] = mapped_column(Integer, nullable=False)
    global_step: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
