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
