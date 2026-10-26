"""Create the explicit baseline schema for the control-plane event store."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    inspect,
)

revision = "0001_control_baseline"
down_revision = None
branch_labels = None
depends_on = None


def _create_index_if_missing(name: str, table: str, columns: list[str]) -> None:
    indexes = {item["name"] for item in inspect(op.get_bind()).get_indexes(table)}
    if name not in indexes:
        op.create_index(name, table, columns)


def upgrade() -> None:
    existing = set(inspect(op.get_bind()).get_table_names())
    if "range_runs" not in existing:
        op.create_table(
            "range_runs",
            Column("run_id", String(36), primary_key=True),
            Column("mode", String(32), nullable=False),
            Column("target_version", String(64), nullable=False),
            Column("status", String(24), nullable=False),
            Column("config", JSON, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("next_seq", Integer, nullable=False),
            Column("last_event_hash", String(64), nullable=False),
        )
    if "range_episodes" not in existing:
        op.create_table(
            "range_episodes",
            Column("episode_id", String(36), primary_key=True),
            Column("run_id", String(36), ForeignKey("range_runs.run_id"), nullable=False),
            Column("episode_index", Integer, nullable=False),
            Column("seed", Integer, nullable=True),
            Column("status", String(24), nullable=False),
            Column("config", JSON, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("run_id", "episode_index"),
        )
    if "range_run_events" not in existing:
        op.create_table(
            "range_run_events",
            Column("event_id", String(36), primary_key=True),
            Column("run_id", String(36), ForeignKey("range_runs.run_id"), nullable=False),
            Column(
                "episode_id", String(36), ForeignKey("range_episodes.episode_id"), nullable=True
            ),
            Column("seq", Integer, nullable=False),
            Column("event_type", String(64), nullable=False),
            Column("occurred_at", DateTime(timezone=True), nullable=False),
            Column("payload", JSON, nullable=False),
            Column("previous_hash", String(64), nullable=False),
            Column("event_hash", String(64), nullable=False),
            UniqueConstraint("run_id", "seq"),
        )
    if "range_executor_jobs" not in existing:
        op.create_table(
            "range_executor_jobs",
            Column("job_id", String(36), primary_key=True),
            Column("run_id", String(36), ForeignKey("range_runs.run_id"), nullable=False),
            Column(
                "episode_id", String(36), ForeignKey("range_episodes.episode_id"), nullable=False
            ),
            Column("action_id", String(36), nullable=False, unique=True),
            Column("idempotency_key", String(36), nullable=False, unique=True),
            Column("status", String(24), nullable=False),
            Column("payload", JSON, nullable=False),
            Column("result", JSON, nullable=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("finished_at", DateTime(timezone=True), nullable=True),
        )
    if "range_control_commands" not in existing:
        op.create_table(
            "range_control_commands",
            Column("command_id", String(36), primary_key=True),
            Column("run_id", String(36), ForeignKey("range_runs.run_id"), nullable=False),
            Column(
                "episode_id", String(36), ForeignKey("range_episodes.episode_id"), nullable=False
            ),
            Column("status", String(24), nullable=False),
            Column("payload", JSON, nullable=False),
            Column("acknowledgement", JSON, nullable=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )
    if "range_target_evidence" not in existing:
        op.create_table(
            "range_target_evidence",
            Column("target_event_id", String(36), primary_key=True),
            Column("run_id", String(36), ForeignKey("range_runs.run_id"), nullable=False),
            Column(
                "episode_id", String(36), ForeignKey("range_episodes.episode_id"), nullable=False
            ),
            Column("action_id", String(36), nullable=False),
            Column("request_id", String(36), nullable=False),
            Column("source", String(32), nullable=False),
            Column("occurred_at", DateTime(timezone=True), nullable=False),
            Column("payload", JSON, nullable=False),
        )

    expected = {
        "range_runs": {
            "run_id",
            "mode",
            "target_version",
            "status",
            "config",
            "created_at",
            "next_seq",
            "last_event_hash",
        },
        "range_episodes": {
            "episode_id",
            "run_id",
            "episode_index",
            "seed",
            "status",
            "config",
            "created_at",
        },
        "range_run_events": {
            "event_id",
            "run_id",
            "episode_id",
            "seq",
            "event_type",
            "occurred_at",
            "payload",
            "previous_hash",
            "event_hash",
        },
        "range_executor_jobs": {
            "job_id",
            "run_id",
            "episode_id",
            "action_id",
            "idempotency_key",
            "status",
            "payload",
            "result",
            "created_at",
            "finished_at",
        },
        "range_control_commands": {
            "command_id",
            "run_id",
            "episode_id",
            "status",
            "payload",
            "acknowledgement",
            "created_at",
        },
        "range_target_evidence": {
            "target_event_id",
            "run_id",
            "episode_id",
            "action_id",
            "request_id",
            "source",
            "occurred_at",
            "payload",
        },
    }
    inspector = inspect(op.get_bind())
    for table, required in expected.items():
        actual = {column["name"] for column in inspector.get_columns(table)}
        if not required.issubset(actual):
            missing = ", ".join(sorted(required - actual))
            raise RuntimeError(f"control baseline table {table} is missing columns: {missing}")

    _create_index_if_missing(
        "ix_range_events_episode_seq", "range_run_events", ["episode_id", "seq"]
    )
    _create_index_if_missing(
        "ix_range_events_type_time", "range_run_events", ["run_id", "event_type", "occurred_at"]
    )
    _create_index_if_missing("ix_range_evidence_request", "range_target_evidence", ["request_id"])
    _create_index_if_missing(
        "ix_range_evidence_episode_time", "range_target_evidence", ["episode_id", "occurred_at"]
    )


def downgrade() -> None:
    raise RuntimeError("control baseline contains durable evidence; restore a database backup")
