"""Add explicit policy decisions, rewards, detections, and job records."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    inspect,
)

revision = "0002_range_policy_records"
down_revision = "0001_control_baseline"
branch_labels = None
depends_on = None


def _create_index_if_missing(name: str, table: str, columns: list[str]) -> None:
    indexes = {item["name"] for item in inspect(op.get_bind()).get_indexes(table)}
    if name not in indexes:
        op.create_index(name, table, columns)


def upgrade() -> None:
    existing = set(inspect(op.get_bind()).get_table_names())
    if "range_decisions" not in existing:
        op.create_table(
            "range_decisions",
            Column("decision_id", String(36), primary_key=True),
            Column("run_id", String(36), ForeignKey("range_runs.run_id"), nullable=False),
            Column(
                "episode_id", String(36), ForeignKey("range_episodes.episode_id"), nullable=False
            ),
            Column("step_id", Integer, nullable=False),
            Column("role", String(8), nullable=False),
            Column("policy_id", String(64), nullable=False),
            Column("observation_schema_id", String(64), nullable=False),
            Column("observation", JSON, nullable=False),
            Column("legal_action_mask", JSON, nullable=False),
            Column("selected_action", JSON, nullable=False),
            Column("inference_ms", Float, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("episode_id", "step_id", "role"),
        )
    if "range_detections" not in existing:
        op.create_table(
            "range_detections",
            Column("detection_id", String(36), primary_key=True),
            Column("run_id", String(36), ForeignKey("range_runs.run_id"), nullable=False),
            Column(
                "episode_id", String(36), ForeignKey("range_episodes.episode_id"), nullable=False
            ),
            Column("source_target_event_id", String(36), nullable=False),
            Column("detector_version", String(64), nullable=False),
            Column("payload", JSON, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )
    if "range_rewards" not in existing:
        op.create_table(
            "range_rewards",
            Column("reward_id", String(36), primary_key=True),
            Column("run_id", String(36), ForeignKey("range_runs.run_id"), nullable=False),
            Column(
                "episode_id", String(36), ForeignKey("range_episodes.episode_id"), nullable=False
            ),
            Column("step_id", Integer, nullable=False),
            Column("role", String(8), nullable=False),
            Column("payload", JSON, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("episode_id", "step_id", "role"),
        )
    if "range_training_runs" not in existing:
        op.create_table(
            "range_training_runs",
            Column("training_run_id", String(36), primary_key=True),
            Column("role", String(8), nullable=False),
            Column("mode", String(16), nullable=False),
            Column("status", String(24), nullable=False),
            Column("config", JSON, nullable=False),
            Column("worker_id", String(64), nullable=True),
            Column("stop_requested", Boolean, nullable=False),
            Column("result", JSON, nullable=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )
    if "range_training_updates" not in existing:
        op.create_table(
            "range_training_updates",
            Column("update_id", String(36), primary_key=True),
            Column(
                "training_run_id",
                String(36),
                ForeignKey("range_training_runs.training_run_id"),
                nullable=False,
            ),
            Column("role", String(8), nullable=False),
            Column("update_index", Integer, nullable=False),
            Column("global_step", Integer, nullable=False),
            Column("metrics", JSON, nullable=False),
            Column("checkpoint_sha256", String(64), nullable=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("training_run_id", "update_index", name="uq_range_update_index"),
        )

    required = {
        "range_decisions": {
            "decision_id",
            "run_id",
            "episode_id",
            "step_id",
            "role",
            "policy_id",
            "observation_schema_id",
            "observation",
            "legal_action_mask",
            "selected_action",
            "inference_ms",
            "created_at",
        },
        "range_detections": {
            "detection_id",
            "run_id",
            "episode_id",
            "source_target_event_id",
            "detector_version",
            "payload",
            "created_at",
        },
        "range_rewards": {
            "reward_id",
            "run_id",
            "episode_id",
            "step_id",
            "role",
            "payload",
            "created_at",
        },
        "range_training_runs": {
            "training_run_id",
            "role",
            "mode",
            "status",
            "config",
            "worker_id",
            "stop_requested",
            "result",
            "created_at",
            "updated_at",
        },
        "range_training_updates": {
            "update_id",
            "training_run_id",
            "role",
            "update_index",
            "global_step",
            "metrics",
            "checkpoint_sha256",
            "created_at",
        },
    }
    inspector = inspect(op.get_bind())
    for table, columns in required.items():
        actual = {column["name"] for column in inspector.get_columns(table)}
        if not columns.issubset(actual):
            missing = ", ".join(sorted(columns - actual))
            raise RuntimeError(f"range policy table {table} is missing columns: {missing}")

    _create_index_if_missing("ix_range_decisions_policy", "range_decisions", ["policy_id"])
    _create_index_if_missing(
        "ix_range_detections_source_target_event_id", "range_detections", ["source_target_event_id"]
    )
    _create_index_if_missing("ix_range_training_runs_status", "range_training_runs", ["status"])
    _create_index_if_missing(
        "ix_range_training_updates_training_run_id", "range_training_updates", ["training_run_id"]
    )


def downgrade() -> None:
    raise RuntimeError(
        "policy decisions and update evidence are durable; roll forward or restore backup"
    )
