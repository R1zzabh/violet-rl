"""Create the durable transition projection for full PPO replay."""

from __future__ import annotations

from alembic import op
from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String, inspect

revision = "0004_range_transitions"
down_revision = "0003_decision_policy_samples"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if "range_transitions" not in inspector.get_table_names():
        op.create_table(
            "range_transitions",
            Column(
                "decision_id",
                String(36),
                ForeignKey("range_decisions.decision_id"),
                primary_key=True,
            ),
            Column("run_id", String(36), ForeignKey("range_runs.run_id"), nullable=False),
            Column(
                "episode_id", String(36), ForeignKey("range_episodes.episode_id"), nullable=False
            ),
            Column("step_id", Integer(), nullable=False),
            Column("role", String(8), nullable=False),
            Column("accepted", Boolean(), nullable=False),
            Column("execution_status", String(24), nullable=False),
            Column("mode", String(16), nullable=False),
            Column("payload", JSON(), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )
    indexes = {item["name"] for item in inspect(op.get_bind()).get_indexes("range_transitions")}
    if "ix_range_transitions_episode" not in indexes:
        op.create_index(
            "ix_range_transitions_episode",
            "range_transitions",
            ["episode_id", "step_id", "role"],
        )


def downgrade() -> None:
    raise RuntimeError("range transitions are immutable experiment evidence")
