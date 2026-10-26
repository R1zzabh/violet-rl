"""Persist the complete categorical policy sample on each agent decision."""

from __future__ import annotations

from alembic import op
from sqlalchemy import JSON, Column, inspect

revision = "0003_decision_policy_samples"
down_revision = "0002_range_policy_records"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("range_decisions")}
    if "policy_sample" not in columns:
        op.add_column("range_decisions", Column("policy_sample", JSON, nullable=True))


def downgrade() -> None:
    raise RuntimeError("decision policy samples are part of immutable experiment evidence")
