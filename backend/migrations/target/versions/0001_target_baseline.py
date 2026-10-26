"""Create the explicit disposable Violet Ledger target schema."""

from __future__ import annotations

from alembic import op
from sqlalchemy import Column, String, UniqueConstraint, inspect

revision = "0001_target_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(inspect(op.get_bind()).get_table_names())
    if "ledger_users" not in existing:
        op.create_table(
            "ledger_users",
            Column("user_ref", String(32), primary_key=True),
            Column("display_name", String(64), nullable=False),
        )
    if "ledger_documents" not in existing:
        op.create_table(
            "ledger_documents",
            Column("document_ref", String(32), primary_key=True),
            Column("owner_ref", String(32), nullable=False),
            Column("canary", String(128), nullable=False),
            Column("body", String(256), nullable=False),
            UniqueConstraint("canary"),
        )
    if "ledger_range_state" not in existing:
        op.create_table(
            "ledger_range_state",
            Column("key", String(32), primary_key=True),
            Column("value", String(64), nullable=False),
        )

    required = {
        "ledger_users": {"user_ref", "display_name"},
        "ledger_documents": {"document_ref", "owner_ref", "canary", "body"},
        "ledger_range_state": {"key", "value"},
    }
    inspector = inspect(op.get_bind())
    for table, columns in required.items():
        actual = {column["name"] for column in inspector.get_columns(table)}
        if not columns.issubset(actual):
            missing = ", ".join(sorted(columns - actual))
            raise RuntimeError(f"target table {table} is missing columns: {missing}")


def downgrade() -> None:
    raise RuntimeError(
        "target baseline contains synthetic fixture data; reset only disposable data"
    )
