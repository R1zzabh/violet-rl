"""Create and seed the isolated SQL injection lab database."""

from __future__ import annotations

from alembic import op
from sqlalchemy import Column, String, inspect

revision = "0001_sqli_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(inspect(op.get_bind()).get_table_names())
    if "sqli_records" not in existing:
        table = op.create_table(
            "sqli_records",
            Column("record_ref", String(32), primary_key=True),
            Column("owner_ref", String(32), nullable=False),
            Column("handle", String(48), nullable=False),
            Column("department", String(48), nullable=False),
            Column("city", String(48), nullable=False),
            Column("email", String(96), nullable=False),
        )
        from violet.sqli_lab.database import SYNTHETIC_RECORDS

        op.bulk_insert(table, list(SYNTHETIC_RECORDS))
    required = {"record_ref", "owner_ref", "handle", "department", "city", "email"}
    actual = {column["name"] for column in inspect(op.get_bind()).get_columns("sqli_records")}
    if not required.issubset(actual):
        raise RuntimeError("SQLi lab table is missing required fixture columns")


def downgrade() -> None:
    raise RuntimeError("SQLi lab fixtures are durable; restore a disposable database to reset")
