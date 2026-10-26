"""Allow independent Red and Blue optimizer updates at the same episode index."""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

revision = "0005_role_scoped_update_indexes"
down_revision = "0004_range_transitions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    constraints = inspect(op.get_bind()).get_unique_constraints("range_training_updates")
    old = next(
        (
            item
            for item in constraints
            if item.get("column_names") == ["training_run_id", "update_index"]
        ),
        None,
    )
    with op.batch_alter_table("range_training_updates") as batch:
        if old and old.get("name"):
            batch.drop_constraint(old["name"], type_="unique")
        if not any(
            item.get("column_names") == ["training_run_id", "role", "update_index"]
            for item in constraints
        ):
            batch.create_unique_constraint(
                "uq_range_update_role_index",
                ["training_run_id", "role", "update_index"],
            )


def downgrade() -> None:
    raise RuntimeError("role-scoped policy updates are durable training evidence")
