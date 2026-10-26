"""Synthetic Violet Ledger storage, isolated from VIOLET's experiment data."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    insert,
    select,
    update,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

metadata = MetaData()
users = Table(
    "ledger_users",
    metadata,
    Column("user_ref", String(32), primary_key=True),
    Column("display_name", String(64), nullable=False),
)
documents = Table(
    "ledger_documents",
    metadata,
    Column("document_ref", String(32), primary_key=True),
    Column("owner_ref", String(32), nullable=False),
    Column("canary", String(128), nullable=False, unique=True),
    Column("body", String(256), nullable=False),
)
range_state = Table(
    "ledger_range_state",
    metadata,
    Column("key", String(32), primary_key=True),
    Column("value", String(64), nullable=False),
)


class TargetDatabase:
    """Owns target fixtures and resets only Violet Ledger's disposable data."""

    def __init__(self, url: str):
        connect_args: dict[str, Any] = (
            {"check_same_thread": False} if url.startswith("sqlite") else {}
        )
        self.engine: Engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        # Local test fixtures may use SQLite. PostgreSQL is initialized only
        # by the database-scoped target-migrate service.
        if self.engine.dialect.name == "sqlite":
            metadata.create_all(self.engine)
        else:
            from sqlalchemy import inspect

            missing = set(metadata.tables) - set(inspect(self.engine).get_table_names())
            if missing:
                raise RuntimeError(
                    "Violet Ledger database schema is not migrated; run target-migrate "
                    f"(missing: {', '.join(sorted(missing))})"
                )
        with self.engine.connect() as connection:
            has_fixtures = connection.execute(select(users.c.user_ref).limit(1)).first() is not None
        if not has_fixtures:
            self.reset(0)

    def reset(self, seed: int) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(documents))
            connection.execute(delete(users))
            connection.execute(delete(range_state))
            connection.execute(
                insert(users),
                [
                    {"user_ref": "alice", "display_name": "Alice"},
                    {"user_ref": "bob", "display_name": "Bob"},
                ],
            )
            for document_ref, owner_ref, body in (
                ("doc_alice", "alice", "Alice's synthetic project ledger."),
                ("doc_bob", "bob", "Bob's synthetic project ledger."),
            ):
                digest = hashlib.sha256(f"violet-ledger:{seed}:{document_ref}".encode()).hexdigest()
                canary = f"VIOLET-CANARY-{digest[:24].upper()}"
                connection.execute(
                    insert(documents).values(
                        document_ref=document_ref,
                        owner_ref=owner_ref,
                        canary=canary,
                        body=body,
                    )
                )
            connection.execute(insert(range_state).values(key="fixture_seed", value=str(seed)))

    def get_document(self, document_ref: str) -> dict[str, str] | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(documents).where(documents.c.document_ref == document_ref)
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def runtime_state(self) -> dict[str, str]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(range_state)).mappings().all()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def save_runtime_state(self, values: dict[str, str]) -> None:
        with self.engine.begin() as connection:
            for key, value in values.items():
                exists = connection.execute(
                    select(range_state.c.key).where(range_state.c.key == key)
                ).first()
                if exists:
                    connection.execute(
                        update(range_state).where(range_state.c.key == key).values(value=value)
                    )
                else:
                    connection.execute(insert(range_state).values(key=key, value=value))

    def owner_of(self, document_ref: str) -> str | None:
        with self.engine.connect() as connection:
            return connection.execute(
                select(documents.c.owner_ref).where(documents.c.document_ref == document_ref)
            ).scalar_one_or_none()

    @property
    def fixture_seed(self) -> int:
        with self.engine.connect() as connection:
            value: str = connection.execute(
                select(range_state.c.value).where(range_state.c.key == "fixture_seed")
            ).scalar_one()
        return int(value)
