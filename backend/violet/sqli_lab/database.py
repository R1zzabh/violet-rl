"""Read-only query adapter for the lab's separate synthetic-record database."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    bindparam,
    create_engine,
    event,
    select,
    text,
)

metadata = MetaData()
records = Table(
    "sqli_records",
    metadata,
    Column("record_ref", String(32), primary_key=True),
    Column("owner_ref", String(32), nullable=False),
    Column("handle", String(48), nullable=False),
    Column("department", String(48), nullable=False),
    Column("city", String(48), nullable=False),
    Column("email", String(96), nullable=False),
)

SYNTHETIC_RECORDS: tuple[dict[str, str], ...] = (
    {
        "record_ref": "rec-001",
        "owner_ref": "researcher",
        "handle": "rhea",
        "department": "engineering",
        "city": "oslo",
        "email": "rhea@example.test",
    },
    {
        "record_ref": "rec-002",
        "owner_ref": "alice",
        "handle": "alice",
        "department": "finance",
        "city": "lisbon",
        "email": "alice@example.test",
    },
    {
        "record_ref": "rec-003",
        "owner_ref": "bob",
        "handle": "bob",
        "department": "engineering",
        "city": "helsinki",
        "email": "bob@example.test",
    },
    {
        "record_ref": "rec-004",
        "owner_ref": "operations",
        "handle": "mika",
        "department": "operations",
        "city": "oslo",
        "email": "mika@example.test",
    },
    {
        "record_ref": "rec-005",
        "owner_ref": "alice",
        "handle": "ana",
        "department": "finance",
        "city": "porto",
        "email": "ana@example.test",
    },
    {
        "record_ref": "rec-006",
        "owner_ref": "bob",
        "handle": "lee",
        "department": "engineering",
        "city": "tallinn",
        "email": "lee@example.test",
    },
)


class SqliDatabase:
    """The application role can only SELECT from the fixture table."""

    def __init__(self, url: str | None = None):
        state_home = Path(os.getenv("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        local_database = state_home / "violet/sqli-lab/database.sqlite3"
        if url is None and os.getenv("SQLI_DATABASE_URL") is None:
            local_database.parent.mkdir(parents=True, exist_ok=True)
        self.url = url or os.getenv("SQLI_DATABASE_URL") or f"sqlite:///{local_database}"
        args: dict[str, Any] = {"check_same_thread": False} if self.url.startswith("sqlite") else {}
        self.engine = create_engine(self.url, pool_pre_ping=True, connect_args=args)
        if self.engine.dialect.name == "sqlite":
            metadata.create_all(self.engine)
            with self.engine.begin() as connection:
                count: int = connection.execute(
                    text("SELECT COUNT(*) FROM sqli_records")
                ).scalar_one()
                if count == 0:
                    connection.execute(records.insert(), list(SYNTHETIC_RECORDS))
            event.listen(self.engine, "connect", self._sqlite_read_only)
            # Apply query_only to the already-open pooled connection as well as
            # all future connections. Keeping it in place preserves :memory:
            # databases used by focused target tests.
            with self.engine.connect() as connection:
                connection.exec_driver_sql("PRAGMA query_only=ON")
        else:
            from sqlalchemy import inspect

            if "sqli_records" not in inspect(self.engine).get_table_names():
                raise RuntimeError("SQLi lab schema is not migrated; run sqli-migrate first")
            event.listen(self.engine, "connect", self._postgres_read_only)

    @staticmethod
    def _sqlite_read_only(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA query_only=ON")
        cursor.close()

    @staticmethod
    def _postgres_read_only(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("SET default_transaction_read_only = on")
        cursor.close()

    def execute_lookup(
        self,
        *,
        actor_ref: str,
        field: str,
        query_value: str,
        parameterized: bool,
    ) -> list[dict[str, str]]:
        # The field is validated by the strict request contract before it reaches here.
        projection = "record_ref, owner_ref, handle, department, city, email"
        statement: Any
        if parameterized:
            statement = (
                select(records)
                .where(
                    records.c.owner_ref == bindparam("actor_ref"),
                    getattr(records.c, field) == bindparam("query_value"),
                )
                .limit(8)
            )
            parameters = {"actor_ref": actor_ref, "query_value": query_value}
        else:
            # This deliberately vulnerable, read-only statement is the lab objective.
            statement = text(
                f"SELECT {projection} FROM sqli_records "
                f"WHERE owner_ref = '{actor_ref}' AND {field} = '{query_value}' LIMIT 8"  # nosec B608
            )
            parameters = {}
        if not parameterized and len(str(statement)) > 220:
            raise ValueError("constructed query exceeds the 220 character lab limit")
        with self.engine.connect() as connection:
            rows = connection.execute(statement, parameters).mappings().all()
        return [{key: str(value) for key, value in row.items()} for row in rows]

    def close(self) -> None:
        self.engine.dispose()
