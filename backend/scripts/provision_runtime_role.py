"""Create a least-privilege PostgreSQL runtime role for one service database."""

from __future__ import annotations

import os
import re

from sqlalchemy import create_engine, text

_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")


def main() -> None:
    owner_url = os.environ.get("DATABASE_URL") or os.environ.get("TARGET_DATABASE_URL")
    role = os.environ.get("RUNTIME_DB_USER", "")
    password = os.environ.get("RUNTIME_DB_PASSWORD", "")
    if not owner_url or not _IDENTIFIER.fullmatch(role) or not password:
        raise RuntimeError("owner URL, safe runtime role name, and runtime password are required")

    quoted_role = f'"{role}"'
    engine = create_engine(owner_url, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": role}
            ).first()
            if exists:
                connection.exec_driver_sql(
                    f"ALTER ROLE {quoted_role} WITH LOGIN PASSWORD %s NOSUPERUSER "
                    "NOCREATEDB NOCREATEROLE NOINHERIT",
                    (password,),
                )
            else:
                connection.exec_driver_sql(
                    f"CREATE ROLE {quoted_role} WITH LOGIN PASSWORD %s NOSUPERUSER "
                    "NOCREATEDB NOCREATEROLE NOINHERIT",
                    (password,),
                )
            database_name = str(connection.execute(text("SELECT current_database()")).scalar_one())
            quoted_database = connection.dialect.identifier_preparer.quote(database_name)
            connection.exec_driver_sql(
                f"GRANT CONNECT ON DATABASE {quoted_database} TO {quoted_role}"
            )
            connection.exec_driver_sql(f"GRANT USAGE ON SCHEMA public TO {quoted_role}")
            connection.exec_driver_sql(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {quoted_role}"
            )
            connection.exec_driver_sql(
                f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {quoted_role}"
            )
            connection.exec_driver_sql(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE "
                f"ON TABLES TO {quoted_role}"
            )
            connection.exec_driver_sql(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT "
                f"ON SEQUENCES TO {quoted_role}"
            )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
