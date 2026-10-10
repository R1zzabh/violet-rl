"""Export a persisted real-range run as redacted JSONL with checksums."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from uuid import UUID

from violet.persistence.export import export_run
from violet.persistence.repository import EventRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=UUID)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs",
    )
    args = parser.parse_args()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL must point to the VIOLET event database")
    repository = EventRepository(database_url)
    try:
        output = export_run(repository, str(args.run_id), args.output_root)
    finally:
        repository.close()
    print(output)


if __name__ == "__main__":
    main()
