"""Typed target controls and file-based command acknowledgements."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import ValidationError

from violet.range.contracts import ControlAcknowledgement, ControlCommand
from violet.range.spool import read_json, write_json_atomic

if TYPE_CHECKING:
    from pathlib import Path

    from violet.target_app.database import TargetDatabase


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class TargetRuntime:
    """Mutable target controls and synthetic sessions, never exposed to policies."""

    def __init__(self, database: TargetDatabase):
        self.database = database
        persisted = database.runtime_state()
        self.ownership_hardened = persisted.get("ownership_hardened") == "true"
        self.blocked_principals = set(
            filter(None, persisted.get("blocked_principals", "").split(","))
        )
        self.rate_limited_principals = set(
            filter(None, persisted.get("rate_limited_principals", "").split(","))
        )
        self.state_version = int(persisted.get("state_version", "0"))
        self.sessions: dict[str, str] = {}
        self.request_counts: dict[str, int] = {}
        self._persist()

    def _persist(self) -> None:
        self.database.save_runtime_state(
            {
                "ownership_hardened": str(self.ownership_hardened).lower(),
                "blocked_principals": ",".join(sorted(self.blocked_principals)),
                "rate_limited_principals": ",".join(sorted(self.rate_limited_principals)),
                "state_version": str(self.state_version),
            }
        )

    def clear_controls(self) -> None:
        self.ownership_hardened = False
        self.blocked_principals.clear()
        self.rate_limited_principals.clear()
        self.sessions.clear()
        self.request_counts.clear()
        self.state_version += 1
        self._persist()

    def apply(self, command: ControlCommand) -> ControlAcknowledgement:
        if command.command == "RESET":
            seed = command.seed or 0
            self.database.reset(seed)
            self.clear_controls()
        else:
            if command.action == "HARDEN_DOCUMENT_OWNERSHIP":
                self.ownership_hardened = True
            elif command.action == "BLOCK_ACTOR" and command.principal_ref:
                self.blocked_principals.add(command.principal_ref)
            elif command.action == "RATE_LIMIT_ACTOR" and command.principal_ref:
                self.rate_limited_principals.add(command.principal_ref)
            else:
                raise ValueError("control action requires a valid principal reference")
            self.state_version += 1
            self._persist()
        return ControlAcknowledgement(
            command_id=command.command_id,
            run_id=command.run_id,
            episode_id=command.episode_id,
            status="applied",
            state_version=self.state_version,
            controls={
                "ownership_hardened": self.ownership_hardened,
                "blocked_principals": sorted(self.blocked_principals),
                "rate_limited_principals": sorted(self.rate_limited_principals),
            },
            fixture_seed=self.database.fixture_seed,
            completed_at=utc_now(),
        )


async def watch_control_directory(
    runtime: TargetRuntime,
    inbox: Path,
    results: Path,
    poll_seconds: float = 0.1,
) -> None:
    """Consume each typed command once and atomically publish its acknowledgement."""
    inbox.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    while True:
        for path in sorted(inbox.glob("*.json")):
            acknowledgement_path = results / f"{path.stem}.json"
            if acknowledgement_path.exists():
                continue
            try:
                try:
                    command = ControlCommand.model_validate(read_json(path))
                    acknowledgement = runtime.apply(command)
                except (ValidationError, ValueError):
                    raw_id = UUID(path.stem)
                    acknowledgement = ControlAcknowledgement(
                        command_id=raw_id,
                        status="rejected",
                        state_version=runtime.state_version,
                        controls={},
                        fixture_seed=runtime.database.fixture_seed,
                        completed_at=utc_now(),
                        error_code="invalid_control",
                    )
                write_json_atomic(acknowledgement_path, acknowledgement.model_dump(mode="json"))
            except Exception as exc:
                try:
                    raw_id = UUID(path.stem)
                    failed = ControlAcknowledgement(
                        command_id=raw_id,
                        status="failed",
                        state_version=runtime.state_version,
                        controls={},
                        fixture_seed=runtime.database.fixture_seed,
                        completed_at=utc_now(),
                        error_code=type(exc).__name__,
                    )
                    write_json_atomic(results / f"{raw_id}.json", failed.model_dump(mode="json"))
                except Exception:
                    pass
        await asyncio.sleep(poll_seconds)
