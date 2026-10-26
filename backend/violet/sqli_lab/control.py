"""File-delivered, typed controls for the isolated SQLi target."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import ValidationError

from violet.range.spool import read_json, write_json_atomic
from violet.sqli_lab.contracts import SqliControlAcknowledgement, SqliControlCommand

if TYPE_CHECKING:
    from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SqliRuntime:
    def __init__(self) -> None:
        self.parameterized = False
        self.rate_limit_guest = False
        self.isolate_guest_remaining = 0
        self.guest_requests: dict[str, int] = {}
        self.state_version = 0
        self.challenge_secrets: dict[str, tuple[str, float]] = {}

    def set_challenge(self, challenge_id: str, phrase: str, ttl_seconds: float = 900.0) -> None:
        now = time.monotonic()
        for expired_id, (_secret, expires_at) in list(self.challenge_secrets.items()):
            if expires_at <= now:
                self.challenge_secrets.pop(expired_id, None)
        self.challenge_secrets[challenge_id] = (phrase, now + ttl_seconds)

    def get_challenge(self, challenge_id: str) -> str | None:
        item = self.challenge_secrets.get(challenge_id)
        if item is None:
            return None
        phrase, expires_at = item
        if expires_at <= time.monotonic():
            self.challenge_secrets.pop(challenge_id, None)
            return None
        return phrase

    def clear_challenge(self, challenge_id: str) -> None:
        self.challenge_secrets.pop(challenge_id, None)

    def apply(self, command: SqliControlCommand) -> SqliControlAcknowledgement:
        if command.action == "RESET":
            self.parameterized = False
            self.rate_limit_guest = False
            self.isolate_guest_remaining = 0
            self.guest_requests.clear()
        elif command.action == "PARAMETERIZE_QUERY":
            self.parameterized = True
        elif command.action == "RATE_LIMIT_GUEST":
            self.rate_limit_guest = True
        elif command.action == "ISOLATE_GUEST_LOOKUP":
            self.isolate_guest_remaining = 2
        self.state_version += 1
        return SqliControlAcknowledgement(
            command_id=command.command_id,
            run_id=command.run_id,
            episode_id=command.episode_id,
            status="applied",
            state_version=self.state_version,
            controls=self.snapshot(),
            completed_at=utc_now(),
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "parameterized": self.parameterized,
            "rate_limit_guest": self.rate_limit_guest,
            "isolate_guest_remaining": self.isolate_guest_remaining,
            "state_version": self.state_version,
        }

    def check_request(self, actor_ref: str, episode_id: str) -> tuple[int | None, str | None]:
        if actor_ref != "guest":
            return None, None
        count = self.guest_requests.get(episode_id, 0) + 1
        self.guest_requests[episode_id] = count
        if self.isolate_guest_remaining > 0:
            self.isolate_guest_remaining -= 1
            return 503, "temporary guest isolation"
        if self.rate_limit_guest and count > 1:
            return 429, "guest lookup rate limit"
        return None, None


async def watch_controls(runtime: SqliRuntime, inbox: Path, results: Path) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    while True:
        for path in sorted(inbox.glob("*.json")):
            result_path = results / path.name
            if result_path.exists():
                continue
            try:
                command = SqliControlCommand.model_validate(read_json(path))
                acknowledgement = runtime.apply(command)
            except (ValidationError, ValueError):
                try:
                    command_id = UUID(path.stem)
                    acknowledgement = SqliControlAcknowledgement(
                        command_id=command_id,
                        run_id=UUID(int=0),
                        episode_id=UUID(int=0),
                        status="rejected",
                        state_version=runtime.state_version,
                        completed_at=utc_now(),
                        error_code="invalid_control",
                    )
                except ValueError:
                    continue
            write_json_atomic(result_path, acknowledgement.model_dump(mode="json"))
        await asyncio.sleep(0.1)
