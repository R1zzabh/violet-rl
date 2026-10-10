"""Portable, redacted export of a persisted real-range run."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from violet.persistence.repository import EventRepository

SENSITIVE_KEY_PARTS = (
    "password",
    "cookie",
    "token",
    "authorization",
    "credential",
    "session_handle",
)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            result[str(key)] = (
                "[REDACTED]"
                if any(part in normalized for part in SENSITIVE_KEY_PARTS)
                else _redact(item)
            )
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _verify_event_chain(events: list[dict[str, Any]]) -> None:
    previous_hash = ""
    for expected_seq, event in enumerate(events, start=1):
        if event["seq"] != expected_seq or event["previous_hash"] != previous_hash:
            raise ValueError("run event sequence or previous hash is invalid")
        occurred_at = datetime.fromisoformat(event["occurred_at"])
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=UTC)
        envelope = {
            "run_id": event["run_id"],
            "episode_id": event["episode_id"],
            "seq": event["seq"],
            "event_type": event["event_type"],
            "occurred_at": occurred_at.isoformat(),
            "payload": event["payload"],
        }
        canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
        actual_hash = hashlib.sha256(f"{previous_hash}{canonical}".encode()).hexdigest()
        if event["event_hash"] != actual_hash:
            raise ValueError(f"event hash mismatch at sequence {expected_seq}")
        previous_hash = actual_hash


def export_run(repository: EventRepository, run_id: str, output_root: Path) -> Path:
    """Write events and a checksum manifest under ``output_root/<run_id>``."""
    events: list[dict[str, Any]] = []
    after_seq = 0
    while True:
        page = repository.events(run_id, after_seq=after_seq, limit=1000)
        if not page:
            break
        events.extend(page)
        after_seq = page[-1]["seq"]
        if len(page) < 1000:
            break
    if not events:
        raise ValueError("run has no persisted events")
    _verify_event_chain(events)

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    with events_path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(_redact(event), sort_keys=True, separators=(",", ":")))
            handle.write("\n")

    digest = hashlib.sha256(events_path.read_bytes()).hexdigest()
    manifest = {
        "schema_id": "violet.range_export.v1",
        "run_id": run_id,
        "exported_at": datetime.now(UTC).isoformat(),
        "event_count": len(events),
        "first_seq": events[0]["seq"],
        "last_seq": events[-1]["seq"],
        "source_event_chain_head": events[-1]["event_hash"],
        "files": {
            "events.jsonl": {
                "size_bytes": events_path.stat().st_size,
                "sha256": digest,
            }
        },
        "redactions": list(SENSITIVE_KEY_PARTS),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir
