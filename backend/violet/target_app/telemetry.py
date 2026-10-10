"""Append-only, secret-free target access evidence."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from pathlib import Path


class TargetTelemetry:
    def __init__(self, evidence_path: Path | None):
        self.evidence_path = evidence_path
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, payload: dict[str, Any]) -> str:
        target_event_id = str(uuid4())
        event = {
            "target_event_id": target_event_id,
            "occurred_at": datetime.now(UTC).isoformat(),
            **payload,
        }
        if self.evidence_path is not None:
            encoded = json.dumps(event, separators=(",", ":"), sort_keys=True)
            fd = os.open(self.evidence_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o640)
            try:
                os.write(fd, (encoded + "\n").encode())
                os.fsync(fd)
            finally:
                os.close(fd)
        return target_event_id
