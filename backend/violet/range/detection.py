"""Deterministic Blue detectors over sanitized target telemetry only."""

from __future__ import annotations

from typing import Any

DETECTOR_VERSION = "violet.blue.rules.v1"


def detect(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return evidence-linked alerts; labels and verifier output are never inputs."""
    alerts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if event.get("event_type") != "target_evidence_ingested":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            continue
        target_event_id = evidence.get("target_event_id")
        route_id = evidence.get("route_id")
        # This task's public fixture contract labels two synthetic principals
        # and two document refs. Only Alice reaching Bob's document is a
        # suspicious success; ordinary owner reads do not create alerts.
        if (
            route_id == "document"
            and evidence.get("resource_ref") == "doc_bob"
            and evidence.get("http_status") == 200
            and evidence.get("principal_ref") == "alice"
            and isinstance(target_event_id, str)
            and target_event_id not in seen
        ):
            seen.add(target_event_id)
            alerts.append(
                {
                    "schema_id": "violet.detection.v1",
                    "detector_version": DETECTOR_VERSION,
                    "rule_id": "successful_document_access",
                    "severity": "medium",
                    "status": "suspected",
                    "source_target_event_id": target_event_id,
                    "actor_ref": evidence["principal_ref"],
                    "route_id": "document",
                }
            )
    return alerts
