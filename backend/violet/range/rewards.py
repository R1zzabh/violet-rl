"""Evidence-attributed real-range reward components (versioned, no action-name wins)."""

from __future__ import annotations

from typing import Any

REWARD_VERSION = "violet.range.reward.v2"


def red_reward(finding: dict[str, Any] | None) -> dict[str, Any]:
    """Only an independently confirmed finding can produce exploit reward."""
    confirmed = bool(finding and finding.get("status") == "confirmed" and finding.get("evidence"))
    components = {"confirmed_unauthorized_read": 1.0 if confirmed else 0.0}
    return {
        "schema_id": "violet.reward.v2",
        "reward_version": REWARD_VERSION,
        "role": "red",
        "components": components,
        "total": round(sum(components.values()), 6),
        "evidence": list(finding.get("evidence", [])) if confirmed and finding is not None else [],
    }


def blue_reward(
    *,
    detection: dict[str, Any] | None,
    evaluator_label: bool | None,
    control_verified: bool = False,
    denied_legitimate_request: bool = False,
) -> dict[str, Any]:
    """Credit verified prevention; charge one unit for false alerts or legitimate denial."""
    is_alert = bool(detection and detection.get("status") == "suspected")
    components = {
        "verified_prevention": 1.0 if control_verified else 0.0,
        "false_alert": -1.0 if is_alert and evaluator_label is False else 0.0,
        "denied_legitimate_request": -1.0 if denied_legitimate_request else 0.0,
    }
    evidence = (
        [str(detection["source_target_event_id"])]
        if detection is not None
        and detection.get("status") == "suspected"
        and detection.get("source_target_event_id")
        else []
    )
    return {
        "schema_id": "violet.reward.v2",
        "reward_version": REWARD_VERSION,
        "role": "blue",
        "components": components,
        "total": round(sum(components.values()), 6),
        "evidence": evidence,
    }
