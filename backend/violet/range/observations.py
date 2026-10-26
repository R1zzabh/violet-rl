"""Role-specific observations assembled from persisted, policy-visible evidence."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Literal, cast
from uuid import UUID

from violet.range.contracts import RangeObservation

_SECRET_OR_TRUTH_KEYS = {
    "canary",
    "fixture_owner_ref",
    "owner_ref",
    "finding",
    "ground_truth",
    "verifier_result",
    "password",
    "session_handle",
    "cookie",
    "token",
    "authorization",
    "credential",
}
_HIDDEN_KEY_PARTS = (
    "canary",
    "owner",
    "finding",
    "verifier",
    "ground_truth",
    "session_handle",
    "password",
    "cookie",
    "token",
    "authorization",
    "credential",
    "secret",
)

RED_ACTION_CATALOG_V2 = (
    "DISCOVER_ROUTE:home",
    "DISCOVER_ROUTE:login",
    "DISCOVER_ROUTE:document",
    "AUTH_TEST:synthetic_alice",
    "AUTH_TEST:synthetic_bob",
    "AUTH_TEST:synthetic_invalid",
    "CHECK_DOCUMENT_ACCESS:doc_alice",
    "CHECK_DOCUMENT_ACCESS:doc_bob",
    "CHECK_TARGET_HEALTH",
    "WAIT",
)
BLUE_ACTION_CATALOG_V2 = (
    "MONITOR",
    "HARDEN_DOCUMENT_OWNERSHIP",
    "RATE_LIMIT_ACTOR:alice",
    "RATE_LIMIT_ACTOR:bob",
    "BLOCK_ACTOR:alice",
    "BLOCK_ACTOR:bob",
)
FEATURE_ENCODING_SCHEMA_ID = "violet.range.features.v2"


def _blue_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _blue_safe(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") not in _SECRET_OR_TRUTH_KEYS
            and str(key).lower().replace("-", "_") != "session_alias"
            and not any(part in str(key).lower().replace("-", "_") for part in _HIDDEN_KEY_PARTS)
        }
    if isinstance(value, list):
        return [_blue_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_blue_safe(item) for item in value)
    return value


def _red_safe(value: Any) -> Any:
    """Keep executor-owned aliases visible to Red while excluding verifier truth."""
    if isinstance(value, dict):
        return {
            str(key): _red_safe(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") in {"session_alias"}
            or (
                str(key).lower().replace("-", "_") not in _SECRET_OR_TRUTH_KEYS
                and not any(
                    part in str(key).lower().replace("-", "_") for part in _HIDDEN_KEY_PARTS
                )
            )
        }
    if isinstance(value, list):
        return [_red_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_red_safe(item) for item in value)
    return value


def build_blue_observation(
    *,
    run_id: UUID,
    episode_id: UUID,
    step_id: int,
    events: list[dict[str, Any]],
    remaining_requests: int,
    target_status: str = "unknown",
    active_controls: dict[str, object] | None = None,
    legal_action_mask: tuple[bool, ...] | None = None,
) -> RangeObservation:
    """Build Blue input only from the telemetry stream, sanitizing recursively."""
    safe_events = tuple(
        _blue_safe(event.get("payload", {}))
        for event in events[-64:]
        if event.get("event_type") in {"target_evidence_ingested", "detection_suspected"}
    )
    observed_controls: dict[str, object] = {}
    suspicious_access_observed = any(
        event.get("event_type") == "target_evidence_ingested"
        and isinstance(event.get("payload"), dict)
        and isinstance(event["payload"].get("evidence"), dict)
        and event["payload"]["evidence"].get("route_id") == "document"
        and event["payload"]["evidence"].get("resource_ref") == "doc_bob"
        and event["payload"]["evidence"].get("principal_ref") == "alice"
        and event["payload"]["evidence"].get("http_status") == 200
        for event in events
    )
    for event in events:
        if event.get("event_type") != "target_control_acknowledged":
            continue
        payload = event.get("payload", {})
        if isinstance(payload, dict):
            controls = payload.get("controls")
            if isinstance(controls, dict):
                observed_controls.update(controls)
            if isinstance(payload.get("state_version"), int):
                observed_controls["state_version"] = payload["state_version"]
    ids: list[UUID] = []
    for event in events[-64:]:
        payload = event.get("payload", {})
        event_id = payload.get("target_event_id") if isinstance(payload, dict) else None
        if event.get("event_type") == "target_evidence_ingested" and event_id:
            try:
                ids.append(UUID(str(event_id)))
            except ValueError:
                continue
    controls_state = _blue_safe(active_controls or observed_controls)
    control_values = controls_state if isinstance(controls_state, dict) else {}
    certified_mask = (
        True,
        suspicious_access_observed and not bool(control_values.get("ownership_hardened")),
        False,  # Rate limiting has no independent effect/availability verifier yet.
        False,
        False,  # Blocking has no independent effect/availability verifier yet.
        False,
    )
    # An injected mask may further restrict legal actions, but cannot enable
    # controls whose security and availability effects have not been verified.
    mask = tuple(
        certified and (legal_action_mask[index] if legal_action_mask else True)
        for index, certified in enumerate(certified_mask)
    )
    return RangeObservation(
        schema_id="blue.range_observation.v2",
        run_id=run_id,
        episode_id=episode_id,
        step_id=step_id,
        target_status=cast(
            "Literal['available', 'unavailable', 'unknown']",
            target_status if target_status in {"available", "unavailable"} else "unknown",
        ),
        remaining_requests=remaining_requests,
        recent_events=safe_events,
        active_controls=controls_state,
        source_event_ids=tuple(ids),
        legal_action_mask=mask,
    )


def build_red_observation(
    *,
    run_id: UUID,
    episode_id: UUID,
    step_id: int,
    events: list[dict[str, Any]],
    remaining_requests: int,
    target_status: str = "unknown",
    legal_action_mask: tuple[bool, ...] | None = None,
) -> RangeObservation:
    """Build Red input from observed submissions/results, excluding verifier data."""
    recent = []
    latest_session_event: dict[str, Any] | None = None
    discovered: set[Literal["home", "login", "document"]] = set()
    source_ids: list[UUID] = []
    tested: set[str] = set()
    has_session = False
    for event in events:
        payload = event.get("payload", {})
        if event.get("event_type") == "action_submitted" and isinstance(payload, dict):
            action = payload.get("action", {})
            if isinstance(action, dict):
                operation = action.get("operation")
                if operation == "DISCOVER_ROUTE":
                    tested.add(f"DISCOVER_ROUTE:{action.get('route_candidate_id')}")
                elif operation == "AUTH_TEST":
                    tested.add(f"AUTH_TEST:{action.get('credential_ref')}")
                elif operation == "CHECK_DOCUMENT_ACCESS":
                    tested.add(f"CHECK_DOCUMENT_ACCESS:{action.get('document_ref')}")
        if event.get("event_type") == "executor_result" and isinstance(payload, dict):
            result = payload.get("result", {})
            if isinstance(result, dict) and result.get("session_alias"):
                has_session = True
                latest_session_event = _red_safe(payload)
    is_available = target_status != "unavailable" and remaining_requests > 0
    mask = legal_action_mask or tuple(
        [key not in tested and is_available for key in RED_ACTION_CATALOG_V2[:8]]
        + [target_status != "unavailable", True]
    )
    mask_values = list(mask)
    if not has_session:
        mask_values[6] = False
        mask_values[7] = False
    if remaining_requests <= 0:
        mask_values[:-1] = [False] * (len(mask_values) - 1)
    for event in events[-64:]:
        if event.get("event_type") not in {
            "action_submitted",
            "executor_result",
            "target_evidence_ingested",
        }:
            continue
        payload = _red_safe(event.get("payload", {}))
        recent.append(payload)
        evidence_id = payload.get("target_event_id") if isinstance(payload, dict) else None
        if evidence_id:
            with suppress(ValueError):
                source_ids.append(UUID(str(evidence_id)))
        action = payload.get("action", {}) if isinstance(payload, dict) else {}
        route = action.get("route_candidate_id") if isinstance(action, dict) else None
        if route in {"home", "login", "document"}:
            discovered.add(route)
    return RangeObservation(
        schema_id="red.range_observation.v2",
        run_id=run_id,
        episode_id=episode_id,
        step_id=step_id,
        target_status=cast(
            "Literal['available', 'unavailable', 'unknown']",
            target_status if target_status in {"available", "unavailable"} else "unknown",
        ),
        remaining_requests=remaining_requests,
        discovered_routes=tuple(sorted(discovered)),
        recent_events=tuple(
            [*recent[-63:], latest_session_event]
            if latest_session_event is not None
            and not any(
                isinstance(event, dict)
                and isinstance(event.get("result"), dict)
                and event["result"].get("session_alias")
                for event in recent[-64:]
            )
            else recent[-64:]
        ),
        source_event_ids=tuple(source_ids),
        legal_action_mask=tuple(mask_values),
    )


def encode_observation(observation: RangeObservation) -> tuple[float, ...]:
    """Encode policy-visible target state with the stable v2 feature ordering."""
    route_slots = ("home", "login", "document")
    status_counts = [0, 0, 0, 0]
    actors: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            status = value.get("http_status")
            if isinstance(status, int) and 200 <= status <= 599:
                status_counts[min(status // 100 - 2, 3)] += 1
            principal = value.get("principal_ref") or value.get("actor_ref")
            if principal in {"alice", "bob"}:
                actors.add(str(principal))
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(observation.recent_events)
    status_total = max(1, sum(status_counts))
    controls = observation.active_controls
    values = [
        float(observation.target_status == "available"),
        float(observation.target_status == "unavailable"),
        observation.remaining_requests / 40.0,
        observation.step_id / 500.0,
    ]
    values.extend(float(route in observation.discovered_routes) for route in route_slots)
    values.extend(count / status_total for count in status_counts)
    values.extend(float(actor in actors) for actor in ("alice", "bob"))
    blocked = controls.get("blocked_principals", [])
    limited = controls.get("rate_limited_principals", [])
    blocked_set = set(blocked) if isinstance(blocked, list) else set()
    limited_set = set(limited) if isinstance(limited, list) else set()
    values.extend(
        [
            float(controls.get("ownership_hardened") is True),
            float("alice" in blocked_set),
            float("bob" in blocked_set),
            float("alice" in limited_set),
            float("bob" in limited_set),
        ]
    )
    values.append(min(len(observation.recent_events), 64) / 64.0)
    values.extend(float(value) for value in observation.legal_action_mask)
    return tuple(values)
