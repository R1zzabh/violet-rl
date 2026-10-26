from uuid import uuid4

from violet.range.observations import (
    build_blue_observation,
    build_red_observation,
    encode_observation,
)


def test_blue_observation_excludes_verifier_truth_and_secrets() -> None:
    run_id, episode_id, target_event_id = uuid4(), uuid4(), uuid4()
    events = [
        {
            "event_type": "target_evidence_ingested",
            "payload": {
                "target_event_id": str(target_event_id),
                "actor_ref": "alice",
                "status": 200,
                "canary": "hidden",
                "fixture_owner_ref": "bob",
                "session_handle": "secret",
                "session_alias": "session-opaque",
                "nested": {
                    "authorization": "secret",
                    "route_id": "documents",
                    "expected_canary": "another hidden value",
                    "fixture_owner": "bob",
                    "verifier_status": "confirmed",
                },
            },
        },
        {
            "event_type": "executor_result",
            "payload": {"finding": {"status": "confirmed"}},
        },
    ]
    observation = build_blue_observation(
        run_id=run_id,
        episode_id=episode_id,
        step_id=2,
        events=events,
        remaining_requests=38,
    )
    encoded = str(observation.model_dump(mode="json"))
    assert observation.schema_id == "blue.range_observation.v2"
    assert observation.source_event_ids == (target_event_id,)
    assert "hidden" not in encoded
    assert "secret" not in encoded
    assert "confirmed" not in encoded
    assert "session-opaque" not in encoded
    assert observation.recent_events[0]["nested"] == {"route_id": "documents"}


def test_red_retains_a_session_alias_when_recent_history_is_long() -> None:
    run_id, episode_id = uuid4(), uuid4()
    events = [
        {
            "event_type": "executor_result",
            "payload": {"result": {"principal_ref": "alice", "session_alias": "session-old"}},
        },
        *[
            {"event_type": "action_submitted", "payload": {"action": {"operation": "WAIT"}}}
            for _ in range(80)
        ],
    ]
    observation = build_red_observation(
        run_id=run_id,
        episode_id=episode_id,
        step_id=10,
        events=events,
        remaining_requests=10,
    )
    assert "session-old" in str(observation.recent_events)
    assert observation.legal_action_mask[6]


def test_red_observation_and_encoder_are_deterministic_and_fixed_width() -> None:
    run_id, episode_id = uuid4(), uuid4()
    events = [
        {
            "event_type": "action_submitted",
            "payload": {"action": {"operation": "DISCOVER_ROUTE", "route_candidate_id": "login"}},
        },
        {"event_type": "executor_result", "payload": {"http_status": 200}},
    ]
    first = build_red_observation(
        run_id=run_id,
        episode_id=episode_id,
        step_id=3,
        events=events,
        remaining_requests=37,
        target_status="available",
    )
    second = build_red_observation(
        run_id=run_id,
        episode_id=episode_id,
        step_id=3,
        events=events,
        remaining_requests=37,
        target_status="available",
    )
    assert first == second
    assert first.discovered_routes == ("login",)
    assert encode_observation(first) == encode_observation(second)
    assert len(encode_observation(first)) == 19 + 10


def test_masks_enforce_sessions_budgets_and_current_controls() -> None:
    from violet.range.observations import BLUE_ACTION_CATALOG_V2, RED_ACTION_CATALOG_V2

    run_id, episode_id = uuid4(), uuid4()
    red = build_red_observation(
        run_id=run_id, episode_id=episode_id, step_id=0, events=[], remaining_requests=40
    )
    assert len(red.legal_action_mask) == len(RED_ACTION_CATALOG_V2)
    assert not red.legal_action_mask[RED_ACTION_CATALOG_V2.index("CHECK_DOCUMENT_ACCESS:doc_bob")]
    logged_in = build_red_observation(
        run_id=run_id,
        episode_id=episode_id,
        step_id=1,
        events=[
            {
                "event_type": "executor_result",
                "payload": {"result": {"session_alias": "session-opaque"}},
            }
        ],
        remaining_requests=35,
    )
    assert "session-opaque" in str(logged_in.recent_events)
    assert logged_in.legal_action_mask[RED_ACTION_CATALOG_V2.index("CHECK_DOCUMENT_ACCESS:doc_bob")]
    blue = build_blue_observation(
        run_id=run_id, episode_id=episode_id, step_id=0, events=[], remaining_requests=40
    )
    assert len(blue.legal_action_mask) == len(BLUE_ACTION_CATALOG_V2)
    assert not blue.legal_action_mask[BLUE_ACTION_CATALOG_V2.index("BLOCK_ACTOR:alice")]
    assert not blue.legal_action_mask[BLUE_ACTION_CATALOG_V2.index("RATE_LIMIT_ACTOR:alice")]
    assert blue.legal_action_mask[BLUE_ACTION_CATALOG_V2.index("MONITOR")]
