from uuid import uuid4

import pytest

from violet.api.range import DecisionSubmission
from violet.range.contracts import AgentDecision, RangeObservation
from violet.range.observations import RED_ACTION_CATALOG_V2


@pytest.mark.parametrize(
    ("action", "action_index"),
    [
        ({"operation": "DISCOVER_ROUTE", "route_candidate_id": "login"}, 1),
        ({"operation": "AUTH_TEST", "credential_ref": "synthetic_alice"}, 3),
        (
            {
                "operation": "CHECK_DOCUMENT_ACCESS",
                "document_ref": "doc_bob",
                "session_alias": "session-alice-01",
            },
            7,
        ),
        ({"operation": "CHECK_TARGET_HEALTH"}, 8),
        ({"operation": "WAIT"}, 9),
    ],
)
def test_red_decision_catalog_validation_handles_every_action_variant(
    action: dict[str, object], action_index: int
) -> None:
    run_id, episode_id = uuid4(), uuid4()
    mask = tuple(True for _ in RED_ACTION_CATALOG_V2)
    observation = RangeObservation(
        schema_id="red.range_observation.v2",
        run_id=run_id,
        episode_id=episode_id,
        step_id=0,
        remaining_requests=40,
        recent_events=({"session_alias": "session-alice-01"},),
        legal_action_mask=mask,
    )
    decision = AgentDecision(
        decision_id=uuid4(),
        run_id=run_id,
        episode_id=episode_id,
        step_id=0,
        role="red",
        policy_id="range-test-v1",
        observation_schema_id="red.range_observation.v2",
        legal_action_mask=mask,
        selected_action=action,
        inference_ms=0,
    )

    submission = DecisionSubmission(decision=decision, observation=observation)

    assert submission.decision.selected_action == action
    assert submission.decision.legal_action_mask[action_index] is True
