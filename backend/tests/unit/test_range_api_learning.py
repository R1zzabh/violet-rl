"""Validation tests for the current real-range training API contract."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from violet.api.range import DecisionSubmission, TrainingRunRequest, TrainingUpdate
from violet.range.contracts import (
    AgentDecision,
    PolicySample,
    RangeObservation,
    TransitionSubmission,
)
from violet.range.observations import RED_ACTION_CATALOG_V2, encode_observation


def test_training_is_joint_and_research_minimums_are_enforced() -> None:
    with pytest.raises(ValidationError, match="together"):
        TrainingRunRequest(role="red", mode="train")
    with pytest.raises(ValidationError, match="10,000"):
        TrainingRunRequest(role="joint", run_kind="research", mode="train", episodes=9999)
    with pytest.raises(ValidationError, match="100 paired"):
        TrainingRunRequest(
            role="joint",
            run_kind="research",
            mode="evaluate",
            episodes=99,
            checkpoint_run_id=uuid4(),
        )
    teaching = TrainingRunRequest(role="joint", mode="train", episodes=1)
    assert teaching.run_kind == "teaching"


def test_joint_optimizer_updates_identify_the_role_that_was_updated() -> None:
    red_update = TrainingUpdate(
        worker_id="trainer-1",
        role="red",
        result={"update_index": 1, "global_step": 2, "metrics": {"loss": 0.25}},
    )
    blue_update = TrainingUpdate(
        worker_id="trainer-1",
        role="blue",
        result={"update_index": 1, "global_step": 2, "metrics": {"loss": 0.5}},
    )
    assert red_update.role == "red"
    assert blue_update.role == "blue"


def test_decision_requires_policy_sample_to_match_observation_and_action() -> None:
    run_id, episode_id = uuid4(), uuid4()
    action_count = len(RED_ACTION_CATALOG_V2)
    observation = RangeObservation(
        schema_id="red.range_observation.v2",
        run_id=run_id,
        episode_id=episode_id,
        step_id=0,
        remaining_requests=40,
        legal_action_mask=tuple(index == 9 for index in range(action_count)),
    )
    sample = PolicySample(
        encoding_schema_id="violet.range.features.v2",
        encoded_observation=encode_observation(observation),
        action_distribution=tuple(1.0 if index == 9 else 0.0 for index in range(action_count)),
        selected_action_index=9,
        selected_action_probability=1.0,
        old_log_probability=0.0,
        value_estimate=0.0,
        entropy=0.0,
        policy_version=0,
        mode="train",
    )
    decision = AgentDecision(
        decision_id=uuid4(),
        run_id=run_id,
        episode_id=episode_id,
        step_id=0,
        role="red",
        policy_id="red-range-test",
        observation_schema_id="red.range_observation.v2",
        legal_action_mask=observation.legal_action_mask,
        selected_action={"operation": "WAIT"},
        inference_ms=0,
        policy_sample=sample,
    )
    submission = DecisionSubmission(decision=decision, observation=observation)
    assert submission.decision.policy_sample == sample


def test_transition_cannot_be_both_terminated_and_truncated() -> None:
    run_id, episode_id = uuid4(), uuid4()
    observation = RangeObservation(
        schema_id="red.range_observation.v2",
        run_id=run_id,
        episode_id=episode_id,
        step_id=1,
        remaining_requests=39,
        legal_action_mask=(True,),
    )
    with pytest.raises(ValidationError, match="both"):
        TransitionSubmission(
            decision_id=uuid4(),
            next_observation=observation,
            next_value=0,
            terminated=True,
            truncated=True,
            mode="train",
        )
