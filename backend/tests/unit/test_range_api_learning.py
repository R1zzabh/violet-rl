"""Schema validation for the persisted training-transition API contract."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from violet.api.range import (
    EpisodeResultRequest,
    LearningTransitionRequest,
    LearningUpdateRequest,
)


def _payload(role: str = "red") -> dict[str, object]:
    size, actions = (7, 8) if role == "red" else (8, 4)
    return {
        "run_id": str(uuid4()),
        "episode_id": str(uuid4()),
        "seed": 42,
        "role": role,
        "schema_id": "violet.range.observation.v2",
        "observation": [0.0] * size,
        "proposed_action": 0,
        "action_mask": [True] * actions,
        "accepted": True,
        "executed_action": 0,
        "reward": 0.0,
        "reward_components": {"no_verified_outcome": 0.0},
        "next_observation": [0.0] * size,
        "next_value": 0.0,
        "terminated": False,
        "truncated": True,
        "policy_version": 0,
        "selected_action_probability": 0.5,
        "policy_entropy": 0.69,
        "mode": "training",
    }


@pytest.mark.parametrize("role", ["red", "blue"])
def test_role_specific_observation_contract_accepts_valid_transition(role: str) -> None:
    assert LearningTransitionRequest.model_validate(_payload(role)).role == role


def test_observation_schema_and_terminal_flags_are_validated() -> None:
    stale = _payload()
    stale["schema_id"] = "violet.range.observation.v1"
    with pytest.raises(ValidationError):
        LearningTransitionRequest.model_validate(stale)
    both_endings = _payload()
    both_endings["terminated"] = True
    with pytest.raises(ValidationError, match="both"):
        LearningTransitionRequest.model_validate(both_endings)


def test_rejected_action_cannot_be_recorded_as_executed() -> None:
    invalid = _payload()
    invalid["accepted"] = False
    with pytest.raises(ValidationError, match="accepted transitions"):
        LearningTransitionRequest.model_validate(invalid)


def test_update_metrics_accept_gradient_and_advantage_diagnostics() -> None:
    request = LearningUpdateRequest(
        run_id=uuid4(),
        episode_id=uuid4(),
        seed=42,
        role="red",
        policy_version=1,
        transition_count=6,
        metrics={
            "loss": 0.1,
            "policy_loss": 0.0,
            "value_loss": 0.2,
            "entropy": 1.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.3,
            "optimizer_steps": 4.0,
            "advantage_mean": 0.0,
            "advantage_std": 1.0,
            "advantage_min": -1.0,
            "advantage_max": 1.0,
        },
    )
    assert request.metrics["gradient_norm"] == 0.3


def test_episode_lifecycle_rejects_inconsistent_failed_run() -> None:
    with pytest.raises(ValidationError, match="failed episode"):
        EpisodeResultRequest(
            run_id=uuid4(),
            episode_id=uuid4(),
            seed=42,
            episode_status="failed",
            run_status="completed",
            summary={},
        )
