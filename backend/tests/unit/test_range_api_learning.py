"""Validation tests for the current real-range training API contract."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from pydantic import ValidationError

from violet.api import range as range_api
from violet.api.range import (
    DecisionSubmission,
    ExposureExperimentStartRequest,
    LearningReportStartRequest,
    TrainingRunRequest,
    TrainingUpdate,
)
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


@pytest.mark.parametrize("current_schema", [True, False])
def test_continuation_accepts_current_reward_contract_and_rejects_unknown(
    monkeypatch, current_schema: bool
) -> None:
    from fastapi import HTTPException

    source_id = uuid4()
    source = {
        "training_run_id": str(source_id),
        "config": {"scenario_id": "sqli_lab_v1"},
    }
    metadata = {
        "reward_schema_id": range_api.SQLI_REWARD_SCHEMA if current_schema else "sqli.reward.unknown",
        "checkpoint_sha256": "a" * 64,
        "policy_versions": {"red": 13, "blue": 13},
        "interaction_protocol_schema_id": range_api.SQLI_INTERACTION_PROTOCOL_SCHEMA,
    }

    class Repository:
        def get_training_run(self, training_run_id):
            assert training_run_id == str(source_id)
            return source

        def create_training_run(self, training_run_id, role, mode, config):
            return {"training_run_id": training_run_id, "role": role, "mode": mode, "config": config}

    monkeypatch.setattr(range_api, "_repository", lambda: Repository())
    monkeypatch.setattr(range_api, "_checkpoint_record_metadata", lambda record: metadata)
    request = TrainingRunRequest(
        role="joint", mode="train", scenario_id="sqli_lab_v1",
        episodes=100, max_steps=32, opponent="none", checkpoint_run_id=source_id,
    )
    if not current_schema:
        with pytest.raises(HTTPException) as caught:
            asyncio.run(range_api.create_training_run(request))
        assert caught.value.status_code == 422
        return
    created = asyncio.run(range_api.create_training_run(request))
    assert created["config"]["source_checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert created["config"]["source_policy_versions"] == {"red": 13, "blue": 13}
    assert created["config"]["reward_schema_id"] == range_api.SQLI_REWARD_SCHEMA


def test_sqli_classroom_campaigns_require_40_episodes_and_40_rounds() -> None:
    with pytest.raises(ValidationError, match="exactly 40"):
        TrainingRunRequest(
            role="joint",
            mode="train",
            scenario_id="sqli_lab_v1",
            campaign_label="live_demo_v1",
            episodes=39,
            max_steps=40,
            opponent="none",
        )
    campaign = TrainingRunRequest(
        role="joint",
        mode="train",
        scenario_id="sqli_lab_v1",
        campaign_label="live_demo_v1",
        episodes=40,
        max_steps=40,
        opponent="none",
    )
    assert campaign.episodes == campaign.max_steps == 40


def test_balanced_pilot_requires_fresh_50_episode_runs_and_unique_held_out_seeds() -> None:
    campaign_id = uuid4()
    training = TrainingRunRequest(
        role="joint",
        mode="train",
        scenario_id="sqli_lab_v1",
        episodes=50,
        max_steps=32,
        seed=17,
        opponent="none",
        campaign_label="balanced_pilot_v1",
        pilot_campaign_id=campaign_id,
    )
    assert training.episodes == 50
    assert training.checkpoint_run_id is None

    with pytest.raises(ValidationError, match="50 episodes"):
        TrainingRunRequest(
            role="joint",
            mode="train",
            scenario_id="sqli_lab_v1",
            episodes=49,
            max_steps=32,
            seed=17,
            opponent="none",
            campaign_label="balanced_pilot_v1",
            pilot_campaign_id=campaign_id,
        )


def test_pilot_status_pools_held_out_metrics_and_applies_the_balance_threshold(monkeypatch) -> None:
    campaign_id = uuid4()
    training_runs = []
    evaluation_runs = []
    for seed in (17, 18, 19):
        training_runs.append(
            {
                "training_run_id": f"train-{seed}",
                "role": "joint",
                "mode": "train",
                "status": "completed",
                "stop_requested": False,
                "created_at": "2026-10-26T10:00:00+00:00",
                "config": {
                    "pilot_campaign_id": str(campaign_id),
                    "campaign_label": "balanced_pilot_v1",
                    "seed": seed,
                    "episodes": 50,
                    "pilot_started_at": "2026-10-26T10:00:00+00:00",
                    "pilot_training_deadline": "2026-10-26T13:00:00+00:00",
                    "pilot_evaluation_deadline": "2026-10-26T14:00:00+00:00",
                },
                "result": {
                    "completed_training_episodes": 50,
                    "episode_summary_count": 50,
                    "policy_versions": {"red": 51, "blue": 51},
                },
            }
        )


        evaluation_runs.append(
            {
                "training_run_id": f"eval-{seed}",
                "role": "joint",
                "mode": "evaluate",
                "status": "completed",
                "config": {
                    "pilot_campaign_id": str(campaign_id),
                    "campaign_label": "balanced_pilot_evaluation_v1",
                    "seed": seed,
                    "episodes": 15,
                    "pilot_evaluation_seeds": list(range(1_000_000_000 + seed * 1_000, 1_000_000_015 + seed * 1_000)),
                },
                "result": {
                    "evaluation": {
                        "requested_matches": 15,
                        "completed_matches": 15,
                        "attempted_matches": 15,
                        "red_wins": 4,
                        "blue_wins": 4,
                        "undecided_matches": 7,
                        "verified_disclosures": 3,
                        "verified_exposure_rate": 0.2,
                        "verified_prevention_rate": 0.8,
                        "legitimate_service_availability": 1.0,
                    }
                },
            }
        )
    monkeypatch.setattr(range_api, "_all_training_runs", lambda: training_runs + evaluation_runs)
    monkeypatch.setattr(
        range_api,
        "_checkpoint_record_metadata",
        lambda record, allow_running=False: {
            "checkpoint_sha256": "a" * 64,
            "completed_training_episodes": 50,
            "policy_versions": {"red": 51, "blue": 51},
        }
        if record["mode"] == "train"
        else None,
    )

    status = range_api._pilot_status(campaign_id)
    assert status["status"] == "completed"
    assert status["training"]["completed_updates"] == 150
    assert status["evaluation"]["completed_matches"] == 45
    assert status["evaluation"]["incomplete_matches"] == 0
    assert status["evaluation"]["red_wins"] == 12
    assert status["evaluation"]["blue_wins"] == 12
    assert status["evaluation"]["verified_disclosures"] == 9
    assert status["evaluation"]["balanced"] is True
    assert status["evaluation"]["balance_verdict"] == "balanced"
    assert len(status["evaluation"]["red_win_rate_wilson_95"]) == 2

    evaluation = TrainingRunRequest(
        role="joint",
        mode="evaluate",
        scenario_id="sqli_lab_v1",
        episodes=15,
        max_steps=32,
        seed=17,
        opponent="none",
        checkpoint_run_id=uuid4(),
        campaign_label="balanced_pilot_evaluation_v1",
        pilot_campaign_id=campaign_id,
        pilot_evaluation_seeds=list(range(1_000_017_000, 1_000_017_015)),
    )
    assert len(evaluation.pilot_evaluation_seeds or []) == 15
    with pytest.raises(ValidationError, match="15 unique held-out seeds"):
        TrainingRunRequest(
            role="joint",
            mode="evaluate",
            scenario_id="sqli_lab_v1",
            episodes=15,
            max_steps=32,
            seed=17,
            opponent="none",
            checkpoint_run_id=uuid4(),
            campaign_label="balanced_pilot_evaluation_v1",
            pilot_campaign_id=campaign_id,
            pilot_evaluation_seeds=[1] * 15,
        )


def test_learning_report_uses_matched_outcomes_and_labels_uncovered_families(monkeypatch) -> None:
    evaluation_id = uuid4()
    seeds = [1_700_000_001, 1_700_000_002]

    def evaluation_record(variant: str, success: list[float]) -> dict:
        return {
            "training_run_id": f"eval-{variant}",
            "role": "joint",
            "mode": "evaluate",
            "status": "completed",
            "config": {
                "checkpoint_run_id": "source-run",
                "checkpoint_variant": variant,
                "campaign_label": "learning_evaluation_v1",
                "learning_evaluation_id": str(evaluation_id),
                "learning_evaluation_seeds": seeds,
            },
            "result": {
                "evaluation": {
                    "checkpoint_sha256": "a" * 64 if variant == "initial" else "b" * 64,
                    "policy_versions": {"red": 0 if variant == "initial" else 2, "blue": 0 if variant == "initial" else 2},
                    "requested_matches": 2,
                    "completed_matches": 2,
                    "incomplete_matches": 0,
                    "verified_disclosures": 1,
                    "verified_exposure_rate": sum(success) / len(success),
                    "verified_prevention_rate": 0.5,
                    "legitimate_service_availability": 1.0,
                    "pair_frozen": True,
                },
                "episodes": [
                    {
                        "seed": seed,
                        "status": "completed",
                        "verified_exposure": value,
                        "verified_prevention_rate": 0.5,
                        "legitimate_query_availability": 1.0,
                    }
                    for seed, value in zip(seeds, success, strict=True)
                ],
            },
        }

    monkeypatch.setattr(
        range_api,
        "_all_training_runs",
        lambda: [evaluation_record("initial", [0.0, 1.0]), evaluation_record("final", [1.0, 1.0])],
    )
    report = range_api._learning_report(evaluation_id)
    assert report["status"] == "completed"
    assert report["matched_seed_schedule"] == seeds
    change = report["paired_changes"]["verified_red_success_rate"]
    assert change["matched_fixtures"] == 2
    assert change["mean_change"] == 0.5
    assert report["initial"]["frozen"] is True
    assert report["family_results"]["unauthorized_record_access"]["status"] == "not_evaluated_in_sqli_scenario"
    assert report["family_results"]["permission_or_workflow_abuse"]["status"] == "not_available_in_current_target"


@pytest.mark.parametrize("variants", [("initial", "final"), ("initial",), ("final",)])
def test_learning_report_endpoint_handles_queued_evaluations_without_results(monkeypatch, variants) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    evaluation_id = uuid4()
    records = [
        {
            "training_run_id": str(uuid4()),
            "status": "queued",
            "config": {
                "learning_evaluation_id": str(evaluation_id),
                "checkpoint_run_id": "source-run",
                "checkpoint_variant": variant,
                "learning_evaluation_seeds": [1_700_000_001],
            },
            "result": None,
        }
        for variant in variants
    ]
    monkeypatch.setattr(range_api, "_all_training_runs", lambda: records)
    app = FastAPI()
    app.include_router(range_api.router, prefix="/range")
    with TestClient(app) as client:
        response = client.get(f"/range/learning-reports/{evaluation_id}")
    assert response.status_code == 200
    report = response.json()
    assert report["status"] == "running"
    assert report["family_results"]["sql_injection"]["initial"] == {}
    assert report["family_results"]["sql_injection"]["later"] == {}
    for variant, key in [("initial", "initial"), ("final", "later")]:
        if variant in variants:
            assert report[key]["status"] == "queued"
            assert report[key]["completed_matches"] is None
        else:
            assert report[key] is None


def test_exposure_report_measures_realized_fixtures_and_matched_test_schedule(monkeypatch) -> None:
    experiment_id = uuid4()
    training_seeds = [1701, 1702]
    test_seeds = [1_800_000_001, 1_800_000_002]

    def branch(name: str, probability: float, campaign: str) -> dict:
        evaluation = campaign == "exposure_experiment_evaluation_v1"
        return {
            "training_run_id": f"{campaign}-{name}",
            "role": "joint",
            "mode": "evaluate" if evaluation else "train",
            "status": "completed",
            "config": {
                "experiment_id": str(experiment_id),
                "experiment_branch": name,
                "campaign_label": campaign,
                "checkpoint_run_id": "source-run" if not evaluation else None,
                "episodes": 2,
                "experiment_episode_seeds": test_seeds if evaluation else training_seeds,
                "vulnerable_configuration_probability": 0.5 if evaluation else probability,
            },
            "result": {
                "completed_training_episodes": 2,
                "checkpoint_sha256": "c" * 64,
                "policy_versions": {"red": 2, "blue": 2},
                "episodes": [
                    {
                        "status": "completed",
                        "initial_server_configuration": {"vulnerable_configuration": index == 1},
                        "verified_exposure": index == 1,
                        "verified_prevention_rate": 1.0,
                        "legitimate_query_availability": 1.0,
                    }
                    for index in range(2)
                ],
                "evaluation": {
                    "requested_matches": 2,
                    "completed_matches": 2,
                    "verified_disclosures": 1,
                    "verified_exposure_rate": 0.5,
                    "verified_prevention_rate": 0.5,
                    "legitimate_service_availability": 1.0,
                } if evaluation else None,
            },
        }

    monkeypatch.setattr(
        range_api,
        "_all_training_runs",
        lambda: [
            branch("control", 0.35, "exposure_experiment_v1"),
            branch("increased", 0.85, "exposure_experiment_v1"),
            branch("control", 0.5, "exposure_experiment_evaluation_v1"),
            branch("increased", 0.5, "exposure_experiment_evaluation_v1"),
        ],
    )
    report = range_api._exposure_experiment_report(experiment_id)
    assert report["status"] == "completed"
    assert report["equal_training_budget"] is True
    assert report["matched_training_seeds"] == training_seeds
    assert report["training"]["increased"]["realized_vulnerable_configuration_frequency"] == 0.5
    assert report["held_out_evaluation"]["test_distribution_unchanged"] is True
    assert report["held_out_evaluation"]["same_seed_schedule"] is True


def test_learning_and_exposure_request_contracts_are_bounded() -> None:
    learning = LearningReportStartRequest(
        training_run_id=uuid4(), evaluation_id=uuid4(), matches=20
    )
    assert learning.matches == 20
    exposure = ExposureExperimentStartRequest(
        source_training_run_id=uuid4(),
        experiment_id=uuid4(),
        episodes_per_branch=30,
        control_vulnerability_probability=0.35,
        increased_vulnerability_probability=0.85,
        evaluation_matches=20,
    )
    assert exposure.increased_vulnerability_probability > exposure.control_vulnerability_probability
    with pytest.raises(ValidationError):
        ExposureExperimentStartRequest(
            source_training_run_id=uuid4(),
            experiment_id=uuid4(),
            episodes_per_branch=501,
        )


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
