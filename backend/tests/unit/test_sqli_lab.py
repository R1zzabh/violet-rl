from __future__ import annotations

import asyncio
import json
from uuid import UUID, uuid4

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import DBAPIError

from violet.api import range as range_api
from violet.api.range import SqliEventSubmission, _find_sqli_target_event
from violet.sqli_lab.app import create_app
from violet.sqli_lab.contracts import (
    ACTION_SCHEMA,
    BLUE_ACTIONS,
    INTERACTION_PROTOCOL_SCHEMA,
    OBSERVATION_SCHEMA,
    RED_COMPONENTS,
    REWARD_SCHEMA,
    SCENARIO_ID,
    TARGET_VERSION,
    SqliControlCommand,
    SqliRequestAction,
)
from violet.sqli_lab.control import SqliRuntime
from violet.sqli_lab.database import SqliDatabase
from violet.sqli_lab.executor_main import SqliExecutor, validate_target_url
from violet.sqli_lab.learning import FEATURES, SqliPPOPolicy, SqliSample, ppo_settings
from violet.sqli_lab.training import (
    CHECKPOINT_CONTRACT,
    PPO_OPTIMIZATION_PASSES,
    SqliEpisodeRunner,
    _episode_winner,
    _intentional_guest_isolation,
    _red_reward_components,
    _transition_batch,
    _validate_checkpoint_contract,
    run_sqli_job,
)

ATTACK = SqliRequestAction(
    field="handle",
    base_term="rhea",
    close_quote=True,
    connector="OR",
    predicate="tautology",
    comment=True,
)
LEGITIMATE = SqliRequestAction(
    field="handle",
    base_term="rhea",
    close_quote=False,
    connector="none",
    predicate="none",
    comment=False,
)


def test_ppo_event_budget_fits_full_32_transition_update_diagnostics():
    diagnostics = [
        {
            "decision_id": str(uuid4()),
            "step_id": step_id,
            "action_indices": [0, 1, 0, 1, 0, 1],
            "old_component_probabilities": [[0.12345678901234567] * width for width in (4, 5, 2, 3, 3, 2)],
            "new_component_probabilities": [[0.7654321098765432] * width for width in (4, 5, 2, 3, 3, 2)],
            "old_selected_component_probabilities": [0.12345678901234567] * 6,
            "new_selected_component_probabilities": [0.7654321098765432] * 6,
            "old_joint_log_probability": -5.123456789012345,
            "new_joint_log_probability": -4.987654321098765,
            "old_joint_probability": 0.0054321098765432,
            "new_joint_probability": 0.0067890123456789,
            "critic_value": 0.12345678901234567,
            "next_value": 0.23456789012345678,
            "reward": 0.1,
            "advantage": 0.2,
            "normalized_advantage": 0.3,
            "return_target": 0.4,
            "terminated": False,
            "truncated": False,
        }
        for step_id in range(32)
    ]
    common = {
        "event_id": str(uuid4()),
        "run_id": str(uuid4()),
        "episode_id": str(uuid4()),
        "seed": 17,
        "payload": {"transition_diagnostics": diagnostics},
    }

    SqliEventSubmission.model_validate({**common, "event_type": "sqli_ppo_update"})

    with pytest.raises(ValueError, match="48 KB"):
        SqliEventSubmission.model_validate({**common, "event_type": "sqli_policy_probe"})


def test_factorized_red_policy_emits_valid_composed_components_and_is_state_dependent():
    policy = SqliPPOPolicy("red", seed=19)
    first = np.zeros(len(FEATURES["red"]), dtype=np.float32)
    second = first.copy()
    second[2] = 0.75
    before = policy.probabilities(first)
    changed = policy.probabilities(second)
    assert any(not np.allclose(left, right) for left, right in zip(before, changed, strict=True))

    action_data, sample = policy.act(second, stochastic=True)
    action = SqliRequestAction.model_validate(action_data)
    assert len(action.compose_value()) <= 96
    assert set(action.components()) == {
        "field",
        "base_term",
        "close_quote",
        "connector",
        "predicate",
        "comment",
    }
    assert sample.inference_ms >= 0


def test_role_local_ppo_updates_only_from_accepted_transition_batch():
    policy = SqliPPOPolicy("blue", seed=23)
    obs = np.asarray([0.25, 0.75, 0.2, 0.1, 0.4, 1.0, 0.0, 0.0], dtype=np.float32)
    _action, sample = policy.act(obs, stochastic=True)
    selected_action = sample.action_indices[0]
    initial_probability = policy.probabilities(obs)[0][selected_action]
    before = policy.policy_version
    no_update = policy.update([])
    assert no_update["sample_count"] == 0
    assert policy.policy_version == before

    metrics = policy.update(
        [
            {
                "observation": obs,
                "action_indices": sample.action_indices,
                "old_log_probability": sample.old_log_probability,
                "value": sample.value_estimate,
                "reward": 1.0,
                "next_value": 0.0,
                "terminated": True,
                "truncated": False,
            }
        ]
    )
    assert metrics["sample_count"] == 1
    assert metrics["optimizer_steps"] == 4
    assert policy.optimizer_step_count == 4
    assert metrics["post_update_approx_kl"] == metrics["approx_kl"]
    assert metrics["post_update_clip_fraction"] == metrics["clip_fraction"]
    assert policy.policy_version == before + 1
    after_first_update = policy.probabilities(obs)[0][selected_action]
    assert after_first_update > initial_probability
    policy.update(
        [
            {
                "observation": obs,
                "action_indices": sample.action_indices,
                "old_log_probability": sample.old_log_probability,
                "value": sample.value_estimate,
                "reward": 1.0,
                "next_value": 0.0,
                "terminated": True,
                "truncated": False,
            }
        ]
    )
    assert policy.optimizer_step_count == 8
    assert policy.policy_version == before + 2
    assert policy.probabilities(obs)[0][selected_action] > after_first_update


def test_ppo_settings_capture_the_values_used_by_the_optimizer():
    settings = ppo_settings(0.0003)
    assert settings == {
        "algorithm": "PPO_clipped_surrogate_GAE",
        "actor_critic": "factorized_gru_actor_critic",
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "policy_loss_coefficient": 1.0,
        "value_loss_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "optimizer": "Adam",
        "learning_rate": 0.0003,
        "gradient_clip_max_norm": 0.5,
        "optimization_passes": 4,
        "advantage_normalization": "mean_std_when_batch_has_multiple_transitions",
    }


def test_red_reward_v2_uses_verified_syntax_progress_and_rejection_only():
    syntax = _red_reward_components(
        accepted=True,
        verified_unauthorized_rows=0,
        accepted_syntax_progress=True,
        query_rejected=False,
    )
    assert syntax == {
        "confirmed_unauthorized_read": 0.0,
        "accepted_syntax_progress": 0.08,
        "rejected_request": 0.0,
    }
    rejected = _red_reward_components(
        accepted=True,
        verified_unauthorized_rows=0,
        accepted_syntax_progress=False,
        query_rejected=True,
    )
    assert sum(rejected.values()) == -0.01
    disclosed = _red_reward_components(
        accepted=True,
        verified_unauthorized_rows=2,
        accepted_syntax_progress=True,
        query_rejected=True,
    )
    assert sum(disclosed.values()) == 1.0
    unknown = _red_reward_components(
        accepted=False,
        verified_unauthorized_rows=2,
        accepted_syntax_progress=True,
        query_rejected=True,
    )
    assert sum(unknown.values()) == 0.0


def test_retained_v1_reward_contract_keeps_its_original_components():
    accepted = _red_reward_components(
        accepted=True,
        verified_unauthorized_rows=0,
        accepted_syntax_progress=True,
        query_rejected=False,
        returned_rows=True,
        reward_schema_id="sqli.reward.v1",
    )
    assert accepted == {
        "confirmed_unauthorized_read": 0.0,
        "returned_row_progress": 0.02,
        "request_cost": -0.02,
    }
    rejected = _red_reward_components(
        accepted=True,
        verified_unauthorized_rows=0,
        accepted_syntax_progress=False,
        query_rejected=True,
        reward_schema_id="sqli.reward.v1",
    )
    assert rejected["request_cost"] == -0.02


def test_transition_batch_excludes_rejected_and_unknown_actions():
    red = SqliPPOPolicy("red", seed=31)
    obs = np.zeros(len(FEATURES["red"]), dtype=np.float32)
    _action, sample = red.act(obs, stochastic=True)
    accepted = {
        "red_decision_id": "decision-accepted",
        "step_id": 0,
        "red_accepted": True,
        "red_reward": 1.0,
        "red_action": SqliRequestAction(field="handle", base_term="rhea", close_quote=False, connector="none", predicate="none", comment=False),
        "red_observation_array": obs,
        "red_sample": sample,
    }
    rejected = {
        "red_decision_id": "decision-rejected",
        "step_id": 1,
        "red_accepted": False,
        "red_reward": 0.0,
        "red_action": SqliRequestAction(field="handle", base_term="rhea", close_quote=False, connector="none", predicate="none", comment=False),
        "red_observation_array": obs,
        "red_sample": sample,
    }
    batch = _transition_batch([accepted, rejected], "red", red)
    assert len(batch) == 1
    assert batch[0]["truncated"] is True


def test_outcome_rules_require_verified_red_or_forty_round_blue_evidence():
    red = _episode_winner(
        status="completed",
        challenge_mode=True,
        challenge_disclosed=True,
        verified_exposure=True,
        steps=7,
        max_steps=40,
        final_legitimate_served=True,
    )
    assert red == ("red", "verified_secret_disclosure")

    blue = _episode_winner(
        status="completed",
        challenge_mode=True,
        challenge_disclosed=False,
        verified_exposure=False,
        steps=40,
        max_steps=40,
        final_legitimate_served=True,
    )
    assert blue == ("blue", "40_rounds_without_disclosure")

    failed_target = _episode_winner(
        status="incomplete",
        challenge_mode=True,
        challenge_disclosed=False,
        verified_exposure=False,
        steps=40,
        max_steps=40,
        final_legitimate_served=True,
    )
    assert failed_target == (None, None)

    unavailable_owner_lookup = _episode_winner(
        status="completed",
        challenge_mode=True,
        challenge_disclosed=False,
        verified_exposure=False,
        steps=40,
        max_steps=40,
        final_legitimate_served=False,
    )
    assert unavailable_owner_lookup == (None, None)


def test_only_verified_blue_guest_isolation_makes_http_503_an_expected_block():
    expected_controls = {"state_version": 4, "isolate_guest_remaining": 2}
    isolated = {
        "status": "completed",
        "http_status": 503,
        "response": {"detail": "temporary guest isolation"},
        "telemetry": {
            "evidence_status": "verified",
            "controls": {"state_version": 4, "isolate_guest_remaining": 1},
        },
    }
    assert _intentional_guest_isolation(isolated, expected_controls)

    unrelated_failure = {
        **isolated,
        "response": {"detail": "target unavailable"},
    }
    assert not _intentional_guest_isolation(unrelated_failure, expected_controls)

    mismatched_version = {
        **isolated,
        "telemetry": {
            **isolated["telemetry"],
            "controls": {"state_version": 3, "isolate_guest_remaining": 1},
        },
    }
    assert not _intentional_guest_isolation(mismatched_version, expected_controls)

    exhausted_control = _intentional_guest_isolation(
        isolated, {"state_version": 4, "isolate_guest_remaining": 0}
    )
    assert not exhausted_control


def test_sqli_target_evidence_lookup_reads_backwards_across_large_jsonl_lines(tmp_path):
    target_id = "recent-target-event"
    wanted = {"target_event_id": target_id, "challenge_disclosed": True}
    records = [
        json.dumps(wanted, separators=(",", ":")),
        json.dumps({"target_event_id": "later-event", "padding": "x" * 80_000}),
    ]
    evidence = tmp_path / "events.jsonl"
    evidence.write_text("\n".join(records) + "\n", encoding="utf-8")

    assert _find_sqli_target_event(evidence, target_id) == wanted


def test_sqlite_fixture_database_allows_only_read_queries(tmp_path):
    database = SqliDatabase(f"sqlite:///{tmp_path / 'sqli.sqlite3'}")
    try:
        rows = database.execute_lookup(
            actor_ref="guest",
            field=ATTACK.field,
            query_value=ATTACK.compose_value(),
            parameterized=False,
        )
        assert len(rows) == 6
        assert {row["owner_ref"] for row in rows} == {"researcher", "alice", "bob", "operations"}
        assert all(row["email"].endswith("@example.test") for row in rows)

        defended = database.execute_lookup(
            actor_ref="guest",
            field=ATTACK.field,
            query_value=ATTACK.compose_value(),
            parameterized=True,
        )
        assert defended == []
        legitimate = database.execute_lookup(
            actor_ref="researcher",
            field=LEGITIMATE.field,
            query_value=LEGITIMATE.compose_value(),
            parameterized=True,
        )
        assert [row["record_ref"] for row in legitimate] == ["rec-001"]

        with pytest.raises(DBAPIError), database.engine.begin() as connection:
            connection.exec_driver_sql("UPDATE sqli_records SET handle='changed'")
    finally:
        database.close()


def test_http_target_records_server_stages_and_parameterization_preserves_legitimate_query(tmp_path, monkeypatch):
    evidence = tmp_path / "events.jsonl"
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'target.sqlite3'}",
        evidence_path=evidence,
        control_inbox=tmp_path / "control-inbox",
        control_results=tmp_path / "control-results",
    )
    with TestClient(app) as client:
        headers = {
            "X-Range-Run-ID": "run-17",
            "X-Range-Episode-ID": "episode-2",
            "X-Range-Request-ID": "request-1",
            "X-Range-Actor": "guest",
        }
        response = client.post(
            "/api/lookup",
            json={"field": "handle", "query_value": ATTACK.compose_value()},
            headers=headers,
        )
        assert response.status_code == 200
        assert len(response.json()["rows"]) == 6
        target_event_id = response.headers["X-Sqli-Target-Event-ID"]
        record = json.loads(evidence.read_text().splitlines()[-1])
        assert record["target_event_id"] == target_event_id
        assert record["request_id"] == "request-1"
        assert record["unauthorized_rows"] == 6
        assert [stage["stage_id"] for stage in record["trace"]] == [
            "request_received",
            "application_route",
            "request_validation",
            "authentication_and_access",
            "blue_controls",
            "database_operation",
            "http_response",
        ]
        monkeypatch.setattr(range_api, "SQLI_TARGET_EVIDENCE", evidence)
        traced = asyncio.run(range_api.get_server_trace(UUID(target_event_id)))
        assert traced["target_event_id"] == target_event_id
        assert traced["trace"] == record["trace"]

        app.state.runtime.parameterized = True
        defended = client.post(
            "/api/lookup",
            json={"field": "handle", "query_value": ATTACK.compose_value()},
            headers={**headers, "X-Range-Request-ID": "request-2"},
        )
        assert defended.status_code == 200
        assert defended.json()["rows"] == []

        legitimate = client.post(
            "/api/lookup",
            json={"field": "handle", "query_value": LEGITIMATE.compose_value()},
            headers={
                **headers,
                "X-Range-Actor": "researcher",
                "X-Range-Request-ID": "request-3",
            },
        )
        assert legitimate.status_code == 200
        assert [row["record_ref"] for row in legitimate.json()["rows"]] == ["rec-001"]

        app.state.runtime.isolate_guest_remaining = 1
        blocked = client.post(
            "/api/lookup",
            json={"field": "handle", "query_value": LEGITIMATE.compose_value()},
            headers={**headers, "X-Range-Request-ID": "request-4"},
        )
        assert blocked.status_code == 503
        blocked_record = json.loads(evidence.read_text().splitlines()[-1])
        assert blocked_record["trace"][-3]["stage_id"] == "blue_controls"
        assert blocked_record["trace"][-3]["status"] == "blocked"
        not_reached = blocked_record["trace"][-2]
        assert not_reached["stage_id"] == "database_operation"
        assert not_reached["status"] == "not_reached"
        assert not_reached["result"]["reason"] == "request stopped by an active server control"


def test_challenge_phrase_is_volatile_and_redacted_before_executor_persistence(tmp_path):
    evidence = tmp_path / "events.jsonl"
    secret = "classroom-canary-482"
    challenge_id = "challenge-volatile-1"
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'challenge.sqlite3'}",
        evidence_path=evidence,
        control_inbox=tmp_path / "control-inbox",
        control_results=tmp_path / "control-results",
    )
    with TestClient(app) as client:
        setup = client.post(
            f"/internal/challenges/{challenge_id}", json={"phrase": secret}
        )
        assert setup.status_code == 200
        assert client.get(f"/internal/challenges/{challenge_id}/phrase").json() == {
            "phrase": secret
        }

        response = client.post(
            "/api/lookup",
            json={"field": "handle", "query_value": ATTACK.compose_value()},
            headers={
                "X-Range-Run-ID": "challenge-run",
                "X-Range-Episode-ID": "challenge-episode",
                "X-Range-Request-ID": "challenge-request",
                "X-Range-Actor": "guest",
                "X-Range-Challenge-ID": challenge_id,
            },
        )
        assert response.status_code == 200
        assert response.json()["challenge_phrase"] == secret
        target_record = json.loads(evidence.read_text().splitlines()[-1])
        assert target_record["challenge_disclosed"] is True
        assert target_record["challenge_id"] == challenge_id
        assert secret not in evidence.read_text()

        redacted = SqliExecutor._redact_phrase(
            {"detail": secret, "nested": [f"prefix-{secret}-suffix"]}, secret
        )
        assert redacted == {
            "detail": "[redacted]",
            "nested": ["prefix-[redacted]-suffix"],
        }
        clear = client.delete(f"/internal/challenges/{challenge_id}")
        assert clear.status_code == 200
        assert client.get(f"/internal/challenges/{challenge_id}/phrase").status_code == 404


def test_target_controls_are_guest_scoped_and_independently_measurable():
    runtime = SqliRuntime()
    reset = SqliControlCommand(
        command_id=__import__("uuid").uuid4(),
        run_id=__import__("uuid").uuid4(),
        episode_id=__import__("uuid").uuid4(),
        action="RESET",
    )
    runtime.apply(reset)
    runtime.apply(
        SqliControlCommand(
            command_id=__import__("uuid").uuid4(),
            run_id=reset.run_id,
            episode_id=reset.episode_id,
            action="RATE_LIMIT_GUEST",
        )
    )
    assert runtime.check_request("guest", str(reset.episode_id))[0] is None
    assert runtime.check_request("guest", str(reset.episode_id))[0] == 429
    assert runtime.check_request("researcher", str(reset.episode_id)) == (None, None)

    runtime.apply(
        SqliControlCommand(
            command_id=__import__("uuid").uuid4(),
            run_id=reset.run_id,
            episode_id=reset.episode_id,
            action="ISOLATE_GUEST_LOOKUP",
        )
    )
    assert runtime.check_request("guest", str(reset.episode_id))[0] == 503
    assert runtime.check_request("researcher", str(reset.episode_id)) == (None, None)
    assert BLUE_ACTIONS == (
        "MONITOR",
        "PARAMETERIZE_QUERY",
        "RATE_LIMIT_GUEST",
        "ISOLATE_GUEST_LOOKUP",
    )


def test_executor_origin_is_fixed_and_rejects_other_hosts():
    assert validate_target_url("http://sqli-target:8002") == "http://sqli-target:8002"
    for value in (
        "http://example.com:8002",
        "https://sqli-target:8002",
        "http://sqli-target:8001",
        "http://user@sqli-target:8002",
        "http://sqli-target:8002/path",
    ):
        with pytest.raises(ValueError):
            validate_target_url(value)


def test_sqli_checkpoint_contract_accepts_its_scenario_and_rejects_incompatible_runs():
    versions = {"red": 1, "blue": 1}
    settings = ppo_settings(0.0003)
    metadata = {
        "schema_id": "sqli.checkpoint_metadata.v3",
        "model_architecture": "factorized_gru_actor_critic.v1",
        "training_run_id": "fresh-pair",
        "target_version": CHECKPOINT_CONTRACT["target_version"],
        "reward_schema_id": CHECKPOINT_CONTRACT["reward_schema_id"],
        "interaction_protocol_schema_id": INTERACTION_PROTOCOL_SCHEMA,
        "completed_training_updates": 0,
        "policy_versions": versions,
        "optimization_passes": PPO_OPTIMIZATION_PASSES,
        "campaign_revision": "balanced_adversarial_gru_ppo4_v4",
        "ppo_settings": settings,
    }
    payload = {
        **CHECKPOINT_CONTRACT,
        "model_architecture": "factorized_gru_actor_critic.v1",
        "training_run_id": "fresh-pair",
        "completed_training_updates": 0,
        "campaign_revision": "balanced_adversarial_gru_ppo4_v4",
        "policy_versions": versions,
        "ppo_settings": settings,
        "checkpoint_metadata": metadata,
    }
    _validate_checkpoint_contract(payload)
    legacy_reward = {**payload, "reward_schema_id": "sqli.reward.v1"}
    legacy_reward["checkpoint_metadata"] = {**metadata, "reward_schema_id": "sqli.reward.v1"}
    _validate_checkpoint_contract(legacy_reward)
    wrong_protocol = {**payload, "interaction_protocol_schema_id": "sqli.round_protocol.red_first.v1"}
    with pytest.raises(ValueError, match="metadata is inconsistent"):
        _validate_checkpoint_contract(wrong_protocol)
    with pytest.raises(ValueError, match="incompatible"):
        _validate_checkpoint_contract({**payload, "scenario_id": "violet_ledger_v1"})


def test_v2_checkpoint_requires_consistent_policy_metadata_and_four_ppo_passes():
    versions = {"red": 117, "blue": 117}
    settings = ppo_settings(0.0003)
    payload = {
        **CHECKPOINT_CONTRACT,
        "format": "sqli.joint_ppo.v2",
        "training_run_id": "classroom-continuation",
        "completed_training_updates": 40,
        "campaign_revision": "red_syntax_progress_ppo4_v2",
        "policy_versions": versions,
        "ppo_settings": settings,
        "checkpoint_metadata": {
            "schema_id": "sqli.checkpoint_metadata.v1",
            "training_run_id": "classroom-continuation",
            "target_version": CHECKPOINT_CONTRACT["target_version"],
            "reward_schema_id": CHECKPOINT_CONTRACT["reward_schema_id"],
            "completed_training_updates": 40,
            "policy_versions": versions,
            "optimization_passes": PPO_OPTIMIZATION_PASSES,
            "campaign_revision": "red_syntax_progress_ppo4_v2",
            "ppo_settings": settings,
        },
    }
    _validate_checkpoint_contract(payload, expected_training_run_id="classroom-continuation")
    payload["checkpoint_metadata"]["completed_training_updates"] = 39
    with pytest.raises(ValueError, match="metadata is inconsistent"):
        _validate_checkpoint_contract(payload)


def test_round_protocol_uses_a_shared_snapshot_applies_blue_first_and_keeps_probes_out_of_rollouts():
    events = []
    jobs = {}
    controls = {}

    def result_for(job_id: str) -> dict:
        job = jobs[job_id]
        purpose = job["purpose"]
        actor = "researcher" if purpose == "verifier_legitimate" else "guest"
        if purpose == "policy":
            rows = [{"record_ref": "rec-guest", "owner_ref": "guest"}]
            syntax_signal = True
        elif purpose == "verifier_legitimate":
            rows = [{"record_ref": "rec-001", "owner_ref": "researcher"}]
            syntax_signal = False
        else:
            rows = []
            syntax_signal = False
        return {
            "job_id": job_id,
            "run_id": job["run_id"],
            "episode_id": job["episode_id"],
            "decision_id": job["decision_id"],
            "step_id": job["step_id"],
            "request_id": f"request-{job_id}",
            "purpose": purpose,
            "status": "completed",
            "actor_ref": actor,
            "http_status": 200,
            "target_event_id": f"target-{job_id}",
            "request": {
                "field": job["action"]["field"],
                "query_value": job["action"]["base_term"],
            },
            "response": {"rows": rows},
            "telemetry": {
                "evidence_status": "verified",
                "unauthorized_rows": 0,
                "row_count": len(rows),
                "query_syntax_signal": syntax_signal,
                "query_rejected": False,
                "query_length": 4,
            },
        }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        payload = json.loads(request.content) if request.content else {}
        if request.method == "POST" and path == "/range/sqli/events":
            events.append(payload)
            return httpx.Response(200, json={"accepted": True}, request=request)
        if request.method == "POST" and path == "/range/sqli/controls":
            command_id = str(uuid4())
            controls[command_id] = payload["action"]
            return httpx.Response(200, json={"command_id": command_id}, request=request)
        if request.method == "GET" and path.startswith("/range/sqli/controls/"):
            command_id = path.rsplit("/", 1)[-1]
            action = controls[command_id]
            active = {
                "parameterized": action == "PARAMETERIZE_QUERY",
                "rate_limit_guest": False,
                "isolate_guest_remaining": 0,
            }
            return httpx.Response(
                200, json={"status": "applied", "command_id": command_id, "controls": active}, request=request
            )
        if request.method == "POST" and path == "/range/sqli/jobs":
            job_id = str(uuid4())
            jobs[job_id] = payload
            return httpx.Response(200, json={"job_id": job_id}, request=request)
        if request.method == "GET" and path.startswith("/range/sqli/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            return httpx.Response(200, json=result_for(job_id), request=request)
        if request.method == "POST" and path.endswith("/finish"):
            return httpx.Response(200, json={"status": payload["status"]}, request=request)
        return httpx.Response(404, json={"detail": path}, request=request)

    class FixedPolicy:
        def __init__(self, role: str):
            self.role = role
            self.policy_version = 1
            self.observations = []
            if role == "red":
                self.action = {
                    "field": "handle",
                    "base_term": "rhea",
                    "close_quote": True,
                    "connector": "OR",
                    "predicate": "tautology",
                    "comment": True,
                }
                distributions = tuple(tuple([1.0 / len(values)] * len(values)) for values in RED_COMPONENTS.values())
                indices = (0, 0, 0, 0, 0, 0)
            else:
                self.action = {"operation": "PARAMETERIZE_QUERY"}
                distributions = ((0.25, 0.25, 0.25, 0.25),)
                indices = (1,)
            self.sample = SqliSample(
                action_indices=indices,
                component_probabilities=distributions,
                selected_probabilities=tuple(value[0] for value in distributions),
                old_log_probability=-0.1,
                value_estimate=0.0,
                entropy=0.0,
                policy_version=1,
                inference_ms=0.1,
            )

        def act(self, observation, *, stochastic: bool):
            self.observations.append(np.asarray(observation, dtype=np.float32).copy())
            return self.action, self.sample

        def reset_hidden_state(self):
            return None

        def value(self, _observation, _recurrent_state=None):
            return 0.0

    red = FixedPolicy("red")
    blue = FixedPolicy("blue")
    run_id, episode_id = uuid4(), uuid4()

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://violet.test"
        ) as client:
            runner = SqliEpisodeRunner(
                client,
                run_id,
                episode_id,
                seed=17,
                max_steps=2,
                red_policy=red,
                blue_policy=blue,
                mode="evaluate",
            )
            return await runner.run()

    result = asyncio.run(exercise())
    assert result["summary"]["status"] == "completed"
    assert len(red.observations) == len(blue.observations) == 2
    assert red.observations[0][3] == 0
    assert red.observations[1][3] > 0  # Red adapts using the preceding target response.

    decisions = [item for item in events if item["event_type"] in {"sqli_red_decision", "sqli_blue_decision"}]
    for step in range(2):
        pair = [item for item in decisions if item["payload"]["step_id"] == step]
        assert len(pair) == 2
        assert pair[0]["payload"]["round_snapshot_id"] == pair[1]["payload"]["round_snapshot_id"]
        assert pair[0]["payload"]["observation_phase"] == pair[1]["payload"]["observation_phase"] == "round_start"
    first_blue = next(item for item in decisions if item["event_type"] == "sqli_blue_decision")
    assert first_blue["payload"]["observation"]["telemetry"]["guest_request_count"] == 0

    for step in range(2):
        ack = next(
            item for item in events
            if item["event_type"] == "sqli_control_acknowledged" and item["payload"].get("step_id") == step
        )
        policy_response = next(
            item for item in events
            if item["event_type"] == "sqli_target_response"
            and item["payload"].get("purpose") == "policy"
            and item["payload"].get("step_id") == step
        )
        assert events.index(ack) < events.index(policy_response)

    transitions = [item["payload"] for item in events if item["event_type"] == "sqli_transition"]
    assert len(transitions) == 4
    assert {item["role"] for item in transitions} == {"red", "blue"}
    assert all(item["learning_eligible"] is False for item in transitions)
    assert len([item for item in events if item["event_type"] == "sqli_target_response" and item["payload"].get("purpose", "").startswith("verifier_")]) == 4


def test_incomplete_requested_training_episode_gets_three_attempts_and_stays_failed(
    monkeypatch, tmp_path
):
    from violet.sqli_lab import training

    attempts = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"stop_requested": False}

    class Client:
        async def post(self, *_args, **_kwargs):
            return Response()

        async def get(self, *_args, **_kwargs):
            return Response()

    class IncompleteEpisodeRunner:
        def __init__(self, _client, _run_id, episode_id, **kwargs):
            self.episode_id = episode_id
            self.seed = kwargs["seed"]

        async def run(self):
            attempts.append(self.episode_id)
            return {
                "status": "incomplete",
                "reason": "unknown_or_unverified_outcome",
                "trace": [],
                "summary": {
                    "episode_id": str(self.episode_id),
                    "seed": self.seed,
                    "status": "incomplete",
                    "reason": "unknown_or_unverified_outcome",
                    "steps": 0,
                    "max_steps": 40,
                },
            }

        async def _event(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(training, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(training, "SqliEpisodeRunner", IncompleteEpisodeRunner)
    run_id = str(uuid4())
    result = __import__("asyncio").run(
        run_sqli_job(
            Client(),
            {
                "training_run_id": run_id,
                "mode": "train",
                "config": {
                    "scenario_id": SCENARIO_ID,
                    "target_version": TARGET_VERSION,
                    "observation_schema_id": OBSERVATION_SCHEMA,
                    "action_schema_id": ACTION_SCHEMA,
                    "reward_schema_id": REWARD_SCHEMA,
                    "interaction_protocol_schema_id": INTERACTION_PROTOCOL_SCHEMA,
                    "seed": 17,
                    "learning_rate": 0.0003,
                    "episodes": 1,
                    "max_steps": 40,
                    "run_kind": "teaching",
                    "campaign_label": "live_demo_v1",
                    "campaign_revision": "red_syntax_progress_ppo4_v2",
                },
            },
        )
    )

    assert len(attempts) == 3
    assert result["episode_summary_count"] == 3
    assert [item["attempt"] for item in result["episodes"]] == [1, 2, 3]
    assert result["completed_training_episodes"] == 0
    assert result["training_campaign_complete"] is False
    assert result["research_criteria_met"] is False
