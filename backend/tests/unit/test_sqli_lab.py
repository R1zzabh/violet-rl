from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import DBAPIError

from violet.sqli_lab.app import create_app
from violet.sqli_lab.contracts import BLUE_ACTIONS, SqliControlCommand, SqliRequestAction
from violet.sqli_lab.control import SqliRuntime
from violet.sqli_lab.database import SqliDatabase
from violet.sqli_lab.executor_main import validate_target_url
from violet.sqli_lab.learning import FEATURES, SqliPPOPolicy
from violet.sqli_lab.training import (
    CHECKPOINT_CONTRACT,
    _transition_batch,
    _validate_checkpoint_contract,
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
    assert metrics["optimizer_steps"] == 1
    assert policy.policy_version == before + 1


def test_transition_batch_excludes_rejected_and_unknown_actions():
    red = SqliPPOPolicy("red", seed=31)
    obs = np.zeros(len(FEATURES["red"]), dtype=np.float32)
    _action, sample = red.act(obs, stochastic=True)
    accepted = {
        "red_accepted": True,
        "red_reward": 1.0,
        "red_observation_array": obs,
        "red_sample": sample,
    }
    rejected = {
        "red_accepted": False,
        "red_reward": 0.0,
        "red_observation_array": obs,
        "red_sample": sample,
    }
    batch = _transition_batch([accepted, rejected], "red", red)
    assert len(batch) == 1
    assert batch[0]["truncated"] is True


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


def test_http_target_correlates_evidence_and_parameterization_preserves_legitimate_query(tmp_path):
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
    _validate_checkpoint_contract(CHECKPOINT_CONTRACT)
    with pytest.raises(ValueError, match="incompatible"):
        _validate_checkpoint_contract({**CHECKPOINT_CONTRACT, "scenario_id": "violet_ledger_v1"})
