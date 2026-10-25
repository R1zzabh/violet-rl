import json
from uuid import uuid4

from violet.persistence.export import export_run
from violet.persistence.repository import EventRepository
from violet.range.contracts import (
    AuthTestAction,
    ControlAcknowledgement,
    ControlCommand,
    ExecutorJob,
    ExecutorResult,
)


def test_event_sequence_survives_repository_restart_and_is_idempotent(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'range-events.sqlite3'}"
    run_id, episode_id = uuid4(), uuid4()
    job_id, action_id, request_id = uuid4(), uuid4(), uuid4()
    target_event_id = uuid4()
    job = ExecutorJob(
        job_id=job_id,
        run_id=run_id,
        episode_id=episode_id,
        step_id=0,
        decision_id=uuid4(),
        action_id=action_id,
        request_id=request_id,
        idempotency_key=action_id,
        action=AuthTestAction(operation="AUTH_TEST", credential_ref="synthetic_alice"),
    )
    result = ExecutorResult(
        job_id=job_id,
        action_id=action_id,
        request_id=request_id,
        status="completed",
        http_status=200,
        response_excerpt='{"authenticated":"true","principal_ref":"alice"}',
        target_event_id=target_event_id,
        principal_ref="alice",
        session_handle="A" * 43,
        started_at="2026-10-10T00:00:00+00:00",
        finished_at="2026-10-10T00:00:00.010+00:00",
        duration_ms=10,
    )
    target_evidence = {
        "target_event_id": str(target_event_id),
        "request_id": str(request_id),
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "occurred_at": "2026-10-10T00:00:00+00:00",
    }

    repository = EventRepository(database_url)
    repository.initialize()
    repository.submit_job(job)
    repository.save_execution(job, result, target_evidence, None)
    repository.save_execution(job, result, target_evidence, None)

    reset = ControlCommand(
        command_id=uuid4(),
        command="RESET",
        seed=17,
        run_id=run_id,
        episode_id=episode_id,
    )
    repository.submit_control(reset)
    repository.save_control_ack(
        ControlAcknowledgement(
            command_id=reset.command_id,
            run_id=run_id,
            episode_id=episode_id,
            status="applied",
            state_version=1,
            controls={"ownership_hardened": False},
            fixture_seed=17,
            completed_at="2026-10-10T00:00:01+00:00",
        )
    )
    before_restart = repository.events(str(run_id))
    repository.close()

    reopened = EventRepository(database_url)
    after_restart = reopened.events(str(run_id))
    assert [event["seq"] for event in after_restart] == list(range(1, 8))
    assert [event["event_type"] for event in after_restart] == [
        "run_created",
        "episode_started",
        "action_submitted",
        "target_evidence_ingested",
        "executor_result",
        "target_control_submitted",
        "target_control_acknowledged",
    ]
    assert len(after_restart) == len(before_restart)
    assert after_restart[3]["payload"]["target_event_id"] == str(target_event_id)
    assert reopened.latest_fixture_seed() == 17
    assert reopened.fixture_seed(str(run_id), str(episode_id)) == 17
    assert reopened.fixture_seed(str(uuid4()), str(episode_id)) == 0
    assert all(
        after_restart[index]["previous_hash"] == after_restart[index - 1]["event_hash"]
        for index in range(1, len(after_restart))
    )
    assert reopened.pending_jobs() == []
    output = export_run(reopened, str(run_id), tmp_path / "exports")
    exported = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["event_count"] == 7
    assert manifest["source_event_chain_head"] == after_restart[-1]["event_hash"]
    assert "A" * 43 not in (output / "events.jsonl").read_text()
    exported_results = [
        event["payload"]["result"] for event in exported if event["event_type"] == "executor_result"
    ]
    assert exported_results[0]["session_handle"] == "[REDACTED]"
    reopened.close()


def test_learning_transitions_and_policy_updates_share_run_chain(tmp_path) -> None:
    repository = EventRepository(f"sqlite:///{tmp_path / 'learning-events.sqlite3'}")
    repository.initialize()
    run_id = str(uuid4())
    episode_ids = [str(uuid4()), str(uuid4())]
    for index, episode_id in enumerate(episode_ids):
        repository.record_learning_transition(
            run_id=run_id,
            episode_id=episode_id,
            role="red",
            seed=42 + index,
            payload={"schema_id": "violet.range.observation.v1", "reward": 1.0},
        )
    repository.record_learning_update(
        run_id=run_id,
        episode_id=episode_ids[-1],
        role="red",
        seed=43,
        payload={"policy_version": 1, "transition_count": 2, "metrics": {"loss": 0.1}},
    )
    events = repository.events(run_id)
    assert [event["event_type"] for event in events] == [
        "run_created",
        "episode_started",
        "learning_transition",
        "episode_started",
        "learning_transition",
        "learning_policy_update",
    ]
    assert [event["seq"] for event in events] == list(range(1, 7))
    assert all(
        events[index]["previous_hash"] == events[index - 1]["event_hash"]
        for index in range(1, len(events))
    )
    repository.close()


def test_episode_result_persists_episode_and_run_completion(tmp_path) -> None:
    repository = EventRepository(f"sqlite:///{tmp_path / 'episode-result.sqlite3'}")
    repository.initialize()
    run_id, episode_id = str(uuid4()), str(uuid4())
    repository.record_learning_transition(
        run_id=run_id,
        episode_id=episode_id,
        role="red",
        seed=42,
        payload={"observation": [0.0] * 7},
    )
    repository.record_episode_result(
        run_id=run_id,
        episode_id=episode_id,
        seed=42,
        episode_status="completed",
        run_status="completed",
        summary={"verified_attack": False, "red_return": -0.1},
    )
    assert repository.range_runs()[0]["status"] == "completed"
    events = repository.events(run_id)
    assert events[-1]["event_type"] == "episode_completed"
    assert events[-1]["payload"]["run_status"] == "completed"
    repository.close()
