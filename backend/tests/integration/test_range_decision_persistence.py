from uuid import uuid4

from violet.persistence.repository import EventRepository
from violet.range.contracts import AgentDecision, ControlAcknowledgement, ControlCommand
from violet.range.observations import build_blue_observation
from violet.range.rewards import red_reward


def test_decision_and_reward_projections_are_idempotent_and_hash_linked(tmp_path) -> None:
    repository = EventRepository(f"sqlite:///{tmp_path / 'decisions.sqlite3'}")
    repository.initialize()
    run_id, episode_id = uuid4(), uuid4()
    reset = ControlCommand(
        command_id=uuid4(), command="RESET", seed=9, run_id=run_id, episode_id=episode_id
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
            fixture_seed=9,
            completed_at="2026-10-13T00:00:00+00:00",
        )
    )
    observation = build_blue_observation(
        run_id=run_id, episode_id=episode_id, step_id=0, events=[], remaining_requests=40
    )
    decision = AgentDecision(
        decision_id=uuid4(),
        run_id=run_id,
        episode_id=episode_id,
        step_id=0,
        role="blue",
        policy_id="rule-based-v1",
        observation_schema_id="blue.range_observation.v2",
        legal_action_mask=observation.legal_action_mask,
        selected_action={"operation": "MONITOR", "principal_ref": None},
        inference_ms=0,
    )
    repository.record_decision(decision, observation)
    repository.record_decision(decision, observation)
    reward = red_reward(None)
    repository.record_reward(str(run_id), str(episode_id), 0, reward)
    repository.record_reward(str(run_id), str(episode_id), 0, reward)

    events = repository.events(str(run_id))
    decision_events = [
        event for event in events if event["event_type"] == "agent_decision_recorded"
    ]
    reward_events = [event for event in events if event["event_type"] == "reward_recorded"]
    assert len(decision_events) == 1
    assert len(reward_events) == 1
    stored_decision = repository.get_decision(str(decision.decision_id))
    assert stored_decision is not None
    assert stored_decision["selected_action"] == decision.selected_action
    assert all(
        events[index]["previous_hash"] == events[index - 1]["event_hash"]
        for index in range(1, len(events))
    )
    repository.close()
