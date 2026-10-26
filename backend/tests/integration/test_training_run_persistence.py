from uuid import UUID, uuid4

from sqlalchemy import select

from violet.persistence.models import Episode
from violet.persistence.repository import EventRepository


def test_training_run_claim_heartbeat_stop_and_update_are_durable(tmp_path) -> None:
    repository = EventRepository(f"sqlite:///{tmp_path / 'training.sqlite'}")
    repository.initialize()
    try:
        training_id = str(uuid4())
        config = {"episodes": 2, "target_version": "violet_ledger_v1"}
        created = repository.create_training_run(training_id, "red", "train", config)
        assert created["status"] == "queued"
        assert repository.create_training_run(training_id, "red", "train", config) == created

        claimed = repository.claim_training_run("worker-a")
        assert claimed is not None and claimed["training_run_id"] == training_id
        heartbeat = repository.heartbeat_training_run(training_id, "worker-a")
        assert heartbeat is not None and heartbeat["status"] == "running"
        repository.record_training_update(
            training_id,
            "red",
            1,
            12,
            {"loss": 0.25},
            "a" * 64,
        )
        metrics = repository.training_updates(training_id, "red")[0]["metrics"]
        assert metrics is not None
        assert metrics == {"loss": 0.25}
        repository.request_training_stop(training_id)
        stopped = repository.update_training_run(training_id, "worker-a", "completed", {"ok": True})
        assert stopped is not None and stopped["status"] == "stopped"
        assert stopped["stop_requested"] is True
        events = repository.events(training_id)
        assert [event["event_type"] for event in events] == [
            "run_created",
            "training_run_created",
            "training_run_started",
            "training_update_recorded",
            "training_stop_requested",
            "training_run_updated",
        ]
        assert all(
            events[index]["previous_hash"] == events[index - 1]["event_hash"]
            for index in range(1, len(events))
        )
    finally:
        repository.close()


def test_episode_indexes_increment_within_a_training_run(tmp_path) -> None:
    from violet.range.contracts import ControlCommand

    repository = EventRepository(f"sqlite:///{tmp_path / 'episodes.sqlite'}")
    repository.initialize()
    run_id = str(uuid4())
    try:
        episode_ids = [str(uuid4()), str(uuid4())]
        for index, episode_id in enumerate(episode_ids):
            repository.submit_control(
                ControlCommand(
                    command_id=uuid4(),
                    command="RESET",
                    seed=index,
                    run_id=UUID(run_id),
                    episode_id=UUID(episode_id),
                )
            )
        with repository.sessions() as session:
            records = session.scalars(
                select(Episode).where(Episode.run_id == run_id).order_by(Episode.episode_index)
            ).all()
        assert [record.episode_index for record in records] == [0, 1]
    finally:
        repository.close()
