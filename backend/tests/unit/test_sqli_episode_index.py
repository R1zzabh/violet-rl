from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from violet.api import range as range_api
from violet.persistence.models import Episode, Run, RunEvent, TrainingRunRecord
from violet.persistence.repository import EventRepository


def test_sqli_episode_index_cursor_paginates_and_filters_every_persisted_phase(monkeypatch) -> None:
    repository = EventRepository("sqlite+pysqlite:///:memory:")
    repository.initialize()
    training_run_id = str(uuid4())
    challenge_run_id = str(uuid4())
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    expected_ids: list[str] = []
    challenge_red_ids: set[str] = set()
    per_run_index = {training_run_id: 0, challenge_run_id: 0}

    with repository.sessions.begin() as session:
        session.add_all(
            [
                Run(
                    run_id=training_run_id,
                    mode="real_range_training",
                    target_version="sqli_target_v1",
                    status="completed",
                    config={"scenario_id": "sqli_lab_v1"},
                ),
                Run(
                    run_id=challenge_run_id,
                    mode="real_range",
                    target_version="sqli_target_v1",
                    status="completed",
                    config={"scenario_id": "sqli_lab_v1"},
                ),
                TrainingRunRecord(
                    training_run_id=training_run_id,
                    role="joint",
                    mode="train",
                    status="completed",
                    config={"scenario_id": "sqli_lab_v1", "run_kind": "teaching"},
                    created_at=base_time,
                    updated_at=base_time,
                ),
            ]
        )
        for ordinal in range(521):
            run_id = training_run_id if ordinal % 2 == 0 else challenge_run_id
            episode_id = str(UUID(int=ordinal + 1))
            episode_index = per_run_index[run_id]
            per_run_index[run_id] += 1
            is_partial = ordinal == 520
            status = "running" if is_partial else "incomplete" if ordinal % 17 == 0 else "completed"
            phase = "training" if run_id == training_run_id else "challenge"
            winner = "red" if ordinal % 4 == 1 else "blue"
            created_at = base_time + timedelta(seconds=ordinal)
            expected_ids.append(episode_id)
            if phase == "challenge" and winner == "red":
                challenge_red_ids.add(episode_id)

            session.add(
                Episode(
                    episode_id=episode_id,
                    run_id=run_id,
                    episode_index=episode_index,
                    seed=ordinal,
                    status=status,
                    config={"scenario_id": "sqli_lab_v1"},
                    created_at=created_at,
                )
            )
            session.add(
                RunEvent(
                    event_id=str(UUID(int=10_000 + ordinal * 2)),
                    run_id=run_id,
                    episode_id=episode_id,
                    seq=episode_index * 2 + 1,
                    event_type="sqli_episode_started",
                    occurred_at=created_at,
                    payload={"episode_kind": phase, "seed": ordinal, "policy_versions": {"blue": ordinal}},
                    previous_hash="",
                    event_hash="a" * 64,
                )
            )
            if not is_partial:
                session.add(
                    RunEvent(
                        event_id=str(UUID(int=10_001 + ordinal * 2)),
                        run_id=run_id,
                        episode_id=episode_id,
                        seq=episode_index * 2 + 2,
                        event_type="sqli_episode_outcome",
                        occurred_at=created_at + timedelta(milliseconds=1),
                        payload={
                            "phase": phase,
                            "status": status,
                            "winner": winner,
                            "reason": f"recorded_{winner}_outcome",
                            "steps": 2,
                            "verified_exposure": winner == "red",
                        },
                        previous_hash="a" * 64,
                        event_hash="b" * 64,
                    )
                )

    monkeypatch.setattr(range_api, "_repository", lambda: repository)
    pages = []
    cursor = None
    while True:
        page = asyncio.run(range_api.list_sqli_episodes(cursor=cursor, limit=100))
        pages.append(page)
        if not page["has_more"]:
            break
        cursor = page["next_cursor"]

    all_items = [item for page in pages for item in page["items"]]
    assert len(all_items) == 521
    assert len({item["episode_id"] for item in all_items}) == 521
    assert {item["episode_id"] for item in all_items} == set(expected_ids)
    assert all_items[0]["episode_id"] == expected_ids[-1]
    assert all_items[-1]["episode_id"] == expected_ids[0]
    assert sum(item["status"] == "running" for item in all_items) == 1
    assert all_items[0]["phase"] == "training"
    assert all_items[0]["outcome_key"] == "unknown"

    filtered = asyncio.run(
        range_api.list_sqli_episodes(
            limit=100,
            run_id=UUID(challenge_run_id),
            phase="challenge",
            outcome="red",
        )
    )
    assert filtered["has_more"] is True
    assert len(filtered["items"]) == 100
    assert {item["episode_id"] for item in filtered["items"]} <= challenge_red_ids
    assert all(item["run_mode"] == "challenge" for item in filtered["items"])
    assert all(item["phase"] == "challenge" and item["outcome_key"] == "red" for item in filtered["items"])

    training = asyncio.run(
        range_api.list_sqli_episodes(limit=10, run_id=UUID(training_run_id), phase="training")
    )
    assert len(training["items"]) == 10
    assert all(item["run_mode"] == "train" and item["run_type"] == "teaching" for item in training["items"])
    repository.close()
