from __future__ import annotations

import asyncio
from uuid import UUID

from violet.api import range as range_api


class ArchiveRepository:
    def __init__(self) -> None:
        self.events = [{"seq": seq, "event_id": str(seq)} for seq in range(1, 1008)]
        self.episodes = [
            {"episode_id": str(index), "episode_index": index, "status": "completed"}
            for index in range(505)
        ]
        self.runs = [{"training_run_id": str(index)} for index in range(205)]

    def episode_exists(self, _run_id: str, _episode_id: str) -> bool:
        return True

    def episode_events_page(self, _run_id: str, _episode_id: str, after_seq: int, limit: int):
        return [event for event in self.events if event["seq"] > after_seq][:limit]

    def list_episodes(
        self,
        _run_id: str,
        limit: int = 100,
        *,
        after_index: int | None = None,
        before_index: int | None = None,
    ):
        if after_index is not None:
            candidates = [item for item in self.episodes if item["episode_index"] > after_index]
            return candidates[:limit]
        candidates = self.episodes
        if before_index is not None:
            candidates = [item for item in candidates if item["episode_index"] < before_index]
        return candidates[-limit:]

    def list_training_runs(self, limit: int = 50, offset: int = 0):
        return list(reversed(self.runs))[offset : offset + limit]


def test_campaign_restart_keeps_the_checkpointed_baseline_after_failed_attempts(monkeypatch) -> None:
    baseline = {
        "training_run_id": "retained-baseline",
        "status": "stopped",
        "result": {"checkpoint_sha256": "baseline-digest"},
        "role": "joint",
        "config": {
            "scenario_id": "sqli_lab_v1",
            "reward_schema_id": "sqli.reward.v1",
        },
    }
    failed_attempt = {
        "training_run_id": "failed-restart",
        "status": "failed",
        "result": {"error": "trainer restart"},
        "config": {},
    }
    completed_restart = {
        "training_run_id": "completed-restart",
        "status": "completed",
        "result": {"checkpoint_sha256": "new-digest"},
        "config": {
            "baseline_training_run_id": "retained-baseline",
            "reward_schema_id": "sqli.reward.v2",
        },
    }
    monkeypatch.setattr(
        range_api,
        "_checkpoint_record_metadata",
        lambda record: {"checkpoint_sha256": "baseline-digest"}
        if record is baseline
        else None,
    )

    assert range_api._retained_campaign_baseline([failed_attempt, baseline]) is baseline
    assert range_api._retained_campaign_baseline(
        [completed_restart, failed_attempt, baseline]
    ) is baseline


def test_checkpoint_metadata_must_match_both_accepted_policy_updates(monkeypatch) -> None:
    digest = "a" * 64
    run_id = "verified-checkpoint"
    updates = [
        {
            "role": role,
            "update_index": 40,
            "created_at": "2026-10-26T00:00:00Z",
            "checkpoint_sha256": digest,
            "metrics": {"policy_version_after": 117},
        }
        for role in ("red", "blue")
    ]

    class Repository:
        def training_updates(self, _training_run_id: str):
            return updates

    monkeypatch.setattr(range_api, "_repository", lambda: Repository())
    record = {
        "training_run_id": run_id,
        "role": "joint",
        "mode": "train",
        "status": "stopped",
        "config": {
            "scenario_id": "sqli_lab_v1",
            "target_version": "sqli_target_v1",
            "reward_schema_id": "sqli.reward.v2",
            "campaign_revision": "red_syntax_progress_ppo4_v2",
        },
        "result": {
            "checkpoint_sha256": digest,
            "completed_training_episodes": 40,
            "policy_versions": {"red": 117, "blue": 117},
            "checkpoint_metadata": {
                "schema_id": "sqli.checkpoint_metadata.v1",
                "training_run_id": run_id,
                "target_version": "sqli_target_v1",
                "reward_schema_id": "sqli.reward.v2",
                "completed_training_updates": 40,
                "optimization_passes": 4,
                "policy_versions": {"red": 117, "blue": 117},
                "campaign_revision": "red_syntax_progress_ppo4_v2",
            },
        },
    }

    assert range_api._checkpoint_record_metadata(record)["checkpoint_sha256"] == digest
    record["result"]["checkpoint_metadata"]["completed_training_updates"] = 39
    assert range_api._checkpoint_record_metadata(record) is None


def test_archive_api_pages_timelines_episodes_and_runs(monkeypatch) -> None:
    repository = ArchiveRepository()
    monkeypatch.setattr(range_api, "_repository", lambda: repository)
    run_id = UUID("00000000-0000-0000-0000-000000000001")
    episode_id = UUID("00000000-0000-0000-0000-000000000002")

    first = asyncio.run(range_api.get_episode_timeline(run_id, episode_id, limit=500))
    second = asyncio.run(
        range_api.get_episode_timeline(
            run_id, episode_id, after_seq=first["next_after_seq"], limit=500
        )
    )
    third = asyncio.run(
        range_api.get_episode_timeline(
            run_id, episode_id, after_seq=second["next_after_seq"], limit=500
        )
    )
    assert [len(page["items"]) for page in (first, second, third)] == [500, 500, 7]
    assert first["has_more"] is True
    assert second["has_more"] is True
    assert third["has_more"] is False
    assert first["items"][0]["seq"] == 1
    assert third["items"][-1]["seq"] == 1007

    latest = asyncio.run(range_api.get_run_episodes(run_id, limit=100))
    older = asyncio.run(
        range_api.get_run_episodes(
            run_id, before_index=latest["next_before_index"], limit=100
        )
    )
    assert len(latest["items"]) == len(older["items"]) == 100
    assert latest["items"][-1]["episode_index"] == 504
    assert older["items"][-1]["episode_index"] == 404
    assert older["has_newer"] is True
    assert older["has_older"] is True

    run_page = asyncio.run(range_api.list_training_runs(limit=100, offset=0))
    next_run_page = asyncio.run(
        range_api.list_training_runs(limit=100, offset=run_page["next_offset"])
    )
    assert len(run_page["items"]) == len(next_run_page["items"]) == 100
    assert run_page["has_more"] is True
    last_run_page = asyncio.run(
        range_api.list_training_runs(limit=100, offset=next_run_page["next_offset"])
    )
    assert len(last_run_page["items"]) == 5
    assert next_run_page["has_more"] is True
    assert last_run_page["has_more"] is False
