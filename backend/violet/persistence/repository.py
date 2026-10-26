"""Transactional repository for run events, executor outbox, and evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from violet.persistence.models import (
    AgentDecisionRecord,
    AgentTransitionRecord,
    Base,
    ControlRecord,
    Episode,
    ExecutorJobRecord,
    RangeDetectionRecord,
    RangeRewardRecord,
    Run,
    RunEvent,
    TargetEvidence,
    TrainingRunRecord,
    TrainingUpdateRecord,
    utc_now,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from violet.range.contracts import (
        ControlAcknowledgement,
        ControlCommand,
        ExecutorJob,
        ExecutorResult,
    )


class EventRepository:
    """Persists canonical events before API responses or live notifications."""

    def __init__(self, database_url: str):
        connect_args: dict[str, Any] = (
            {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        )
        self.engine: Engine = create_engine(
            database_url, pool_pre_ping=True, connect_args=connect_args
        )
        self.sessions: sessionmaker[Session] = sessionmaker(
            bind=self.engine, expire_on_commit=False
        )

    def initialize(self) -> None:
        # SQLite is used by isolated unit/integration fixtures. Production
        # PostgreSQL schemas are owned by the range-migrate Alembic service.
        if self.engine.dialect.name == "sqlite":
            Base.metadata.create_all(self.engine)
            return
        missing = set(Base.metadata.tables) - set(inspect(self.engine).get_table_names())
        if missing:
            raise RuntimeError(
                "VIOLET database schema is not migrated; run range-migrate "
                f"(missing: {', '.join(sorted(missing))})"
            )

    def close(self) -> None:
        self.engine.dispose()

    @staticmethod
    def _timestamp(value: datetime) -> str:
        """Serialize SQL timestamps as UTC even when SQLite drops tzinfo."""
        value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return value.isoformat()

    @staticmethod
    def _locked_run(session: Session, run_id: str) -> Run | None:
        return session.scalar(select(Run).where(Run.run_id == run_id).with_for_update())

    @staticmethod
    def _append_event(
        session: Session,
        run: Run,
        event_type: str,
        payload: dict[str, Any],
        episode_id: str | None,
        event_id: str | None = None,
    ) -> RunEvent:
        occurred_at = utc_now()
        seq = run.next_seq
        envelope = {
            "run_id": run.run_id,
            "episode_id": episode_id,
            "seq": seq,
            "event_type": event_type,
            "occurred_at": occurred_at.isoformat(),
            "payload": payload,
        }
        canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
        previous_hash = run.last_event_hash
        event_hash = hashlib.sha256(f"{previous_hash}{canonical}".encode()).hexdigest()
        event = RunEvent(
            event_id=event_id or str(uuid4()),
            run_id=run.run_id,
            episode_id=episode_id,
            seq=seq,
            event_type=event_type,
            occurred_at=occurred_at,
            payload=payload,
            previous_hash=previous_hash,
            event_hash=event_hash,
        )
        run.next_seq += 1
        run.last_event_hash = event_hash
        session.add(event)
        return event

    def record_lab_event(
        self,
        *,
        run_id: str,
        episode_id: str,
        seed: int,
        event_id: str,
        event_type: str,
        payload: dict[str, Any],
        scenario_id: str,
        target_version: str,
    ) -> None:
        """Append an idempotent event for a versioned non-Ledger lab scenario."""
        with self.sessions.begin() as session:
            existing_event = session.get(RunEvent, event_id)
            if existing_event is not None:
                if (
                    existing_event.run_id != run_id
                    or existing_event.episode_id != episode_id
                    or existing_event.event_type != event_type
                    or existing_event.payload != payload
                ):
                    raise ValueError("event ID was already used for different content")
                return
            run, episode = self._ensure_run_episode(
                session, run_id, episode_id, target_version, seed
            )
            if run.target_version != target_version:
                raise ValueError("event target version does not match the persisted run")
            if not run.config:
                run.config = {"scenario_id": scenario_id, "target_version": target_version}
            elif run.config.get("scenario_id") not in (None, scenario_id):
                raise ValueError("event scenario does not match the persisted run")
            if not episode.config:
                episode.config = {"scenario_id": scenario_id, "target_version": target_version}
            self._append_event(
                session,
                run,
                event_type,
                payload,
                episode_id,
                event_id=event_id,
            )

    @classmethod
    def _ensure_run_episode(
        cls,
        session: Session,
        run_id: str,
        episode_id: str,
        target_version: str,
        seed: int | None = None,
    ) -> tuple[Run, Episode]:
        run = session.scalar(select(Run).where(Run.run_id == run_id).with_for_update())
        if run is None:
            run = Run(run_id=run_id, target_version=target_version)
            session.add(run)
            session.flush()
            cls._append_event(
                session,
                run,
                "run_created",
                {"mode": "real_range", "target_version": target_version},
                None,
            )
        episode = session.get(Episode, episode_id)
        if episode is None:
            current_index = session.scalar(
                select(func.max(Episode.episode_index)).where(Episode.run_id == run_id)
            )
            episode = Episode(
                episode_id=episode_id,
                run_id=run_id,
                episode_index=(int(current_index) + 1) if current_index is not None else 0,
                seed=seed,
            )
            session.add(episode)
            session.flush()
            cls._append_event(
                session,
                run,
                "episode_started",
                {"target_version": target_version},
                episode_id,
            )
        elif episode.run_id != run_id:
            raise ValueError("episode does not belong to the requested run")
        return run, episode

    def submit_job(self, job: ExecutorJob) -> None:
        with self.sessions.begin() as session:
            existing = session.get(ExecutorJobRecord, str(job.job_id))
            if existing is not None:
                return
            run, _episode = self._ensure_run_episode(
                session, str(job.run_id), str(job.episode_id), job.target_id
            )
            payload = job.model_dump(mode="json")
            session.add(
                ExecutorJobRecord(
                    job_id=str(job.job_id),
                    run_id=str(job.run_id),
                    episode_id=str(job.episode_id),
                    action_id=str(job.action_id),
                    idempotency_key=str(job.idempotency_key),
                    status="queued",
                    payload=payload,
                )
            )
            self._append_event(
                session,
                run,
                (
                    "action_submitted"
                    if job.purpose == "policy"
                    else "evaluator_probe_submitted"
                    if job.purpose == "evaluator"
                    else "manual_action_submitted"
                ),
                {
                    "purpose": job.purpose,
                    "job_id": str(job.job_id),
                    "decision_id": str(job.decision_id),
                    "action_id": str(job.action_id),
                    "request_id": str(job.request_id),
                    "action": job.action.model_dump(mode="json"),
                    "action_catalog_schema_id": "red.action_catalog.v2",
                },
                str(job.episode_id),
            )

    def record_decision(self, decision: Any, observation: Any) -> None:
        """Commit a role decision and its exact policy-visible observation atomically."""
        decision_data = decision.model_dump(mode="json")
        observation_data = observation.model_dump(mode="json")
        with self.sessions.begin() as session:
            existing = session.get(AgentDecisionRecord, str(decision.decision_id))
            if existing is not None:
                if (
                    existing.run_id != str(decision.run_id)
                    or existing.episode_id != str(decision.episode_id)
                    or existing.step_id != decision.step_id
                    or existing.role != decision.role
                    or existing.policy_id != decision.policy_id
                    or existing.observation_schema_id != decision.observation_schema_id
                    or existing.observation != observation_data
                    or existing.legal_action_mask != list(decision.legal_action_mask)
                    or existing.selected_action != decision_data["selected_action"]
                    or existing.policy_sample != decision_data.get("policy_sample")
                    or existing.inference_ms != decision.inference_ms
                ):
                    raise ValueError("decision ID was already used for different decision content")
                return
            step_record = session.scalar(
                select(AgentDecisionRecord).where(
                    AgentDecisionRecord.episode_id == str(decision.episode_id),
                    AgentDecisionRecord.step_id == decision.step_id,
                    AgentDecisionRecord.role == decision.role,
                )
            )
            if step_record is not None:
                raise ValueError("a decision for this role and episode step is already recorded")
            run, _episode = self._ensure_run_episode(
                session, str(decision.run_id), str(decision.episode_id), "violet_ledger_v1"
            )
            session.add(
                AgentDecisionRecord(
                    decision_id=str(decision.decision_id),
                    run_id=str(decision.run_id),
                    episode_id=str(decision.episode_id),
                    step_id=decision.step_id,
                    role=decision.role,
                    policy_id=decision.policy_id,
                    observation_schema_id=decision.observation_schema_id,
                    observation=observation_data,
                    legal_action_mask=list(decision.legal_action_mask),
                    selected_action=decision_data["selected_action"],
                    policy_sample=decision_data.get("policy_sample"),
                    inference_ms=decision.inference_ms,
                )
            )
            self._append_event(
                session,
                run,
                "agent_decision_recorded",
                {"decision": decision_data, "observation": observation_data},
                str(decision.episode_id),
            )

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            record = session.get(AgentDecisionRecord, decision_id)
            if record is None:
                return None
            return {
                "decision_id": record.decision_id,
                "run_id": record.run_id,
                "episode_id": record.episode_id,
                "step_id": record.step_id,
                "role": record.role,
                "policy_id": record.policy_id,
                "observation_schema_id": record.observation_schema_id,
                "observation": record.observation,
                "legal_action_mask": record.legal_action_mask,
                "selected_action": record.selected_action,
                "policy_sample": record.policy_sample,
                "inference_ms": record.inference_ms,
            }

    def get_reward(
        self, run_id: str, episode_id: str, step_id: int, role: str
    ) -> dict[str, Any] | None:
        with self.sessions() as session:
            record = session.scalar(
                select(RangeRewardRecord).where(
                    RangeRewardRecord.run_id == run_id,
                    RangeRewardRecord.episode_id == episode_id,
                    RangeRewardRecord.step_id == step_id,
                    RangeRewardRecord.role == role,
                )
            )
            return record.payload if record is not None else None

    def record_transition(
        self,
        run_id: str,
        episode_id: str,
        decision_id: str,
        payload: dict[str, Any],
    ) -> None:
        with self.sessions.begin() as session:
            decision = session.get(AgentDecisionRecord, decision_id)
            episode = session.get(Episode, episode_id)
            run = self._locked_run(session, run_id)
            if (
                run is None
                or episode is None
                or episode.run_id != run_id
                or decision is None
                or decision.run_id != run_id
                or decision.episode_id != episode_id
            ):
                raise ValueError("transition decision does not belong to this run and episode")
            reward = session.scalar(
                select(RangeRewardRecord).where(
                    RangeRewardRecord.episode_id == episode_id,
                    RangeRewardRecord.step_id == decision.step_id,
                    RangeRewardRecord.role == decision.role,
                )
            )
            if reward is None or reward.run_id != run_id or reward.payload != payload.get("reward"):
                raise ValueError("transition reward is missing or mismatched")
            existing = session.get(AgentTransitionRecord, decision_id)
            if existing is not None:
                if existing.payload != payload:
                    raise ValueError("transition already recorded with different content")
                return
            transition = AgentTransitionRecord(
                decision_id=decision_id,
                run_id=run_id,
                episode_id=episode_id,
                step_id=decision.step_id,
                role=decision.role,
                accepted=bool(payload["accepted"]),
                execution_status=str(payload["execution_status"]),
                mode=str(payload["mode"]),
                payload=payload,
            )
            session.add(transition)
            self._append_event(
                session,
                run,
                "agent_transition_recorded",
                payload,
                episode_id,
            )

    def episode_exists(self, run_id: str, episode_id: str) -> bool:
        with self.sessions() as session:
            episode = session.get(Episode, episode_id)
            return bool(episode and episode.run_id == run_id)

    def finish_episode(
        self, run_id: str, episode_id: str, status: str, reason: str | None = None
    ) -> None:
        if status not in {"completed", "incomplete", "failed"}:
            raise ValueError("invalid episode status")
        with self.sessions.begin() as session:
            run = self._locked_run(session, run_id)
            episode = session.get(Episode, episode_id, with_for_update=True)
            if episode is None or run is None or episode.run_id != run_id:
                raise ValueError("range episode does not belong to this run")
            if episode.status in {"completed", "incomplete", "failed"}:
                if episode.status != status:
                    raise ValueError("episode already reached a terminal status")
                return
            episode.status = status
            self._append_event(
                session,
                run,
                "episode_finished",
                {"episode_id": episode_id, "status": status, "reason": reason},
                episode_id,
            )

    def record_detection(self, run_id: str, episode_id: str, detection: dict[str, Any]) -> bool:
        detection_id = str(uuid4())
        source_id = str(detection["source_target_event_id"])
        with self.sessions.begin() as session:
            run = self._locked_run(session, run_id)
            episode = session.get(Episode, episode_id)
            evidence = session.get(TargetEvidence, source_id)
            if (
                run is None
                or episode is None
                or episode.run_id != run_id
                or evidence is None
                or evidence.run_id != run_id
                or evidence.episode_id != episode_id
            ):
                raise ValueError("detection run or episode does not exist")
            existing = session.scalar(
                select(RangeDetectionRecord).where(
                    RangeDetectionRecord.source_target_event_id == source_id,
                    RangeDetectionRecord.detector_version == str(detection["detector_version"]),
                )
            )
            if existing is not None:
                return False
            session.add(
                RangeDetectionRecord(
                    detection_id=detection_id,
                    run_id=run_id,
                    episode_id=episode_id,
                    source_target_event_id=source_id,
                    detector_version=str(detection["detector_version"]),
                    payload=detection,
                )
            )
            self._append_event(session, run, "detection_suspected", detection, episode_id)
            return True

    def record_reward(
        self, run_id: str, episode_id: str, step_id: int, reward: dict[str, Any]
    ) -> None:
        role = str(reward["role"])
        with self.sessions.begin() as session:
            run = self._locked_run(session, run_id)
            episode = session.get(Episode, episode_id)
            if run is None or episode is None or episode.run_id != run_id:
                raise ValueError("reward run or episode does not exist")
            existing = session.scalar(
                select(RangeRewardRecord).where(
                    RangeRewardRecord.episode_id == episode_id,
                    RangeRewardRecord.step_id == step_id,
                    RangeRewardRecord.role == role,
                )
            )
            if existing is not None:
                if existing.payload != reward:
                    raise ValueError("reward already recorded with different attribution")
                return
            session.add(
                RangeRewardRecord(
                    reward_id=str(uuid4()),
                    run_id=run_id,
                    episode_id=episode_id,
                    step_id=step_id,
                    role=role,
                    payload=reward,
                )
            )
            self._append_event(session, run, "reward_recorded", reward, episode_id)

    def create_training_run(
        self, training_run_id: str, role: str, mode: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        with self.sessions.begin() as session:
            existing = session.get(TrainingRunRecord, training_run_id)
            if existing is not None:
                if existing.role != role or existing.mode != mode or existing.config != config:
                    raise ValueError("training run ID was already used for different configuration")
                return self._training_run_payload(existing)
            record = TrainingRunRecord(
                training_run_id=training_run_id, role=role, mode=mode, config=config
            )
            session.add(record)
            run = self._locked_run(session, training_run_id)
            if run is None:
                run = Run(
                    run_id=training_run_id,
                    mode="real_range_training",
                    target_version=str(config.get("target_version", "violet_ledger_v1")),
                    config=config,
                )
                session.add(run)
                session.flush()
                self._append_event(
                    session,
                    run,
                    "run_created",
                    {
                        "mode": "real_range",
                        "target_version": str(config.get("target_version", "violet_ledger_v1")),
                    },
                    None,
                )
            self._append_event(
                session,
                run,
                "training_run_created",
                {"training_run_id": training_run_id, "role": role, "mode": mode, "config": config},
                None,
            )
            session.flush()
            return self._training_run_payload(record)

    @staticmethod
    def _training_run_payload(record: TrainingRunRecord) -> dict[str, Any]:
        return {
            "training_run_id": record.training_run_id,
            "role": record.role,
            "mode": record.mode,
            "status": record.status,
            "config": record.config,
            "worker_id": record.worker_id,
            "stop_requested": record.stop_requested,
            "result": record.result,
            "created_at": EventRepository._timestamp(record.created_at),
            "updated_at": EventRepository._timestamp(record.updated_at),
        }

    def get_training_run(self, training_run_id: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            record = session.get(TrainingRunRecord, training_run_id)
            return self._training_run_payload(record) if record else None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            record = session.get(Run, run_id)
            if record is None:
                return None
            return {
                "run_id": record.run_id,
                "mode": record.mode,
                "target_version": record.target_version,
                "status": record.status,
                "config": record.config,
            }

    def list_training_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.sessions() as session:
            records = session.scalars(
                select(TrainingRunRecord).order_by(TrainingRunRecord.created_at.desc()).limit(limit)
            ).all()
            return [self._training_run_payload(record) for record in records]

    def training_updates(
        self, training_run_id: str, role: str | None = None
    ) -> list[dict[str, Any]]:
        with self.sessions() as session:
            statement = select(TrainingUpdateRecord).where(
                TrainingUpdateRecord.training_run_id == training_run_id
            )
            if role is not None:
                statement = statement.where(TrainingUpdateRecord.role == role)
            records = session.scalars(
                statement.order_by(TrainingUpdateRecord.update_index, TrainingUpdateRecord.role)
            ).all()
            return [
                {
                    "training_run_id": record.training_run_id,
                    "role": record.role,
                    "update_index": record.update_index,
                    "global_step": record.global_step,
                    "metrics": record.metrics,
                    "checkpoint_sha256": record.checkpoint_sha256,
                    "created_at": self._timestamp(record.created_at),
                }
                for record in records
            ]

    def list_episodes(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.sessions() as session:
            records = session.scalars(
                select(Episode)
                .where(Episode.run_id == run_id)
                .order_by(Episode.episode_index.desc())
                .limit(limit)
            ).all()
            return [
                {
                    "episode_id": record.episode_id,
                    "run_id": record.run_id,
                    "episode_index": record.episode_index,
                    "seed": record.seed,
                    "status": record.status,
                    "config": record.config,
                    "created_at": self._timestamp(record.created_at),
                }
                for record in reversed(records)
            ]

    def request_training_stop(self, training_run_id: str) -> dict[str, Any] | None:
        with self.sessions.begin() as session:
            record = session.get(TrainingRunRecord, training_run_id, with_for_update=True)
            if record is None:
                return None
            if record.status in {"queued", "running"}:
                record.stop_requested = True
                record.updated_at = utc_now()
                if record.status == "queued":
                    record.status = "stopped"
                run = self._locked_run(session, training_run_id)
                if run is not None:
                    if record.status == "stopped":
                        run.status = "stopped"
                    self._append_event(
                        session,
                        run,
                        "training_stop_requested",
                        {"training_run_id": training_run_id, "status": record.status},
                        None,
                    )
            return self._training_run_payload(record)

    def claim_training_run(self, worker_id: str) -> dict[str, Any] | None:
        with self.sessions.begin() as session:
            stale_before = utc_now() - timedelta(minutes=2)
            stale_records = session.scalars(
                select(TrainingRunRecord)
                .where(
                    TrainingRunRecord.status == "running",
                    TrainingRunRecord.updated_at < stale_before,
                )
                .with_for_update(skip_locked=True)
            ).all()
            for stale in stale_records:
                run = self._locked_run(session, stale.training_run_id)
                updates = session.scalars(
                    select(TrainingUpdateRecord).where(
                        TrainingUpdateRecord.training_run_id == stale.training_run_id,
                        TrainingUpdateRecord.checkpoint_sha256.is_not(None),
                    )
                ).all()
                # Each recorded digest names the joint Red + Blue checkpoint.
                # One role update is enough to prove that checkpoint was made
                # durable before the worker lease expired.
                resumable_demo = (
                    stale.config.get("scenario_id") == "sqli_lab_v1"
                    and stale.config.get("campaign_label") == "live_demo_v1"
                    and stale.mode == "train"
                    and any(item.checkpoint_sha256 for item in updates)
                )
                if resumable_demo:
                    stale.status = "queued"
                    stale.worker_id = None
                    stale.updated_at = utc_now()
                    if run is not None:
                        self._append_event(
                            session,
                            run,
                            "training_worker_requeued",
                            {
                                "training_run_id": stale.training_run_id,
                                "reason": "checkpoint_resume_available",
                            },
                            None,
                        )
                    continue
                stale.status = "failed"
                stale.result = {"error": "worker_lease_expired", "outcome": "episode_state_unknown"}
                stale.updated_at = utc_now()
                if run is not None:
                    run.status = "failed"
                    self._append_event(
                        session,
                        run,
                        "training_worker_lost",
                        {"training_run_id": stale.training_run_id, "worker_id": stale.worker_id},
                        None,
                    )
            record = session.scalar(
                select(TrainingRunRecord)
                .where(
                    TrainingRunRecord.status == "queued",
                    TrainingRunRecord.stop_requested.is_(False),
                )
                .order_by(TrainingRunRecord.created_at)
                .with_for_update(skip_locked=True)
            )
            if record is None:
                return None
            record.status = "running"
            record.worker_id = worker_id
            record.updated_at = utc_now()
            run = self._locked_run(session, record.training_run_id)
            if run is not None:
                self._append_event(
                    session,
                    run,
                    "training_run_started",
                    {"training_run_id": record.training_run_id, "worker_id": worker_id},
                    None,
                )
            session.flush()
            return self._training_run_payload(record)

    def heartbeat_training_run(self, training_run_id: str, worker_id: str) -> dict[str, Any] | None:
        with self.sessions.begin() as session:
            record = session.get(TrainingRunRecord, training_run_id, with_for_update=True)
            if record is None:
                return None
            if record.worker_id != worker_id or record.status != "running":
                raise ValueError("training run is not owned by this worker")
            record.updated_at = utc_now()
            return self._training_run_payload(record)

    def update_training_run(
        self,
        training_run_id: str,
        worker_id: str,
        status: str,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if status not in {"running", "completed", "stopped", "failed"}:
            raise ValueError("invalid training run status")
        with self.sessions.begin() as session:
            record = session.get(TrainingRunRecord, training_run_id, with_for_update=True)
            if record is None:
                return None
            if record.worker_id != worker_id:
                raise ValueError("training run is owned by another worker")
            if record.stop_requested and status == "completed":
                status = "stopped"
            record.status = status
            record.result = result if result is not None else record.result
            record.updated_at = utc_now()
            run = self._locked_run(session, training_run_id)
            if run is not None:
                run.status = status
                self._append_event(
                    session,
                    run,
                    "training_run_updated",
                    {"training_run_id": training_run_id, "status": status, "result": result or {}},
                    None,
                )
            return self._training_run_payload(record)

    def record_training_update(
        self,
        training_run_id: str,
        role: str,
        update_index: int,
        global_step: int,
        metrics: dict[str, Any],
        checkpoint_sha256: str | None = None,
    ) -> None:
        with self.sessions.begin() as session:
            existing = session.scalar(
                select(TrainingUpdateRecord).where(
                    TrainingUpdateRecord.training_run_id == training_run_id,
                    TrainingUpdateRecord.role == role,
                    TrainingUpdateRecord.update_index == update_index,
                )
            )
            if existing is not None:
                if (
                    existing.metrics != metrics
                    or existing.checkpoint_sha256 != checkpoint_sha256
                    or existing.global_step != global_step
                ):
                    raise ValueError("training update index already exists with different content")
                return
            session.add(
                TrainingUpdateRecord(
                    update_id=str(uuid4()),
                    training_run_id=training_run_id,
                    role=role,
                    update_index=update_index,
                    global_step=global_step,
                    metrics=metrics,
                    checkpoint_sha256=checkpoint_sha256,
                )
            )
            run = self._locked_run(session, training_run_id)
            if run is not None:
                self._append_event(
                    session,
                    run,
                    "training_update_recorded",
                    {
                        "role": role,
                        "update_index": update_index,
                        "global_step": global_step,
                        "metrics": metrics,
                        "checkpoint_sha256": checkpoint_sha256,
                    },
                    None,
                )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            record = session.get(ExecutorJobRecord, job_id)
            if record is None:
                return None
            return {
                "payload": record.payload,
                "status": record.status,
                "result": record.result,
            }

    def get_job_by_idempotency(self, idempotency_key: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            record = session.scalar(
                select(ExecutorJobRecord).where(
                    ExecutorJobRecord.idempotency_key == idempotency_key
                )
            )
            if record is None:
                return None
            return {"payload": record.payload, "status": record.status, "result": record.result}

    def has_target_evidence(
        self,
        target_event_id: str | None,
        request_id: str,
        run_id: str,
        episode_id: str,
        action_id: str,
    ) -> bool:
        if target_event_id is None:
            return False
        with self.sessions() as session:
            record = session.get(TargetEvidence, target_event_id)
            return bool(
                record
                and record.request_id == request_id
                and record.run_id == run_id
                and record.episode_id == episode_id
                and record.action_id == action_id
            )

    def pending_jobs(self) -> list[dict[str, Any]]:
        with self.sessions() as session:
            records = session.scalars(
                select(ExecutorJobRecord).where(ExecutorJobRecord.status == "queued")
            ).all()
            return [record.payload for record in records]

    def save_execution(
        self,
        job: ExecutorJob,
        result: ExecutorResult,
        target_payload: dict[str, Any] | None,
        finding: dict[str, Any] | None,
    ) -> None:
        with self.sessions.begin() as session:
            record = session.get(ExecutorJobRecord, str(job.job_id))
            if record is None or record.result is not None:
                return
            run = self._locked_run(session, str(job.run_id))
            if run is None:
                raise ValueError("run record is missing for executor result")
            if target_payload is not None and result.target_event_id is not None:
                existing_evidence = session.get(TargetEvidence, str(result.target_event_id))
                if existing_evidence is None:
                    occurred_at = _parse_timestamp(target_payload.get("occurred_at"))
                    session.add(
                        TargetEvidence(
                            target_event_id=str(result.target_event_id),
                            run_id=str(job.run_id),
                            episode_id=str(job.episode_id),
                            action_id=str(job.action_id),
                            request_id=str(job.request_id),
                            source="target_access_log",
                            occurred_at=occurred_at,
                            payload=target_payload,
                        )
                    )
                    self._append_event(
                        session,
                        run,
                        "target_evidence_ingested",
                        {
                            "target_event_id": str(result.target_event_id),
                            "request_id": str(job.request_id),
                            "action_id": str(job.action_id),
                            "source": "target_access_log",
                            "evidence": target_payload,
                        },
                        str(job.episode_id),
                    )
            result_payload = result.model_dump(mode="json")
            record.status = result.status
            record.result = result_payload
            record.finished_at = utc_now()
            self._append_event(
                session,
                run,
                "executor_result",
                {
                    "job_id": str(job.job_id),
                    "decision_id": str(job.decision_id),
                    "action_id": str(job.action_id),
                    "request_id": str(job.request_id),
                    "result": result_payload,
                    "finding": finding,
                },
                str(job.episode_id),
            )

    def submit_control(self, command: ControlCommand) -> None:
        if command.run_id is None or command.episode_id is None:
            raise ValueError("control command requires run and episode IDs")
        with self.sessions.begin() as session:
            existing = session.get(ControlRecord, str(command.command_id))
            if existing is not None:
                if existing.payload != command.model_dump(mode="json"):
                    raise ValueError("control command ID was reused for different content")
                return
            run, episode = self._ensure_run_episode(
                session,
                str(command.run_id),
                str(command.episode_id),
                "violet_ledger_v1",
                seed=command.seed,
            )
            if command.command == "RESET":
                episode.seed = command.seed
            payload = command.model_dump(mode="json")
            session.add(
                ControlRecord(
                    command_id=str(command.command_id),
                    run_id=str(command.run_id),
                    episode_id=str(command.episode_id),
                    status="queued",
                    payload=payload,
                )
            )
            self._append_event(
                session,
                run,
                "target_control_submitted",
                {"command_id": str(command.command_id), "command": payload},
                str(command.episode_id),
            )

    def save_control_ack(self, acknowledgement: ControlAcknowledgement) -> None:
        with self.sessions.begin() as session:
            record = session.get(ControlRecord, str(acknowledgement.command_id))
            if record is None or record.acknowledgement is not None:
                return
            run = self._locked_run(session, record.run_id)
            if run is None:
                return
            payload = acknowledgement.model_dump(mode="json")
            record.status = acknowledgement.status
            record.acknowledgement = payload
            self._append_event(
                session,
                run,
                "target_control_acknowledged",
                payload,
                record.episode_id,
            )

    def latest_fixture_seed(self) -> int:
        with self.sessions() as session:
            records = session.scalars(
                select(ControlRecord)
                .where(ControlRecord.acknowledgement.is_not(None))
                .order_by(ControlRecord.created_at.desc())
            ).all()
            for record in records:
                if record.payload.get("command") == "RESET" and record.acknowledgement:
                    seed = record.acknowledgement.get("fixture_seed", 0)
                    if isinstance(seed, int):
                        return seed
                    try:
                        return int(str(seed))
                    except ValueError:
                        continue
        return 0

    def fixture_seed(self, run_id: str, episode_id: str) -> int:
        """Return the fixture seed recorded for this episode, without cross-run leakage."""
        with self.sessions() as session:
            episode = session.get(Episode, episode_id)
            if episode is None or episode.run_id != run_id:
                return 0
            if episode.seed is not None:
                return episode.seed
            return 0

    def get_control(self, command_id: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            record = session.get(ControlRecord, command_id)
            if record is None:
                return None
            return {
                "payload": record.payload,
                "status": record.status,
                "acknowledgement": record.acknowledgement,
            }

    def pending_controls(self) -> list[dict[str, Any]]:
        with self.sessions() as session:
            records = session.scalars(
                select(ControlRecord).where(ControlRecord.acknowledgement.is_(None))
            ).all()
            return [record.payload for record in records]

    def latest_target_control_state(self) -> dict[str, Any]:
        with self.sessions() as session:
            records = session.scalars(
                select(ControlRecord)
                .where(ControlRecord.acknowledgement.is_not(None))
                .order_by(ControlRecord.created_at.desc())
                .limit(1)
            ).all()
            if not records or records[0].acknowledgement is None:
                return {"state_version": 0, "controls": {}}
            return {
                "state_version": records[0].acknowledgement.get("state_version", 0),
                "controls": records[0].acknowledgement.get("controls", {}),
            }

    def events(self, run_id: str, after_seq: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self.sessions() as session:
            records = session.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == run_id, RunEvent.seq > after_seq)
                .order_by(RunEvent.seq)
                .limit(limit)
            ).all()
            return [
                {
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "episode_id": event.episode_id,
                    "seq": event.seq,
                    "event_type": event.event_type,
                    "occurred_at": event.occurred_at.isoformat(),
                    "payload": event.payload,
                    "previous_hash": event.previous_hash,
                    "event_hash": event.event_hash,
                }
                for event in records
            ]

    def lab_summary_events(
        self, run_id: str, after_seq: int = 0, limit: int = 10000
    ) -> list[dict[str, Any]]:
        """Return compact per-episode learning and performance records, not full traces."""
        event_types = {"sqli_episode_summary", "sqli_ppo_update", "sqli_episode_outcome"}
        with self.sessions() as session:
            records = session.scalars(
                select(RunEvent)
                .where(
                    RunEvent.run_id == run_id,
                    RunEvent.seq > after_seq,
                    RunEvent.event_type.in_(event_types),
                )
                .order_by(RunEvent.seq)
                .limit(limit)
            ).all()
            return [
                {
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "episode_id": event.episode_id,
                    "seq": event.seq,
                    "event_type": event.event_type,
                    "occurred_at": self._timestamp(event.occurred_at),
                    "payload": event.payload,
                    "previous_hash": event.previous_hash,
                    "event_hash": event.event_hash,
                }
                for event in records
            ]

    def episode_events(
        self, run_id: str, episode_id: str, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Return this episode's latest events in sequence order, regardless of run length."""
        with self.sessions() as session:
            records = list(
                session.scalars(
                    select(RunEvent)
                    .where(RunEvent.run_id == run_id, RunEvent.episode_id == episode_id)
                    .order_by(RunEvent.seq.desc())
                    .limit(limit)
                ).all()
            )
            records.reverse()
            return [
                {
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "episode_id": event.episode_id,
                    "seq": event.seq,
                    "event_type": event.event_type,
                    "occurred_at": event.occurred_at.isoformat(),
                    "payload": event.payload,
                    "previous_hash": event.previous_hash,
                    "event_hash": event.event_hash,
                }
                for event in records
            ]

    def episode_events_page(
        self, run_id: str, episode_id: str, after_seq: int = 0, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self.sessions() as session:
            records = session.scalars(
                select(RunEvent)
                .where(
                    RunEvent.run_id == run_id,
                    RunEvent.episode_id == episode_id,
                    RunEvent.seq > after_seq,
                )
                .order_by(RunEvent.seq)
                .limit(limit)
            ).all()
            return [
                {
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "episode_id": event.episode_id,
                    "seq": event.seq,
                    "event_type": event.event_type,
                    "occurred_at": self._timestamp(event.occurred_at),
                    "payload": event.payload,
                    "previous_hash": event.previous_hash,
                    "event_hash": event.event_hash,
                }
                for event in records
            ]

    def event_seq(self, run_id: str, event_type: str, field: str, value: str) -> int | None:
        with self.sessions() as session:
            records = session.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == run_id, RunEvent.event_type == event_type)
                .order_by(RunEvent.seq)
            ).all()
            for event in records:
                if event.payload.get(field) == value:
                    return event.seq
        return None

    def record_control_verification(
        self, command_id: str, verification: dict[str, Any]
    ) -> dict[str, Any]:
        with self.sessions.begin() as session:
            control = session.get(ControlRecord, command_id)
            if control is None or control.acknowledgement is None:
                raise ValueError("control acknowledgement is missing")
            run = session.scalar(select(Run).where(Run.run_id == control.run_id).with_for_update())
            if run is None:
                raise ValueError("run record is missing for control verification")
            for event in session.scalars(
                select(RunEvent)
                .where(
                    RunEvent.run_id == control.run_id,
                    RunEvent.event_type == "control_verification",
                )
                .order_by(RunEvent.seq)
            ).all():
                existing = event.payload
                if existing.get("command_id") == command_id and existing.get(
                    "evidence"
                ) == verification.get("evidence"):
                    return existing
            self._append_event(
                session,
                run,
                "control_verification",
                verification,
                control.episode_id,
            )
            return verification


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return utc_now()
