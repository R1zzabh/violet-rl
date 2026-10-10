"""Transactional repository for run events, executor outbox, and evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from violet.persistence.models import (
    Base,
    ControlRecord,
    Episode,
    ExecutorJobRecord,
    Run,
    RunEvent,
    TargetEvidence,
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
        Base.metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    @staticmethod
    def _append_event(
        session: Session,
        run: Run,
        event_type: str,
        payload: dict[str, Any],
        episode_id: str | None,
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
            event_id=str(uuid4()),
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
            episode = Episode(episode_id=episode_id, run_id=run_id, seed=seed)
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
                "action_submitted",
                {
                    "job_id": str(job.job_id),
                    "decision_id": str(job.decision_id),
                    "action_id": str(job.action_id),
                    "request_id": str(job.request_id),
                    "action": job.action.model_dump(mode="json"),
                    "observation_schema_id": "red.range.v2",
                },
                str(job.episode_id),
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
            run = session.get(Run, str(job.run_id))
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
            run = session.get(Run, record.run_id)
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
                    return int(record.acknowledgement.get("fixture_seed", 0))
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
