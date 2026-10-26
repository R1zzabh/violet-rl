"""CPU PPO worker for the real Violet Ledger range.

The worker has API access only. It never imports target or executor transports;
all interactions pass through the typed RangeEpisodeRunner and VIOLET API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID, uuid4

import httpx
import numpy as np
import torch
from torch.distributions import Categorical

from violet.algorithms.ppo import ActorCritic, RolloutBuffer, Transition
from violet.range.contracts import RangeObservation
from violet.range.episode import RangeEpisodeRunner
from violet.range.observations import (
    BLUE_ACTION_CATALOG_V2,
    FEATURE_ENCODING_SCHEMA_ID,
    RED_ACTION_CATALOG_V2,
    encode_observation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

DEVICE = torch.device("cpu")
API_URL = os.getenv("VIOLET_API_URL", "http://backend:8000").rstrip("/")
TOKEN = os.getenv("VIOLET_TRAINER_TOKEN", "")
WORKER_ID = os.getenv("VIOLET_WORKER_ID", "range-trainer-1")
ARTIFACT_ROOT = Path(os.getenv("VIOLET_RUNS_DIR", "/app/runs"))


def _find_session_aliases(value: Any) -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(value, dict):
        principal = value.get("principal_ref")
        alias = value.get("session_alias")
        if principal in {"alice", "bob"} and isinstance(alias, str):
            found[str(principal)] = alias
        for child in value.values():
            found.update(_find_session_aliases(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_find_session_aliases(child))
    return found


def _decode_action(role: str, action_index: int, observation: RangeObservation) -> dict[str, Any]:
    if role == "red":
        catalog = RED_ACTION_CATALOG_V2
        entry = catalog[action_index]
        if entry.startswith("DISCOVER_ROUTE:"):
            return {"operation": "DISCOVER_ROUTE", "route_candidate_id": entry.split(":", 1)[1]}
        if entry.startswith("AUTH_TEST:"):
            return {"operation": "AUTH_TEST", "credential_ref": entry.split(":", 1)[1]}
        if entry.startswith("CHECK_DOCUMENT_ACCESS:"):
            document_ref = entry.split(":", 1)[1]
            aliases = _find_session_aliases(observation.recent_events)
            # A typed catalog choice names a fixed fixture resource. The session
            # alias is selected only from the policy-visible successful logins.
            principal = "bob" if document_ref == "doc_alice" else "alice"
            alias = aliases.get(principal) or next(iter(aliases.values()), None)
            return {
                "operation": "CHECK_DOCUMENT_ACCESS",
                "document_ref": document_ref,
                "session_alias": alias,
            }
        if entry == "CHECK_TARGET_HEALTH":
            return {"operation": "CHECK_TARGET_HEALTH"}
        return {"operation": "WAIT"}
    entry = BLUE_ACTION_CATALOG_V2[action_index]
    if ":" in entry:
        operation, principal = entry.split(":", 1)
        return {"operation": operation, "principal_ref": principal}
    return {"operation": entry}


class PPOCatalogPolicy:
    """Masked categorical policy adapted to the bounded Red or Blue catalog."""

    def __init__(self, role: str, model: ActorCritic, *, stochastic: bool, seed: int):
        self.role = role
        self.model = model
        self.stochastic = stochastic
        self.rng = np.random.default_rng(seed)
        self.sample: dict[str, Any] | None = None
        self.policy_version = 0

    def __call__(self, observation: RangeObservation) -> dict[str, Any]:
        encoded = np.asarray(encode_observation(observation), dtype=np.float32)
        mask = np.asarray(observation.legal_action_mask, dtype=bool)
        if not mask.any():
            raise ValueError("range observation masks every action")
        started = time.perf_counter()
        obs_tensor = torch.as_tensor(encoded, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model.forward(obs_tensor)
            logits = logits.masked_fill(~mask_tensor, float("-inf"))
            distribution = Categorical(logits=logits)
            action_tensor = (
                distribution.sample()  # type: ignore[no-untyped-call]
                if self.stochastic
                else torch.argmax(logits, dim=-1)
            )
            log_prob = distribution.log_prob(action_tensor)  # type: ignore[no-untyped-call]
        action_index = int(action_tensor.item())
        self.sample = {
            "encoding_schema_id": FEATURE_ENCODING_SCHEMA_ID,
            "observation": encoded.tolist(),
            "mask": mask.tolist(),
            "action": action_index,
            "action_distribution": distribution.probs.squeeze(0).cpu().tolist(),
            "selected_action_probability": float(distribution.probs[0, action_index].item()),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
            "entropy": float(distribution.entropy().item()),  # type: ignore[no-untyped-call]
            "policy_version": self.policy_version,
            "mode": "train" if self.stochastic else "evaluate",
            "inference_ms": (time.perf_counter() - started) * 1000,
        }
        action = _decode_action(self.role, action_index, observation)
        if (
            self.role == "red"
            and action.get("operation") == "CHECK_DOCUMENT_ACCESS"
            and not action.get("session_alias")
        ):
            raise RuntimeError("legal document access action has no visible session alias")
        return action

    def consume_sample(self) -> dict[str, Any] | None:
        sample, self.sample = self.sample, None
        return sample

    def estimate_value(self, observation: RangeObservation) -> float:
        encoded = torch.as_tensor(
            np.asarray(encode_observation(observation), dtype=np.float32),
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)
        with torch.no_grad():
            _logits, value = self.model.forward(encoded)
        return float(value.item())


def _baseline(role: str, mode: str, seed: int) -> Callable[[RangeObservation], dict[str, Any]]:
    rng = random.Random(seed)

    def choose(observation: RangeObservation) -> dict[str, Any]:
        if mode == "random":
            legal = [i for i, allowed in enumerate(observation.legal_action_mask) if allowed]
            return _decode_action(role, rng.choice(legal), observation)
        if role == "red":
            aliases = _find_session_aliases(observation.recent_events)
            if "alice" in aliases:
                document_action = RED_ACTION_CATALOG_V2.index("CHECK_DOCUMENT_ACCESS:doc_bob")
                if not observation.legal_action_mask[document_action]:
                    return {"operation": "WAIT"}
                return {
                    "operation": "CHECK_DOCUMENT_ACCESS",
                    "document_ref": "doc_bob",
                    "session_alias": aliases["alice"],
                }
            return {"operation": "AUTH_TEST", "credential_ref": "synthetic_alice"}
        has_document_request = any(
            isinstance(event, dict) and "CHECK_DOCUMENT_ACCESS" in json.dumps(event)
            for event in observation.recent_events
        )
        return {
            "operation": "HARDEN_DOCUMENT_OWNERSHIP"
            if has_document_request and not observation.active_controls.get("ownership_hardened")
            else "MONITOR"
        }

    return choose


def _ppo_update(
    model: ActorCritic, optimizer: torch.optim.Optimizer, transitions: list[Transition], lr: float
) -> dict[str, float]:
    if not transitions:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    buffer = RolloutBuffer()
    for transition in transitions:
        buffer.add(transition)
    buffer.finalize(0.0, 0.99, 0.95)
    batch = buffer.tensors(DEVICE)
    logits, values = model.forward(batch["obs"])
    logits = logits.masked_fill(~batch["mask"], float("-inf"))
    distribution = Categorical(logits=logits)
    new_log_probs = distribution.log_prob(batch["actions"])  # type: ignore[no-untyped-call]
    ratios = torch.exp(new_log_probs - batch["old_log_probs"])
    clipped = torch.clamp(ratios, 0.8, 1.2) * batch["advantages"]
    policy_loss = -torch.minimum(ratios * batch["advantages"], clipped).mean()
    value_loss = torch.nn.functional.mse_loss(values, batch["returns"])
    entropy = distribution.entropy().mean()  # type: ignore[no-untyped-call]
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    advantages = batch["advantages"].detach()
    log_ratio = new_log_probs.detach() - batch["old_log_probs"]
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "clip_fraction": float(((ratios.detach() - 1.0).abs() > 0.2).float().mean().item()),
        "approx_kl": float((((ratios.detach() - 1.0) - log_ratio).mean()).item()),
        "gradient_norm": float(gradient_norm.item()),
        "optimizer_steps": 1.0,
        "sample_count": float(len(transitions)),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()),
        "advantage_min": float(advantages.min().item()),
        "advantage_max": float(advantages.max().item()),
    }


def _episode_reward(trace: list[dict[str, Any]], role: str) -> float:
    return sum(float(step.get(f"{role}_reward", {}).get("total", 0.0)) for step in trace)


def _episode_security_metrics(trace: list[dict[str, Any]]) -> dict[str, Any]:
    unauthorized = 0.0
    prevented = 0.0
    false_alerts = 0.0
    denied_legitimate = 0.0
    owner_probes = 0
    owner_reads_served = 0
    for step in trace:
        red_reward = step.get("red_reward", {})
        red_components = red_reward.get("components", {}) if isinstance(red_reward, dict) else {}
        unauthorized = max(
            unauthorized, float(red_components.get("confirmed_unauthorized_read", 0.0))
        )
        blue_reward = step.get("blue_reward", {})
        blue_components = blue_reward.get("components", {}) if isinstance(blue_reward, dict) else {}
        prevented = max(prevented, float(blue_components.get("verified_prevention", 0.0)))
        false_alerts += float(blue_components.get("false_alert", 0.0) < 0.0)
        denied_legitimate += float(blue_components.get("denied_legitimate_request", 0.0) < 0.0)
        blue = step.get("blue", {})
        verification = blue.get("control_verification") if isinstance(blue, dict) else None
        checks = verification.get("checks", {}) if isinstance(verification, dict) else {}
        for check in ("alice_owner_read_served", "bob_owner_read_served"):
            if isinstance(checks.get(check), bool):
                owner_probes += 1
                owner_reads_served += int(checks[check])
    availability = owner_reads_served / owner_probes if owner_probes else None
    return {
        "confirmed_unauthorized_read": unauthorized,
        "verified_prevention": prevented,
        "net_security_score": prevented - unauthorized,
        "legitimate_owner_reads_served": owner_reads_served,
        "legitimate_owner_read_probes": owner_probes,
        "legitimate_owner_access_rate": availability,
        "false_alerts": false_alerts,
        "denied_legitimate_requests": denied_legitimate,
    }


def _paired_bootstrap(
    candidate: list[float], baseline: list[float], seed: int
) -> tuple[float, float, float]:
    if not candidate or len(candidate) != len(baseline):
        raise ValueError("paired evaluation requires equally sized non-empty outcomes")
    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(5000, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


async def _run_joint_job(client: httpx.AsyncClient, job: dict[str, Any]) -> dict[str, Any]:
    """Train/evaluate independent Red and Blue policies on the same episodes."""
    training_id = str(job["training_run_id"])
    config = job["config"]
    mode = str(job["mode"])
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    catalogs = {"red": RED_ACTION_CATALOG_V2, "blue": BLUE_ACTION_CATALOG_V2}
    models: dict[str, ActorCritic] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    policies: dict[str, PPOCatalogPolicy] = {}
    checkpoint_digest: str | None = None
    for index, role in enumerate(("red", "blue")):
        sample_observation = RangeObservation(
            schema_id=cast(
                "Literal['red.range_observation.v2', 'blue.range_observation.v2']",
                f"{role}.range_observation.v2",
            ),
            run_id=UUID(training_id),
            episode_id=uuid4(),
            step_id=0,
            remaining_requests=40,
            legal_action_mask=tuple(True for _ in catalogs[role]),
        )
        model = ActorCritic(len(encode_observation(sample_observation)), len(catalogs[role])).to(
            DEVICE
        )
        models[role] = model
        optimizers[role] = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
        policies[role] = PPOCatalogPolicy(
            role,
            model,
            stochastic=mode == "train",
            seed=seed + index,
        )

    checkpoint_run_id = config.get("checkpoint_run_id")
    if checkpoint_run_id:
        source = await client.get(f"/range/training-runs/{checkpoint_run_id}")
        source.raise_for_status()
        source_run = source.json()
        if source_run.get("status") != "completed" or source_run.get("role") != "joint":
            raise ValueError("joint checkpoint source is incomplete or incompatible")
        source_checkpoint_path = (
            ARTIFACT_ROOT / str(checkpoint_run_id) / "checkpoints" / "joint" / "final.pt"
        )
        checkpoint_digest = hashlib.sha256(source_checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_digest != (source_run.get("result") or {}).get("checkpoint_sha256"):
            raise ValueError("joint checkpoint digest does not match its training record")
        torch_version_type = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
        safe_globals = [torch_version_type] if torch_version_type is not None else []
        with torch.serialization.safe_globals(safe_globals):
            payload = torch.load(source_checkpoint_path, map_location=DEVICE, weights_only=True)
        if (
            payload.get("format") != "violet.range.joint_ppo.v1"
            or payload.get("role") != "joint"
            or payload.get("feature_encoding_schema_id") != FEATURE_ENCODING_SCHEMA_ID
            or payload.get("observation_schema_ids")
            != {role: f"{role}.range_observation.v2" for role in catalogs}
            or payload.get("action_catalogs")
            != {role: list(catalog) for role, catalog in catalogs.items()}
            or payload.get("target_version") != "violet_ledger_v1"
        ):
            raise ValueError("joint checkpoint is incompatible with current range schemas")
        for role in catalogs:
            models[role].load_state_dict(payload["model_state_dicts"][role], strict=True)
            policies[role].policy_version = int(payload["policy_versions"][role])
            if mode == "train":
                optimizers[role].load_state_dict(payload["optimizer_state_dicts"][role])

    run_dir = ARTIFACT_ROOT / training_id
    config_dir = run_dir / "config"
    metrics_dir = run_dir / "metrics"
    checkpoint_dir = run_dir / "checkpoints" / "joint"
    evaluation_dir = run_dir / "evaluation"
    for directory in (config_dir, metrics_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text(
        json.dumps(
            {
                "training_run_id": training_id,
                "role": "joint",
                "mode": mode,
                "seed": seed,
                "target_version": "violet_ledger_v1",
                "feature_encoding_schema_id": FEATURE_ENCODING_SCHEMA_ID,
                "observation_schema_ids": {
                    role: f"{role}.range_observation.v2" for role in catalogs
                },
                "action_catalog_schema_ids": {
                    role: f"{role}.action_catalog.v2" for role in catalogs
                },
                "git_commit": os.getenv("VIOLET_GIT_COMMIT", "unknown"),
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "torch_version": str(torch.__version__),
                "config": config,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    def save_checkpoint(episode_count: int, global_steps: dict[str, int]) -> tuple[Path, str]:
        path = checkpoint_dir / "final.pt"
        torch.save(
            {
                "format": "violet.range.joint_ppo.v1",
                "role": "joint",
                "feature_encoding_schema_id": FEATURE_ENCODING_SCHEMA_ID,
                "observation_schema_ids": {
                    role: f"{role}.range_observation.v2" for role in catalogs
                },
                "action_catalogs": {role: list(catalog) for role, catalog in catalogs.items()},
                "target_version": "violet_ledger_v1",
                "git_commit": os.getenv("VIOLET_GIT_COMMIT", "unknown"),
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "torch_version": str(torch.__version__),
                "seed": seed,
                "episode_count": episode_count,
                "global_steps": global_steps,
                "policy_versions": {role: policies[role].policy_version for role in catalogs},
                "model_state_dicts": {role: models[role].state_dict() for role in catalogs},
                "optimizer_state_dicts": {role: optimizers[role].state_dict() for role in catalogs},
            },
            path,
        )
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    episode_summaries: list[dict[str, Any]] = []
    candidate_scores: list[float] = []
    baseline_scores: list[float] = []
    candidate_security: list[dict[str, Any]] = []
    baseline_security: list[dict[str, Any]] = []
    global_steps = {"red": 0, "blue": 0}
    update_index = 0
    last_metrics: dict[str, dict[str, float]] = {}
    requested_episodes = int(config["episodes"])
    research_run = config.get("run_kind") == "research"
    max_attempts = requested_episodes * 2 if research_run else requested_episodes
    episode_index = 0
    completed_training_episodes = 0
    while episode_index < max_attempts and (
        (mode == "train" and (not research_run or completed_training_episodes < requested_episodes))
        or (mode == "evaluate" and (not research_run or len(candidate_scores) < requested_episodes))
    ):
        if await _stop_requested(client, training_id):
            break
        episode_id = uuid4()
        candidate_runner = RangeEpisodeRunner(
            API_URL,
            UUID(training_id),
            episode_id,
            red_policy=policies["red"],
            blue_policy=policies["blue"],
            red_policy_id=f"red-ppo-{training_id}",
            blue_policy_id=f"blue-ppo-{training_id}",
            request_timeout=5.0,
            stop_check=lambda: _stop_requested(client, training_id),
        )
        candidate_result = await candidate_runner.run(
            seed=seed + episode_index,
            max_steps=int(config["max_steps"]),
        )
        trace = candidate_result.get("trace", [])
        score = _episode_reward(trace, "red") + _episode_reward(trace, "blue")
        episode_record: dict[str, Any] = {
            "episode_id": str(episode_id),
            "seed": seed + episode_index,
            "status": candidate_result.get("status"),
            "steps": len(trace),
            "red_return": _episode_reward(trace, "red"),
            "blue_return": _episode_reward(trace, "blue"),
            "joint_return": score,
        }
        if candidate_result.get("status") == "completed" and mode == "train":
            completed_training_episodes += 1
            episode_transitions: dict[str, list[Transition]] = {"red": [], "blue": []}
            for role in catalogs:
                for step in trace:
                    sample = step.get(f"{role}_sample")
                    transition_record = step.get(f"{role}_transition")
                    if not isinstance(transition_record, dict):
                        continue
                    if not transition_record.get("accepted"):
                        # Rejected actions do not enter PPO data and end the
                        # preceding accepted trace segment for GAE purposes.
                        if episode_transitions[role]:
                            episode_transitions[role][-1].truncated = True
                        continue
                    if not isinstance(sample, dict):
                        continue
                    reward = transition_record["reward"]
                    episode_transitions[role].append(
                        Transition(
                            obs=np.asarray(sample["observation"], dtype=np.float32),
                            mask=np.asarray(sample["mask"], dtype=bool),
                            action=int(sample["action"]),
                            log_prob=float(sample["log_prob"]),
                            value=float(sample["value"]),
                            reward=float(reward["total"]),
                            terminated=bool(transition_record["terminated"]),
                            truncated=bool(transition_record["truncated"]),
                            next_value=float(transition_record["next_value"]),
                        )
                    )
                if not episode_transitions[role]:
                    raise RuntimeError(f"completed episode has no accepted {role} transitions")

            update_index += 1
            for role in catalogs:
                metrics = _ppo_update(
                    models[role],
                    optimizers[role],
                    episode_transitions[role],
                    float(config["learning_rate"]),
                )
                last_metrics[role] = metrics
                global_steps[role] += len(episode_transitions[role])
                policies[role].policy_version += 1
            _checkpoint_path, checkpoint_digest = save_checkpoint(update_index, global_steps)
            for role in catalogs:
                update = {
                    "worker_id": WORKER_ID,
                    "role": role,
                    "status": "running",
                    "result": {
                        "update_index": update_index,
                        "episode_index": update_index,
                        "global_step": global_steps[role],
                        "metrics": last_metrics[role],
                        "checkpoint_sha256": checkpoint_digest,
                        "policy_version": policies[role].policy_version,
                        "update_boundary": "episode_end",
                    },
                }
                update_response = await client.post(
                    f"/range/training-runs/{training_id}/updates", json=update
                )
                update_response.raise_for_status()
                with (metrics_dir / f"{role}.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "update_index": update_index,
                                "episode_id": str(episode_id),
                                "global_step": global_steps[role],
                                **last_metrics[role],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        elif candidate_result.get("status") == "completed" and mode == "evaluate":
            baseline_episode_id = uuid4()
            baseline_runner = RangeEpisodeRunner(
                API_URL,
                UUID(training_id),
                baseline_episode_id,
                red_policy=_baseline("red", str(config["opponent"]), seed + episode_index),
                blue_policy=_baseline("blue", str(config["opponent"]), seed + episode_index + 1),
                red_policy_id=f"red-{config['opponent']}-baseline-v1",
                blue_policy_id=f"blue-{config['opponent']}-baseline-v1",
                request_timeout=5.0,
                stop_check=lambda: _stop_requested(client, training_id),
            )
            baseline_result = await baseline_runner.run(
                seed=seed + episode_index,
                max_steps=int(config["max_steps"]),
            )
            baseline_trace = baseline_result.get("trace", [])
            baseline_score = _episode_reward(baseline_trace, "red") + _episode_reward(
                baseline_trace, "blue"
            )
            episode_record.update(
                {
                    "baseline_episode_id": str(baseline_episode_id),
                    "baseline_status": baseline_result.get("status"),
                    "baseline_joint_return": baseline_score,
                }
            )
            if baseline_result.get("status") == "completed":
                candidate_scores.append(score)
                baseline_scores.append(baseline_score)
                candidate_security.append(_episode_security_metrics(trace))
                baseline_security.append(_episode_security_metrics(baseline_trace))
            episode_record["candidate_trace"] = trace
            episode_record["baseline_trace"] = baseline_trace
        episode_summaries.append(episode_record)
        with (run_dir / "episodes.jsonl").open("a") as stream:
            artifact_record = {**episode_record}
            if mode == "train":
                artifact_record["trace"] = trace
            stream.write(json.dumps(artifact_record, sort_keys=True) + "\n")
        episode_index += 1

    checkpoint_file: Path | None = None
    if mode == "train" and update_index:
        checkpoint_file, checkpoint_digest = save_checkpoint(update_index, global_steps)
    evaluation_summary = None
    if mode == "evaluate":
        if candidate_scores:
            difference, ci_low, ci_high = _paired_bootstrap(candidate_scores, baseline_scores, seed)
            interval: list[float] | None = [ci_low, ci_high]
            candidate_net = [item["net_security_score"] for item in candidate_security]
            baseline_net = [item["net_security_score"] for item in baseline_security]
            net_difference, net_ci_low, net_ci_high = _paired_bootstrap(
                candidate_net, baseline_net, seed + 1
            )
        else:
            difference, ci_low, interval = 0.0, 0.0, None
            net_difference, net_ci_low, net_ci_high = 0.0, 0.0, 0.0
        source = await client.get(f"/range/training-runs/{checkpoint_run_id}")
        source.raise_for_status()
        source_result = source.json().get("result") or {}
        seed_count = int(source_result.get("independent_training_seeds", 1))
        owner_probes = sum(int(item["legitimate_owner_read_probes"]) for item in candidate_security)
        owner_reads = sum(int(item["legitimate_owner_reads_served"]) for item in candidate_security)
        availability_rate = owner_reads / owner_probes if owner_probes else None
        availability_satisfied = (
            availability_rate == 1.0 if availability_rate is not None else False
        )
        supported = (
            len(candidate_scores) >= 100
            and net_ci_low > 0
            and seed_count >= 5
            and availability_satisfied
        )
        evaluation_summary = {
            "schema_id": "violet.range.evaluation.v2",
            "target_version": "violet_ledger_v1",
            "policy_role": "joint",
            "primary_metric": "paired_net_security_score",
            "baseline": str(config["opponent"]),
            "checkpoint_run_id": checkpoint_run_id,
            "paired_episodes": len(candidate_scores),
            "requested_pairs": int(config["episodes"]),
            "candidate_mean": float(np.mean(candidate_scores)) if candidate_scores else None,
            "baseline_mean": float(np.mean(baseline_scores)) if baseline_scores else None,
            "mean_paired_difference": difference,
            "paired_bootstrap_ci_95": interval,
            "candidate_security_metrics": {
                "confirmed_unauthorized_read_rate": float(
                    np.mean([item["confirmed_unauthorized_read"] for item in candidate_security])
                )
                if candidate_security
                else None,
                "verified_prevention_rate": float(
                    np.mean([item["verified_prevention"] for item in candidate_security])
                )
                if candidate_security
                else None,
                "legitimate_owner_access_rate": availability_rate,
                "legitimate_owner_read_probes": owner_probes,
                "legitimate_owner_reads_served": owner_reads,
                "false_alerts": sum(int(item["false_alerts"]) for item in candidate_security),
                "denied_legitimate_requests": sum(
                    int(item["denied_legitimate_requests"]) for item in candidate_security
                ),
            },
            "baseline_security_metrics": {
                "confirmed_unauthorized_read_rate": float(
                    np.mean([item["confirmed_unauthorized_read"] for item in baseline_security])
                )
                if baseline_security
                else None,
                "verified_prevention_rate": float(
                    np.mean([item["verified_prevention"] for item in baseline_security])
                )
                if baseline_security
                else None,
            },
            "paired_net_security_score_difference": net_difference,
            "paired_net_security_score_ci_95": [net_ci_low, net_ci_high]
            if candidate_scores
            else None,
            "availability_criterion_met": availability_satisfied,
            "independent_training_seeds": seed_count,
            "required_training_seeds": 5,
            "required_held_out_pairs": 100,
            "supported_improvement": supported,
            "claim": "supported improvement" if supported else "no supported improvement",
            "evaluation_seed_start": seed,
            "feature_encoding_schema_id": FEATURE_ENCODING_SCHEMA_ID,
        }
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        episodes_path = evaluation_dir / "episodes.jsonl"
        episodes_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in episode_summaries)
        )
        evaluation_summary["episodes_sha256"] = hashlib.sha256(
            episodes_path.read_bytes()
        ).hexdigest()
        summary_path = evaluation_dir / "summary.json"
        summary_path.write_text(json.dumps(evaluation_summary, sort_keys=True, indent=2) + "\n")
        evaluation_summary["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    if mode == "train" and not update_index:
        raise RuntimeError("no complete joint episode was updated")
    research_criteria_met = not research_run or (
        completed_training_episodes >= requested_episodes
        if mode == "train"
        else len(candidate_scores) >= requested_episodes
    )
    return {
        "episodes": episode_summaries,
        "episode_updates": update_index,
        "completed_training_episodes": completed_training_episodes,
        "independent_training_seeds": 1 if mode == "train" else None,
        "requested_episodes": requested_episodes,
        "research_criteria_met": research_criteria_met,
        "optimizer_metrics": last_metrics,
        "checkpoint": str(checkpoint_file) if checkpoint_file else None,
        "checkpoint_sha256": checkpoint_digest,
        "global_steps": global_steps,
        "evidence_level": "real_target_episode" if episode_summaries else "no_episode_completed",
        "evaluation": evaluation_summary,
    }


async def _run_job(client: httpx.AsyncClient, job: dict[str, Any]) -> dict[str, Any]:
    training_id = str(job["training_run_id"])
    role = str(job["role"])
    if (job.get("config") or {}).get("scenario_id") == "sqli_lab_v1":
        from violet.sqli_lab.training import run_sqli_job

        return await run_sqli_job(client, job)
    if role == "joint":
        return await _run_joint_job(client, job)
    config = job["config"]
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    action_count = len(RED_ACTION_CATALOG_V2 if role == "red" else BLUE_ACTION_CATALOG_V2)
    sample_obs = RangeObservation(
        schema_id=cast(
            "Literal['red.range_observation.v2', 'blue.range_observation.v2']",
            f"{role}.range_observation.v2",
        ),
        run_id=UUID(training_id),
        episode_id=uuid4(),
        step_id=0,
        remaining_requests=40,
        legal_action_mask=tuple(True for _ in range(action_count)),
    )
    obs_dim = len(encode_observation(sample_obs))
    model = ActorCritic(obs_dim, action_count).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    checkpoint_run_id = config.get("checkpoint_run_id")
    if checkpoint_run_id:
        source = await client.get(f"/range/training-runs/{checkpoint_run_id}")
        source.raise_for_status()
        source_run = source.json()
        if source_run.get("status") != "completed" or source_run.get("role") != role:
            raise ValueError("checkpoint source is incomplete or has a different policy role")
        source_checkpoint_path = (
            ARTIFACT_ROOT / str(checkpoint_run_id) / "checkpoints" / role / "final.pt"
        )
        checkpoint_digest = hashlib.sha256(source_checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_digest != (source_run.get("result") or {}).get("checkpoint_sha256"):
            raise ValueError("checkpoint digest does not match its persisted training record")
        # Verify the database-persisted digest before loading. Older checkpoints
        # serialized torch.__version__ as TorchVersion rather than str; allow
        # only that harmless metadata class in the restricted unpickler.
        torch_version_type = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
        safe_globals = [torch_version_type] if torch_version_type is not None else []
        with torch.serialization.safe_globals(safe_globals):
            payload = torch.load(source_checkpoint_path, map_location=DEVICE, weights_only=True)
        if (
            payload.get("format") != "violet.range.ppo.v2"
            or payload.get("role") != role
            or payload.get("observation_schema_id") != f"{role}.range_observation.v2"
            or payload.get("action_catalog")
            != list(RED_ACTION_CATALOG_V2 if role == "red" else BLUE_ACTION_CATALOG_V2)
            or payload.get("target_version") != "violet_ledger_v1"
        ):
            raise ValueError("checkpoint is incompatible with this range policy schema or target")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if job["mode"] == "train" and payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
    primary = PPOCatalogPolicy(role, model, stochastic=job["mode"] == "train", seed=seed)
    opponent_role = "blue" if role == "red" else "red"
    opponent = _baseline(opponent_role, "scripted", seed + 1)
    episodes: list[dict[str, Any]] = []
    transitions: list[Transition] = []
    candidate_scores: list[float] = []
    baseline_scores: list[float] = []
    for episode_index in range(int(config["episodes"])):
        if await _stop_requested(client, training_id):
            break
        episode_id = uuid4()
        if role == "red":
            red_policy: Callable[[RangeObservation], dict[str, Any]] = primary
            blue_policy: Callable[[RangeObservation], dict[str, Any]] = opponent
        else:
            red_policy = opponent
            blue_policy = primary
        runner = RangeEpisodeRunner(
            API_URL,
            UUID(training_id),
            episode_id,
            red_policy=red_policy,
            blue_policy=blue_policy,
            red_policy_id=f"{role}-ppo-{training_id}" if role == "red" else "red-scripted-v1",
            blue_policy_id=f"{role}-ppo-{training_id}" if role == "blue" else "blue-scripted-v1",
            request_timeout=5.0,
            stop_check=lambda: _stop_requested(client, training_id),
        )
        result = await runner.run(seed=seed + episode_index, max_steps=int(config["max_steps"]))
        trace = result.get("trace", [])
        score = _episode_reward(trace, role)
        if job["mode"] == "train":
            for step in trace:
                sample = step.get(f"{role}_sample")
                transition_record = step.get(f"{role}_transition")
                if (
                    sample
                    and isinstance(transition_record, dict)
                    and transition_record.get("accepted")
                ):
                    reward_data = transition_record["reward"]
                    transitions.append(
                        Transition(
                            obs=np.asarray(sample["observation"], dtype=np.float32),
                            mask=np.asarray(sample["mask"], dtype=bool),
                            action=int(sample["action"]),
                            log_prob=float(sample["log_prob"]),
                            value=float(sample["value"]),
                            reward=float(reward_data["total"]),
                            terminated=bool(transition_record["terminated"]),
                            truncated=bool(transition_record["truncated"]),
                            next_value=float(transition_record["next_value"]),
                        )
                    )
        episodes.append(
            {
                "episode_id": str(episode_id),
                "status": result.get("status"),
                "steps": len(trace),
                "total_reward": score,
            }
        )
        if job["mode"] == "evaluate" and result.get("status") == "completed":
            baseline_episode_id = uuid4()
            policy_baseline = _baseline(role, str(config["opponent"]), seed + episode_index)
            if role == "red":
                baseline_red, baseline_blue = policy_baseline, opponent
            else:
                baseline_red, baseline_blue = opponent, policy_baseline
            baseline_runner = RangeEpisodeRunner(
                API_URL,
                UUID(training_id),
                baseline_episode_id,
                red_policy=baseline_red,
                blue_policy=baseline_blue,
                red_policy_id=f"{config['opponent']}-{role}-baseline",
                blue_policy_id=f"{config['opponent']}-{role}-baseline",
                stop_check=lambda: _stop_requested(client, training_id),
            )
            baseline_result = await baseline_runner.run(
                seed=seed + episode_index, max_steps=int(config["max_steps"])
            )
            baseline_trace = baseline_result.get("trace", [])
            baseline_score = _episode_reward(baseline_trace, role)
            episodes[-1]["baseline_episode_id"] = str(baseline_episode_id)
            episodes[-1]["baseline_status"] = baseline_result.get("status")
            episodes[-1]["baseline_reward"] = baseline_score
            if baseline_result.get("status") == "completed":
                candidate_scores.append(score)
                baseline_scores.append(baseline_score)
    metrics = (
        _ppo_update(model, optimizer, transitions, float(config["learning_rate"]))
        if job["mode"] == "train"
        else {}
    )
    if job["mode"] == "train" and not transitions:
        raise RuntimeError(
            "no range transitions completed; refusing to create an untrained checkpoint"
        )
    run_dir = ARTIFACT_ROOT / training_id
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "config" / "run.json").write_text(
        json.dumps(
            {
                "training_run_id": training_id,
                "role": role,
                "mode": job["mode"],
                "seed": seed,
                "target_version": "violet_ledger_v1",
                "git_commit": os.getenv("VIOLET_GIT_COMMIT", "unknown"),
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "torch_version": str(torch.__version__),
                "feature_encoding_schema_id": FEATURE_ENCODING_SCHEMA_ID,
                "action_catalog_schema_id": f"{role}.action_catalog.v2",
                "observation_schema_id": f"{role}.range_observation.v2",
                "config": config,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    role_dir = run_dir / "checkpoints" / role
    role_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path: Path | None = role_dir / "final.pt"
    digest: str | None = None
    checkpoint_payload = {
        "format": "violet.range.ppo.v2",
        "role": role,
        "feature_encoding_schema_id": FEATURE_ENCODING_SCHEMA_ID,
        "observation_schema_id": f"{role}.range_observation.v2",
        "action_catalog_schema_id": f"{role}.action_catalog.v2",
        "action_catalog": list(RED_ACTION_CATALOG_V2 if role == "red" else BLUE_ACTION_CATALOG_V2),
        "target_version": "violet_ledger_v1",
        "git_commit": os.getenv("VIOLET_GIT_COMMIT", "unknown"),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episodes": len(episodes),
        "global_step": len(transitions),
    }
    if job["mode"] == "train":
        assert checkpoint_path is not None
        torch.save(checkpoint_payload, checkpoint_path)
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        update = {
            "worker_id": WORKER_ID,
            "role": role,
            "status": "running",
            "result": {
                "update_index": 1,
                "global_step": len(transitions),
                "metrics": metrics,
                "checkpoint_sha256": digest,
            },
        }
        update_response = await client.post(
            f"/range/training-runs/{training_id}/updates", json=update
        )
        update_response.raise_for_status()
        metrics_path = run_dir / "metrics" / "training.jsonl"
        metrics_path.write_text(
            json.dumps({"update": 1, "global_step": len(transitions), **metrics}) + "\n"
        )
    else:
        checkpoint_path = None
    evaluation_summary = None
    if job["mode"] == "evaluate":
        if candidate_scores:
            difference, ci_low, ci_high = _paired_bootstrap(candidate_scores, baseline_scores, seed)
            interval: list[float] | None = [ci_low, ci_high]
        else:
            difference, ci_low, ci_high = 0.0, 0.0, 0.0
            interval = None
        training_seed_count = 1
        supported = len(candidate_scores) >= 100 and ci_low > 0 and training_seed_count >= 5
        evaluation_summary = {
            "schema_id": "violet.range.evaluation.v1",
            "target_version": "violet_ledger_v1",
            "role": role,
            "primary_metric": "mean_role_reward",
            "baseline": str(config["opponent"]),
            "checkpoint_run_id": checkpoint_run_id,
            "paired_episodes": len(candidate_scores),
            "requested_pairs": int(config["episodes"]),
            "candidate_mean": float(np.mean(candidate_scores)) if candidate_scores else None,
            "baseline_mean": float(np.mean(baseline_scores)) if baseline_scores else None,
            "mean_paired_difference": difference,
            "paired_bootstrap_ci_95": interval,
            "training_seed_count": training_seed_count,
            "required_training_seeds": 5,
            "required_held_out_pairs": 100,
            "supported_improvement": supported,
            "claim": "supported improvement" if supported else "no supported improvement",
            "evaluation_seed_start": seed,
            "feature_encoding_schema_id": FEATURE_ENCODING_SCHEMA_ID,
            "action_catalog_schema_id": f"{role}.action_catalog.v2",
            "observation_schema_id": f"{role}.range_observation.v2",
        }
        eval_dir = ARTIFACT_ROOT / training_id / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        episodes_path = eval_dir / "episodes.jsonl"
        episodes_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in episodes)
        )
        summary_path = eval_dir / "summary.json"
        summary_path.write_text(json.dumps(evaluation_summary, sort_keys=True, indent=2) + "\n")
        evaluation_summary["episodes_sha256"] = hashlib.sha256(
            episodes_path.read_bytes()
        ).hexdigest()
        evaluation_summary["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    return {
        "episodes": episodes,
        "optimizer_metrics": metrics,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": digest,
        "global_step": len(transitions),
        "evidence_level": "real_target_episode" if episodes else "no_episode_completed",
        "evaluation": evaluation_summary,
    }


async def _stop_requested(client: httpx.AsyncClient, training_id: str) -> bool:
    heartbeat = await client.post(f"/range/training-runs/{training_id}/heartbeat")
    heartbeat.raise_for_status()
    response = await client.get(f"/range/training-runs/{training_id}")
    response.raise_for_status()
    return bool(response.json().get("stop_requested"))


async def worker() -> None:
    if not TOKEN:
        raise RuntimeError("VIOLET_TRAINER_TOKEN is required for the range trainer")
    headers = {"x-violet-trainer-token": TOKEN, "x-violet-worker-id": WORKER_ID}
    async with httpx.AsyncClient(base_url=API_URL, headers=headers, timeout=10.0) as client:
        while True:
            try:
                response = await client.post("/range/training-runs/claim")
                response.raise_for_status()
                job = response.json().get("job")
                if job is None:
                    challenge_response = await client.post("/range/sqli/challenges/claim")
                    challenge_response.raise_for_status()
                    challenge_job = challenge_response.json().get("job")
                    if challenge_job is None:
                        await asyncio.sleep(2.0)
                        continue
                    challenge_id = str(challenge_job["challenge_id"])
                    try:
                        from violet.sqli_lab.training import run_sqli_challenge

                        result = await run_sqli_challenge(client, challenge_job)
                        status = str(result.get("status", "incomplete"))
                        outcome = result.get("summary") or {}
                    except Exception as exc:
                        status = "failed"
                        outcome = {
                            "schema_id": "sqli.episode_outcome.v2",
                            "phase": "challenge",
                            "status": "failed",
                            "reason": type(exc).__name__,
                            "winner": None,
                            "checkpoint_run_id": challenge_job.get("training_run_id"),
                        }
                    finish_response = await client.post(
                        f"/range/sqli/challenges/{challenge_id}/finish",
                        json={"status": status, "outcome": outcome},
                    )
                    finish_response.raise_for_status()
                    continue
                training_id = str(job["training_run_id"])
                try:
                    result = await _run_job(client, job)
                    status = (
                        "stopped"
                        if await _stop_requested(client, training_id)
                        else "completed"
                        if result.get("research_criteria_met", True)
                        else "failed"
                    )
                except Exception as exc:
                    result = {"error": type(exc).__name__, "message": str(exc)[:300]}
                    status = "failed"
                update_response = await client.post(
                    f"/range/training-runs/{training_id}/worker-update",
                    json={"worker_id": WORKER_ID, "status": status, "result": result},
                )
                update_response.raise_for_status()
            except (httpx.HTTPError, OSError):
                await asyncio.sleep(3.0)


def main() -> None:
    asyncio.run(worker())


if __name__ == "__main__":
    main()
