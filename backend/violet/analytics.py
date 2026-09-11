# violet/analytics.py — behavior discovery from evaluation trajectories

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from violet.types import ATTACKER_AGENT, EpisodeResult


@dataclass
class BehaviorPattern:
    """A discovered behavior backed by measured trajectories."""

    id: str
    agent: str  # "red" or "blue"
    condition: str  # the observed precondition
    observed_behavior: str  # what the policy tended to do
    frequency: int  # how many episodes exhibited it
    total_evaluated: int  # denominator
    baseline_count: int  # same condition under baseline policy
    baseline_total: int
    improvement_description: str
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    side_effects: List[str] = field(default_factory=list)

    @property
    def frequency_pct(self) -> float:
        return 100.0 * self.frequency / max(1, self.total_evaluated)

    @property
    def baseline_pct(self) -> float:
        return 100.0 * self.baseline_count / max(1, self.baseline_total)


@dataclass
class ActionDistribution:
    """Action selection frequency for an agent, optionally conditioned on state."""

    agent: str
    action_counts: Dict[str, int]
    total: int
    condition: str = ""


def analyze_red_behaviors(
    episode_results: List[EpisodeResult],
    baseline_results: List[EpisodeResult],
) -> List[BehaviorPattern]:
    """Discover red behaviors from evaluation trajectories.

    Example: "under condition: initial foothold on WEB, observed tendency: lateral
    movement toward APP in 71/100 episodes, baseline 43/100."
    """
    behaviors: List[BehaviorPattern] = []

    # Behavior 01: lateral movement toward APP after WEB compromise
    episodes_with_web_compromised = 0
    episodes_lateral_to_app = 0
    for ep in episode_results:
        events = ep.events
        web_compromised = any(
            e.get("agent") == "red"
            and e.get("action") == "EXPLOIT"
            and e.get("target") == "web"
            and e.get("result") == "exploit_success"
            for e in events
        )
        if web_compromised:
            episodes_with_web_compromised += 1
            lateral_to_app = any(
                e.get("agent") == "red"
                and e.get("action") == "LATERAL_MOVE"
                and e.get("target") == "app"
                for e in events
            )
            if lateral_to_app:
                episodes_lateral_to_app += 1

    baseline_web = sum(
        1
        for ep in baseline_results
        if any(
            e.get("agent") == "red"
            and e.get("action") == "EXPLOIT"
            and e.get("target") == "web"
            and e.get("result") == "exploit_success"
            for e in ep.events
        )
    )
    baseline_lateral = sum(
        1
        for ep in baseline_results
        if any(
            e.get("agent") == "red"
            and e.get("action") == "LATERAL_MOVE"
            and e.get("target") == "app"
            for e in ep.events
        )
    )

    behaviors.append(BehaviorPattern(
        id="DISCOVERED-01",
        agent="red",
        condition="Initial foothold established on WEB (exploited successfully)",
        observed_behavior="Lateral movement toward APP",
        frequency=episodes_lateral_to_app,
        total_evaluated=episodes_with_web_compromised,
        baseline_count=baseline_lateral,
        baseline_total=max(1, baseline_web),
        improvement_description=f"{episodes_lateral_to_app}/{max(1, episodes_with_web_compromised)} evaluated vs {baseline_lateral}/{max(1, baseline_web)} baseline",
        confidence="HIGH" if episodes_with_web_compromised >= 50 else "MEDIUM",
    ))

    return behaviors


def analyze_blue_behaviors(
    episode_results: List[EpisodeResult],
    baseline_results: List[EpisodeResult],
) -> List[BehaviorPattern]:
    """Discover blue behaviors from evaluation trajectories.

    Example: "under condition: suspicious lateral movement detected, blue policy
    selects ISOLATE APP in 82% of episodes, baseline 31%."
    """
    behaviors: List[BehaviorPattern] = []

    # Condition: lateral movement event occurred
    episodes_with_lateral = 0
    episodes_isolated_app = 0
    for ep in episode_results:
        events = ep.events
        lateral_event = any(
            e.get("agent") == "red"
            and e.get("action") == "LATERAL_MOVE"
            for e in events
        )
        if lateral_event:
            episodes_with_lateral += 1
            isolated_app = any(
                e.get("agent") == "blue"
                and e.get("action") == "ISOLATE"
                and e.get("target") == "app"
                for e in events
            )
            if isolated_app:
                episodes_isolated_app += 1

    baseline_lateral_count = sum(
        1
        for ep in baseline_results
        if any(
            e.get("agent") == "red"
            and e.get("action") == "LATERAL_MOVE"
            for e in ep.events
        )
    )
    baseline_isolated_app = sum(
        1
        for ep in baseline_results
        if any(
            e.get("agent") == "blue"
            and e.get("action") == "ISOLATE"
            and e.get("target") == "app"
            for e in ep.events
        )
    )

    behaviors.append(BehaviorPattern(
        id="DISCOVERED-07",
        agent="blue",
        condition="Suspicious lateral movement detected (RED LATERAL_MOVE event)",
        observed_behavior="ISOLATE APP",
        frequency=episodes_isolated_app,
        total_evaluated=episodes_with_lateral,
        baseline_count=baseline_isolated_app,
        baseline_total=max(1, baseline_lateral_count),
        improvement_description=f"{episodes_isolated_app}/{max(1, episodes_with_lateral)} evaluated vs {baseline_isolated_app}/{max(1, baseline_lateral_count)} baseline",
        confidence="HIGH" if episodes_with_lateral >= 50 else "MEDIUM",
        side_effects=["Service availability impact measured separately"],
    ))

    return behaviors


def action_distribution(
    episode_results: List[EpisodeResult],
    agent: str,
    condition_fn: Optional[Callable[[EpisodeResult], bool]] = None,
) -> ActionDistribution:
    """Compute action selection frequency for *agent* across trajectories.

    If *condition_fn* is provided, only episodes matching the condition are counted
    (state-conditioned distribution).
    """
    counts: Dict[str, int] = defaultdict(int)
    total = 0

    for ep in episode_results:
        if condition_fn is not None and not condition_fn(ep):
            continue
        for ev in ep.events:
            if ev.get("agent") == agent:
                action = ev.get("action", "UNKNOWN")
                counts[action] += 1
                total += 1

    return ActionDistribution(
        agent=agent,
        action_counts=dict(counts),
        total=total,
        condition=condition_fn.__name__ if condition_fn else "",
    )


def heatmap_data(
    episode_results: List[EpisodeResult],
    agent: str,
) -> Dict[str, Dict[str, int]]:
    """Return a state x action count table for conditional policy analysis.

    Rows = observed state (derived from alert/compromise flags in events).
    Columns = action selected.
    """
    data: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ep in episode_results:
        for ev in ep.events:
            if ev.get("agent") == agent:
                # Derive a coarse state label from context (to be refined in V2)
                state = "UNKNOWN"
                for e in ep.events:
                    if e.get("step", 0) <= ev.get("step", 0):
                        if e.get("agent") == "red" and e.get("result") == "exploit_success":
                            state = f"{e.get('target','?')} compromised"
                        elif e.get("agent") == "blue" and e.get("result") == "monitor_detected":
                            state = f"{e.get('target','?')} alert"
                action = ev.get("action", "UNKNOWN")
                data[state][action] += 1
    return dict(data)


def behavior_discovery_report(
    run_id: str,
    red_results: List[EpisodeResult],
    blue_results: List[EpisodeResult],
    red_baseline: List[EpisodeResult],
    blue_baseline: List[EpisodeResult],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Produce the full behavior discovery report (for the dashboard + docs)."""
    red_behaviors = analyze_red_behaviors(red_results, red_baseline)
    blue_behaviors = analyze_blue_behaviors(blue_results, blue_baseline)

    red_dist = action_distribution(red_results, ATTACKER_AGENT)
    blue_dist = action_distribution(blue_results, DEFENDER_AGENT)

    red_dist_cond = action_distribution(
        red_results,
        ATTACKER_AGENT,
        condition_fn=lambda ep: any(
            e.get("agent") == "red"
            and e.get("action") == "EXPLOIT"
            and e.get("target") == "web"
            and e.get("result") == "exploit_success"
            for e in ep.events
        ),
    )
    blue_dist_cond = action_distribution(
        blue_results,
        DEFENDER_AGENT,
        condition_fn=lambda ep: any(
            e.get("agent") == "red"
            and e.get("action") == "LATERAL_MOVE"
            for e in ep.events
        ),
    )

    report = {
        "run_id": run_id,
        "red_behaviors": [
            {
                "id": b.id,
                "condition": b.condition,
                "observed_behavior": b.observed_behavior,
                "frequency": f"{b.frequency} / {b.total_evaluated}",
                "baseline": f"{b.baseline_count} / {b.baseline_total}",
                "improvement": b.improvement_description,
                "confidence": b.confidence,
            }
            for b in red_behaviors
        ],
        "blue_behaviors": [
            {
                "id": b.id,
                "condition": b.condition,
                "observed_behavior": b.observed_behavior,
                "frequency": f"{b.frequency} / {b.total_evaluated}",
                "baseline": f"{b.baseline_count} / {b.baseline_total}",
                "improvement": b.improvement_description,
                "confidence": b.confidence,
                "side_effects": b.side_effects,
            }
            for b in blue_behaviors
        ],
        "red_action_distribution": {
            "condition": red_dist.condition,
            "total": red_dist.total,
            "actions": dict(red_dist.action_counts),
        },
        "red_action_distribution_conditioned": {
            "condition": red_dist_cond.condition,
            "total": red_dist_cond.total,
            "actions": dict(red_dist_cond.action_counts),
        },
        "blue_action_distribution": {
            "condition": blue_dist.condition,
            "total": blue_dist.total,
            "actions": dict(blue_dist.action_counts),
        },
        "blue_action_distribution_conditioned": {
            "condition": blue_dist_cond.condition,
            "total": blue_dist_cond.total,
            "actions": dict(blue_dist_cond.action_counts),
        },
        "red_heatmap": heatmap_data(red_results, ATTACKER_AGENT),
        "blue_heatmap": heatmap_data(blue_results, DEFENDER_AGENT),
    }

    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "behavior_report.json", "w") as f:
            json.dump(report, f, indent=2)

    return report


# Backward compat for callables used above

