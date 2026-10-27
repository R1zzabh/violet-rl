"use client";

import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Pause,
  Play,
  RefreshCw,
  SkipBack,
  SkipForward,
} from "lucide-react";
import {
  rangeApi,
  type RangeEvent,
  type SqliEpisodeIndexItem,
  type SqliEpisodeOutcomeKey,
  type SqliEpisodePhase,
} from "../../lib/range-api";

type Filters = {
  runId: string;
  phase: SqliEpisodePhase | "all";
  status: "running" | "completed" | "incomplete" | "failed" | "all";
  outcome: SqliEpisodeOutcomeKey | "all";
};
type IndexPage = {
  items: SqliEpisodeIndexItem[];
  nextCursor: string | null;
  hasMore: boolean;
};
type TimelineSnapshot = { runId: string; episodeId: string; items: RangeEvent[] };
type RunSnapshot = { runId: string; record: Record<string, unknown> };
type PlaybackState = { episodeId: string; index: number; playing: boolean };
type PpoSettings = {
  algorithm: string;
  actor_critic: string;
  gamma: number;
  gae_lambda: number;
  clip_range: number;
  policy_loss_coefficient: number;
  value_loss_coefficient: number;
  entropy_coefficient: number;
  learning_rate: number | null;
  optimizer: string;
  gradient_clip_max_norm: number;
  optimization_passes: number;
  advantage_normalization: string;
  provenance: string;
};
type DerivedValue = {
  advantage: number;
  normalizedAdvantage: number;
  returnTarget: number;
  recorded?: boolean;
  oldJointProbability?: number;
  newJointProbability?: number;
  oldJointLogProbability?: number;
  newJointLogProbability?: number;
};
type NextBlueUse = { episodeIndex: number; policyVersion: number | null } | null;

const EMPTY_FILTERS: Filters = { runId: "", phase: "all", status: "all", outcome: "all" };
const EMPTY_EVENTS: RangeEvent[] = [];
const object = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const str = (value: unknown, fallback = "—"): string =>
  typeof value === "string" && value.length ? value : fallback;
const num = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
const formatTimestamp = (value: string, includeSeconds = true): string => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "timestamp unavailable";
  return `${date.toISOString().slice(0, includeSeconds ? 19 : 16).replace("T", " ")} UTC`;
};
const fixed = (value: unknown, digits = 4): string => {
  const parsed = num(value);
  return parsed === null ? "unavailable" : parsed.toFixed(digits);
};

function redact(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redact);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([key, item]) => [
      key,
      /phrase/i.test(key) ? "[redacted]" : redact(item),
    ]));
  }
  return value;
}

const pretty = (value: unknown) => JSON.stringify(redact(value ?? {}), null, 2);

function parsePpoSettings(value: unknown, provenance: string): PpoSettings | null {
  const raw = object(value);
  const numericFields = [
    "gamma",
    "gae_lambda",
    "clip_range",
    "policy_loss_coefficient",
    "value_loss_coefficient",
    "entropy_coefficient",
    "learning_rate",
    "gradient_clip_max_norm",
    "optimization_passes",
  ] as const;
  if (
    numericFields.some((field) => num(raw[field]) === null)
    || typeof raw.algorithm !== "string"
    || typeof raw.actor_critic !== "string"
    || typeof raw.optimizer !== "string"
    || typeof raw.advantage_normalization !== "string"
  ) return null;
  return {
    algorithm: raw.algorithm,
    actor_critic: raw.actor_critic,
    gamma: num(raw.gamma)!,
    gae_lambda: num(raw.gae_lambda)!,
    clip_range: num(raw.clip_range)!,
    policy_loss_coefficient: num(raw.policy_loss_coefficient)!,
    value_loss_coefficient: num(raw.value_loss_coefficient)!,
    entropy_coefficient: num(raw.entropy_coefficient)!,
    learning_rate: num(raw.learning_rate),
    optimizer: raw.optimizer,
    gradient_clip_max_norm: num(raw.gradient_clip_max_norm)!,
    optimization_passes: num(raw.optimization_passes)!,
    advantage_normalization: raw.advantage_normalization,
    provenance,
  };
}

function knownSettingsForRevision(revision: string | null, learningRate: number | null): PpoSettings | null {
  if (revision !== "red_syntax_progress_ppo4_v2" && revision !== "retained_baseline_reward_v1_ppo4"
    && revision !== "balanced_adversarial_gru_ppo4_v4") {
    return null;
  }
  return {
    algorithm: "PPO_clipped_surrogate_GAE",
    actor_critic: revision === "balanced_adversarial_gru_ppo4_v4"
      ? "factorized_gru_actor_critic" : "factorized_actor_critic",
    gamma: 0.99,
    gae_lambda: 0.95,
    clip_range: 0.2,
    policy_loss_coefficient: 1,
    value_loss_coefficient: 0.5,
    entropy_coefficient: 0.01,
    learning_rate: learningRate,
    optimizer: "Adam",
    gradient_clip_max_norm: 0.5,
    optimization_passes: 4,
    advantage_normalization: "mean_std_when_batch_has_multiple_transitions",
    provenance: `known settings for recorded campaign revision ${revision}`,
  };
}

function settingsFromEvents(events: RangeEvent[], runConfig: Record<string, unknown> | null): PpoSettings | null {
  for (const event of events) {
    if (event.event_type !== "sqli_ppo_update") continue;
    const metrics = object(event.payload.metrics);
    const recorded = object(event.payload.ppo_settings ?? metrics.ppo_settings);
    const settings = parsePpoSettings({
      ...recorded,
      optimization_passes: recorded.optimization_passes
        ?? event.payload.optimization_passes
        ?? metrics.optimizer_steps,
    }, "recorded on this PPO update");
    if (settings) return settings;
  }

  const result = object(runConfig?.result);
  const metadata = object(result.checkpoint_metadata);
  const sourceRun = object(runConfig?.checkpoint_source);
  const sourceResult = object(sourceRun.result);
  const sourceMetadata = object(sourceResult.checkpoint_metadata);
  const recordedSettings = [
    runConfig?.ppo_settings,
    result.ppo_settings,
    metadata.ppo_settings,
    sourceRun.ppo_settings,
    sourceResult.ppo_settings,
    sourceMetadata.ppo_settings,
  ];
  for (const candidate of recordedSettings) {
    const settings = parsePpoSettings(candidate, "recorded in the run or checkpoint metadata");
    if (settings) return settings;
  }

  const config = object(runConfig?.config);
  const sourceConfig = object(sourceRun.config);
  const revision = typeof config.campaign_revision === "string" ? config.campaign_revision : null;
  const sourceRevision = typeof sourceConfig.campaign_revision === "string"
    ? sourceConfig.campaign_revision : null;
  return knownSettingsForRevision(
    revision ?? sourceRevision,
    num(config.learning_rate) ?? num(sourceConfig.learning_rate),
  );
}

function deriveGae(events: RangeEvent[], settings: PpoSettings | null): Map<string, DerivedValue> {
  const result = new Map<string, DerivedValue>();
  if (!settings) return result;
  const updatedRoles = new Set(events
    .filter((event) => event.event_type === "sqli_ppo_update"
      && (event.payload.updated === true || event.payload.update_status === "applied"))
    .map((event) => str(event.payload.role, "")));

  for (const role of updatedRoles) {
    const transitions = events
      .filter((event) => event.event_type === "sqli_transition"
        && event.payload.role === role
        && event.payload.accepted === true
        && event.payload.learning_eligible === true)
      .sort((left, right) => (num(left.payload.step_id) ?? 0) - (num(right.payload.step_id) ?? 0));
    const advantages = new Array<number>(transitions.length).fill(0);
    let nextAdvantage = 0;
    for (let index = transitions.length - 1; index >= 0; index -= 1) {
      const payload = transitions[index].payload;
      const terminal = payload.terminated === true;
      const boundary = payload.truncated === true;
      const value = num(payload.value_estimate) ?? 0;
      const nextValue = num(payload.next_value) ?? 0;
      const reward = num(payload.reward) ?? 0;
      const continuation = terminal || boundary ? 0 : 1;
      const delta = reward + settings.gamma * nextValue * (terminal ? 0 : 1) - value;
      nextAdvantage = delta + settings.gamma * settings.gae_lambda * continuation * nextAdvantage;
      advantages[index] = nextAdvantage;
    }
    const mean = advantages.length ? advantages.reduce((sum, value) => sum + value, 0) / advantages.length : 0;
    const variance = advantages.length
      ? advantages.reduce((sum, value) => sum + (value - mean) ** 2, 0) / advantages.length
      : 0;
    const standardDeviation = Math.sqrt(variance);
    transitions.forEach((event, index) => {
      const value = num(event.payload.value_estimate) ?? 0;
      result.set(event.event_id, {
        advantage: advantages[index],
        normalizedAdvantage: advantages.length > 1
          ? (advantages[index] - mean) / (standardDeviation + 1e-8)
          : advantages[index],
        returnTarget: advantages[index] + value,
      });
    });
  }
  return result;
}

function recordedUpdateValues(events: RangeEvent[]): Map<string, DerivedValue> {
  const byDecision = new Map<string, DerivedValue>();
  for (const event of events) {
    if (event.event_type !== "sqli_ppo_update" || event.payload.updated !== true) continue;
    const diagnostics = Array.isArray(event.payload.transition_diagnostics)
      ? event.payload.transition_diagnostics : [];
    for (const raw of diagnostics) {
      const item = object(raw);
      if (typeof item.decision_id !== "string") continue;
      const advantage = num(item.advantage);
      const normalizedAdvantage = num(item.normalized_advantage);
      const returnTarget = num(item.return_target);
      if (advantage === null || normalizedAdvantage === null || returnTarget === null) continue;
      byDecision.set(item.decision_id, {
        advantage,
        normalizedAdvantage,
        returnTarget,
        recorded: true,
        oldJointProbability: num(item.old_joint_probability) ?? undefined,
        newJointProbability: num(item.new_joint_probability) ?? undefined,
        oldJointLogProbability: num(item.old_joint_log_probability) ?? undefined,
        newJointLogProbability: num(item.new_joint_log_probability) ?? undefined,
      });
    }
  }
  const byTransition = new Map<string, DerivedValue>();
  for (const event of events) {
    if (event.event_type !== "sqli_transition") continue;
    const decisionId = str(event.payload.decision_id, "");
    const value = byDecision.get(decisionId);
    if (value) byTransition.set(event.event_id, value);
  }
  return byTransition;
}

function eventLabel(event: RangeEvent): string {
  const labels: Record<string, string> = {
    sqli_episode_started: "Episode opened",
    sqli_red_decision: "Red chose a request",
    sqli_target_response: event.payload.phase === "request_submitted"
      ? "Target request submitted" : "Target response recorded",
    sqli_blue_decision: "Blue chose a control",
    sqli_control_acknowledged: "Control acknowledged",
    sqli_verification: "Independent verification",
    sqli_reward: "Round reward",
    sqli_transition: `${str(event.payload.role, "Policy")} learning transition`,
    sqli_policy_probe: `${str(event.payload.role, "Policy")} same-observation probe`,
    sqli_ppo_update: `${str(event.payload.role, "Policy")} PPO update`,
    sqli_episode_summary: "Episode summary",
    sqli_episode_outcome: "Episode outcome",
  };
  return labels[event.event_type] ?? event.event_type.replaceAll("_", " ");
}

function correlationIds(event: RangeEvent): [string, string][] {
  return ["decision_id", "request_id", "target_event_id", "job_id", "command_id", "challenge_id"]
    .flatMap((key) => typeof event.payload[key] === "string" && event.payload[key]
      ? [[key, String(event.payload[key]) as string]] : []);
}

function DataValue({ label, value }: { label: string; value: unknown }) {
  return <div className="episode-data-value"><span>{label}</span><strong>{value === null || value === undefined || value === "" ? "unavailable" : String(value)}</strong></div>;
}

function JsonDetails({ label, value }: { label: string; value: unknown }) {
  return <details className="episode-json-details">
    <summary>{label}</summary>
    <pre>{pretty(value)}</pre>
  </details>;
}

function EventDetails({ event, derived, settings }: {
  event: RangeEvent;
  derived: DerivedValue | undefined;
  settings: PpoSettings | null;
}) {
  const payload = event.payload;
  const metrics = object(payload.metrics);
  const action = object(payload.action ?? payload.selected_action);
  const verification = object(payload.exposure_verification ?? payload.verification
    ?? (event.event_type === "sqli_verification" ? payload : undefined));
  const transitionReward = object(payload.reward);
  const redReward = object(object(payload.red).total !== undefined ? payload.red : transitionReward);
  const blueReward = object(object(payload.blue).total !== undefined ? payload.blue : transitionReward);

  switch (event.event_type) {
    case "sqli_episode_started":
      return <>
        <p className="episode-explainer">The runner recorded its seed, phase, policy versions, and schema before interacting with the target.</p>
        <div className="episode-data-grid">
          <DataValue label="Episode kind" value={payload.episode_kind} />
          <DataValue label="Seed" value={payload.seed} />
          <DataValue label="Maximum rounds" value={payload.max_steps} />
          <DataValue label="Starting Red / Blue policy" value={pretty(payload.policy_versions)} />
        </div>
      </>;
    case "sqli_red_decision":
      return <>
        <p className="episode-explainer">Red samples one bounded request action from its current observation. The target response is shown in a separate recorded event.</p>
        <div className="episode-data-grid">
          {Object.entries(action).map(([key, value]) => <DataValue key={key} label={key.replaceAll("_", " ")} value={value} />)}
          <DataValue label="Policy version" value={payload.policy_version == null ? null : `v${payload.policy_version}`} />
          <DataValue label="Available action mask" value={pretty(payload.available_action_mask ?? object(payload.observation).legal_action_mask)} />
          <DataValue label="Selected component probabilities" value={pretty(payload.selected_component_probabilities)} />
          <DataValue label="Sampled joint action probability" value={fixed(payload.sampled_action_probability ?? Math.exp(num(payload.joint_log_probability ?? payload.old_log_probability) ?? Number.NaN))} />
          <DataValue label="Joint log probability" value={fixed(payload.joint_log_probability ?? payload.old_log_probability)} />
          <DataValue label="Critic value" value={fixed(payload.critic_value ?? payload.value_estimate)} />
        </div>
        <JsonDetails label="Red observation" value={payload.observation} />
        <JsonDetails label="Red recurrent state before and after decision" value={{ before: payload.recurrent_state_before, after: payload.recurrent_state_after }} />
        <JsonDetails label="Full action probability distributions" value={payload.component_distributions} />
      </>;
    case "sqli_target_response": {
      const response = object(payload.response);
      const request = object(payload.request);
      const rows = Array.isArray(response.rows) ? response.rows : [];
      const submitted = payload.phase === "request_submitted";
      const shownRequest = submitted ? {
        actor_ref: payload.actor_ref,
        components: payload.components,
        query_value: payload.composed_query_value,
      } : request;
      return <>
        <p className="episode-explainer">This frame reads the saved executor and target record. Replay never sends the HTTP request again.</p>
        <div className="episode-data-grid">
          <DataValue label="HTTP request" value={`${str(payload.method, "POST")} ${str(payload.route, "/api/lookup")}`} />
          <DataValue label="Request body" value={pretty(shownRequest)} />
          <DataValue label="HTTP status" value={payload.http_status == null ? null : `HTTP ${payload.http_status}`} />
          <DataValue label="Evidence status" value={payload.evidence_status ?? object(payload.telemetry).evidence_status} />
          <DataValue label="Returned rows" value={submitted ? "response not recorded" : rows.length} />
          <DataValue label="Unauthorized rows" value={object(payload.telemetry).unauthorized_rows} />
        </div>
        {submitted
          ? <div className="episode-missing">The request submission was saved, but its target response is missing. No response is inferred.</div>
          : <JsonDetails label="Actual returned rows and response body" value={response} />}
      </>;
    }
    case "sqli_blue_decision":
      return <>
        <p className="episode-explainer">Blue sees the round-start telemetry and samples a typed target control before Red&apos;s request is executed.</p>
        <div className="episode-data-grid">
          <DataValue label="Control action" value={action.operation} />
          <DataValue label="Policy version" value={payload.policy_version == null ? null : `v${payload.policy_version}`} />
          <DataValue label="Available action mask" value={pretty(payload.available_action_mask ?? object(payload.observation).legal_action_mask)} />
          <DataValue label="Selected component probability" value={pretty(payload.selected_component_probabilities)} />
          <DataValue label="Sampled joint action probability" value={fixed(payload.sampled_action_probability ?? Math.exp(num(payload.joint_log_probability ?? payload.old_log_probability) ?? Number.NaN))} />
          <DataValue label="Joint log probability" value={fixed(payload.joint_log_probability ?? payload.old_log_probability)} />
          <DataValue label="Critic value" value={fixed(payload.critic_value ?? payload.value_estimate)} />
        </div>
        <JsonDetails label="Blue observation" value={payload.observation} />
        <JsonDetails label="Blue recurrent state before and after decision" value={{ before: payload.recurrent_state_before, after: payload.recurrent_state_after }} />
        <JsonDetails label="Full action probability distributions" value={payload.component_distributions} />
      </>;
    case "sqli_control_acknowledged": {
      const acknowledgement = object(payload.acknowledgement);
      return <>
        <p className="episode-explainer">The acknowledgment records whether the target applied Blue&apos;s command and which settings became active.</p>
        <div className="episode-data-grid">
          <DataValue label="Command" value={payload.action} />
          <DataValue label="Acknowledgment" value={acknowledgement.status ?? payload.status} />
          <DataValue label="Control state" value={pretty(acknowledgement.controls ?? payload.controls)} />
        </div>
      </>;
    }
    case "sqli_verification":
      return <>
        <p className="episode-explainer">Independent probes check both unauthorized access prevention and legitimate lookup availability.</p>
        <div className="episode-data-grid">
          <DataValue label="Evidence join" value={verification.evidence_verified === true ? "verified" : verification.status ?? "partial / unavailable"} />
          <DataValue label="Verified exposure" value={verification.verified_exposure} />
          <DataValue label="Challenge disclosed" value={verification.challenge_disclosed} />
          <DataValue label="Unauthorized rows" value={verification.unauthorized_rows_returned ?? verification.exposure_count} />
          <DataValue label="Attack probe HTTP" value={verification.attack_probe_status} />
          <DataValue label="Attack prevention" value={verification.verified_prevention} />
          <DataValue label="Legitimate lookup HTTP" value={verification.legitimate_query_status} />
          <DataValue label="Legitimate lookup served" value={verification.legitimate_query_served} />
        </div>
      </>;
    case "sqli_reward":
      return <>
        <p className="episode-explainer">Rewards are the recorded per-round learning signal. They are not an outcome score by themselves.</p>
        <div className="episode-data-grid">
          <DataValue label="Red reward" value={num(redReward.total)?.toFixed(4)} />
          <DataValue label="Blue reward" value={num(blueReward.total)?.toFixed(4)} />
          <DataValue label="Red components" value={pretty(redReward.components)} />
          <DataValue label="Blue components" value={pretty(blueReward.components)} />
        </div>
      </>;
    case "sqli_transition":
      return <>
        <p className="episode-explainer">A transition joins the recorded decision to its reward and next observation. PPO inputs below come from the persisted update diagnostics when available; older records are labeled as recomputed.</p>
        <div className="episode-data-grid">
          <DataValue label="Role / accepted" value={`${str(payload.role)} · ${payload.accepted === true ? "accepted" : "not accepted"}`} />
          <DataValue label="Included in PPO batch" value={payload.learning_eligible === true ? "eligible" : "no"} />
          <DataValue label="Reward" value={num(payload.reward)?.toFixed(4)} />
          <DataValue label="Value estimate" value={fixed(payload.value_estimate)} />
          <DataValue label={derived?.recorded ? "Recorded PPO advantage" : "Recomputed GAE advantage"} value={derived ? fixed(derived.advantage) : null} />
          <DataValue label={derived?.recorded ? "Recorded normalized advantage" : "Recomputed normalized advantage"} value={derived ? fixed(derived.normalizedAdvantage) : null} />
          <DataValue label={derived?.recorded ? "Recorded return target" : "Recomputed return target"} value={derived ? fixed(derived.returnTarget) : null} />
          {derived?.recorded && <>
            <DataValue label="Old joint probability" value={fixed(derived.oldJointProbability)} />
            <DataValue label="New joint probability" value={fixed(derived.newJointProbability)} />
            <DataValue label="Old joint log probability" value={fixed(derived.oldJointLogProbability)} />
            <DataValue label="New joint log probability" value={fixed(derived.newJointLogProbability)} />
          </>}
          <DataValue label="Old action distributions" value={pretty(payload.component_distributions)} />
          <DataValue label="Transition boundary" value={payload.truncated === true ? "truncated" : payload.terminated === true ? "terminal" : "continues"} />
        </div>
        {!derived && <div className="episode-missing">Derived GAE values are unavailable because this transition was not part of a recorded applied update or its PPO settings are missing.</div>}
        {derived && settings && <p className="episode-footnote">{derived.recorded ? "Exact values persisted by this optimizer update." : `Recomputed for this historical event using γ = ${settings.gamma}, λ = ${settings.gae_lambda}.`}</p>}
      </>;
    case "sqli_policy_probe":
      return <>
        <p className="episode-explainer">The trainer evaluated one reference observation before and after the episode update. This observed policy shift describes that state and does not measure general performance.</p>
        <div className="episode-data-grid">
          <DataValue label="Policy version" value={`v${str(payload.policy_version_before)} → v${str(payload.policy_version_after)}`} />
          <DataValue label="Same reference state" value={payload.same_reference_observation === true ? "yes" : "not confirmed"} />
          <DataValue label="Before probabilities" value={pretty(payload.before_action_probabilities)} />
          <DataValue label="After probabilities" value={pretty(payload.after_action_probabilities)} />
        </div>
      </>;
    case "sqli_ppo_update": {
      const updateSettings = object(payload.ppo_settings ?? metrics.ppo_settings);
      return <>
        <p className="episode-explainer">PPO compares the saved joint action probabilities with the same observations and recurrent states used for the update. The recorded diagnostics include per-transition distributions, returns, advantages, losses, and clipping statistics.</p>
        <div className="episode-data-grid">
          <DataValue label="Update status" value={payload.update_status} />
          <DataValue label="Policy version" value={`v${str(payload.policy_version_before)} → v${str(payload.policy_version)}`} />
          <DataValue label="Optimizer passes" value={payload.optimization_passes ?? metrics.optimizer_steps} />
          <DataValue label="Total loss" value={fixed(metrics.loss)} />
          <DataValue label="Policy loss" value={fixed(metrics.policy_loss)} />
          <DataValue label="Value loss" value={fixed(metrics.value_loss)} />
          <DataValue label="Entropy" value={fixed(metrics.entropy)} />
          <DataValue label="Gradient norm" value={fixed(metrics.gradient_norm)} />
          <DataValue label="Post-update KL" value={fixed(metrics.post_update_approx_kl ?? metrics.approx_kl)} />
          <DataValue label="Post-update clip fraction" value={fixed(metrics.post_update_clip_fraction ?? metrics.clip_fraction)} />
          <DataValue label="Checkpoint SHA-256" value={payload.checkpoint_sha256} />
        </div>
        <JsonDetails label={updateSettings.gamma === undefined ? "PPO settings · unavailable in this historical event" : "Exact PPO settings"} value={updateSettings.gamma === undefined ? settings : updateSettings} />
        <JsonDetails label="Exact transitions and old / new action probabilities used by PPO" value={payload.transition_diagnostics} />
      </>;
    }
    case "sqli_episode_summary":
    case "sqli_episode_outcome":
      return <>
        <p className="episode-explainer">This is the persisted episode result after evidence checks. The winner is assigned from the recorded verification rules.</p>
        <div className="episode-data-grid">
          <DataValue label="Phase" value={payload.phase ?? payload.kind} />
          <DataValue label="Status" value={payload.status} />
          <DataValue label="Winner" value={payload.winner ?? "no winner recorded"} />
          <DataValue label="Reason" value={payload.reason} />
          <DataValue label="Rounds" value={payload.steps} />
          <DataValue label="Red / Blue return" value={`${str(payload.red_return)} / ${str(payload.blue_return)}`} />
          <DataValue label="Blue policy version after episode" value={object(payload.policy_versions).blue} />
          <DataValue label="Checkpoint" value={payload.checkpoint_sha256} />
        </div>
      </>;
    default:
      return <p className="episode-explainer">This saved event is included in the replay sequence. Its exact, phrase-redacted payload is available below.</p>;
  }
}

function EpisodeRow({ item, selected, onSelect }: {
  item: SqliEpisodeIndexItem;
  selected: boolean;
  onSelect: () => void;
}) {
  const winner = item.outcome?.winner ?? (item.outcome_key === "unknown" ? "outcome pending" : "no winner");
  return <button className={`episode-index-row ${selected ? "is-selected" : ""}`} onClick={onSelect} aria-pressed={selected}>
    <span className="episode-row-top"><span className={`pill ${item.phase === "challenge" ? "pill-amber" : item.phase === "training" ? "pill-blue" : ""}`}>{item.phase}</span><span className="pill">{item.status}</span></span>
    <strong>Episode #{item.episode_index + 1} · {winner}</strong>
    <span className="episode-row-meta">run {item.run_id.slice(0, 8)} · seed {item.seed ?? "—"} · {formatTimestamp(item.timestamp, false)}</span>
    <span className="episode-row-meta">{item.run_mode} / {item.run_type} · {item.outcome?.reason ?? "outcome evidence unavailable"}</span>
  </button>;
}

async function readTimeline(runId: string, episodeId: string): Promise<RangeEvent[]> {
  const all: RangeEvent[] = [];
  let cursor = 0;
  for (;;) {
    const page = await rangeApi.episodeTimeline(runId, episodeId, cursor, 500);
    all.push(...page.items);
    if (!page.has_more || !page.items.length || page.next_after_seq <= cursor) break;
    cursor = page.next_after_seq;
  }
  return all.sort((left, right) => left.seq - right.seq);
}

export default function EpisodesPage() {
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [runIdDraft, setRunIdDraft] = useState("");
  const [cursor, setCursor] = useState<string | null>(null);
  const [cursorHistory, setCursorHistory] = useState<(string | null)[]>([]);
  const [page, setPage] = useState<IndexPage | null>(null);
  const [selectedEpisodeId, setSelectedEpisodeId] = useState("");
  const [timelineSnapshot, setTimelineSnapshot] = useState<TimelineSnapshot | null>(null);
  const [runSnapshot, setRunSnapshot] = useState<RunSnapshot | null>(null);
  const [nextBlueSnapshot, setNextBlueSnapshot] = useState<{ episodeId: string; value: NextBlueUse } | null>(null);
  const [playback, setPlayback] = useState<PlaybackState>({ episodeId: "", index: 0, playing: false });
  const [speed, setSpeed] = useState(1);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [loadingIndex, setLoadingIndex] = useState(true);
  const [error, setError] = useState("");

  const selected = page?.items.find((item) => item.episode_id === selectedEpisodeId) ?? null;
  const selectedRunId = selected?.run_id ?? "";
  const selectedEpisodeIndex = selected?.episode_index ?? -1;
  const selectedPhase = selected?.phase ?? "other";
  const selectedRunMode = selected?.run_mode ?? "";
  const selectedStatus = selected?.status ?? "";
  const events = timelineSnapshot?.runId === selectedRunId && timelineSnapshot.episodeId === selectedEpisodeId
    ? timelineSnapshot.items : EMPTY_EVENTS;
  const runConfig = runSnapshot?.runId === selectedRunId ? runSnapshot.record : null;
  const loadingTimeline = Boolean(selected && (timelineSnapshot?.runId !== selected.run_id
    || timelineSnapshot.episodeId !== selected.episode_id));
  const frames = events;
  const frameIndex = playback.episodeId === selectedEpisodeId ? playback.index : 0;
  const playing = playback.episodeId === selectedEpisodeId && playback.playing;
  const nextBlueUse = nextBlueSnapshot?.episodeId === selectedEpisodeId ? nextBlueSnapshot.value : null;
  const currentFrame = frames[frameIndex] ?? null;
  const settings = settingsFromEvents(events, runConfig);
  const derivedTransitions = deriveGae(events, settings);
  for (const [eventId, value] of recordedUpdateValues(events)) derivedTransitions.set(eventId, value);
  const blueUpdate = [...events].reverse().find((event) =>
    event.event_type === "sqli_ppo_update" && event.payload.role === "blue");
  const targetEvents = events.filter((event) => event.event_type === "sqli_target_response");
  const savedResponses = new Set(targetEvents
    .filter((event) => event.payload.phase !== "request_submitted")
    .map((event) => str(event.payload.request_id, "")));
  const unansweredRequests = targetEvents.filter((event) => event.payload.phase === "request_submitted"
    && !savedResponses.has(str(event.payload.request_id, "")));
  const hasVerificationRecord = events.some((event) => event.event_type === "sqli_verification");
  const hasOutcomeRecord = events.some((event) => event.event_type === "sqli_episode_outcome"
    || event.event_type === "sqli_episode_summary");

  const fetchIndex = useCallback(async (cursorValue: string | null, keepSelection: boolean) => {
    setLoadingIndex(true);
    setError("");
    try {
      const result = await rangeApi.sqliEpisodes({
        runId: filters.runId.trim() || undefined,
        phase: filters.phase,
        status: filters.status,
        outcome: filters.outcome,
        cursor: cursorValue,
        limit: 50,
      });
      setPage({ items: result.items, nextCursor: result.next_cursor, hasMore: result.has_more });
      setSelectedEpisodeId((current) => keepSelection && result.items.some((item) => item.episode_id === current)
        ? current : result.items[0]?.episode_id ?? "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load the SQLi episode index.");
    } finally {
      setLoadingIndex(false);
    }
  }, [filters]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(() => {
      if (!cancelled) void fetchIndex(cursor, true);
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [cursor, fetchIndex, refreshVersion]);

  useEffect(() => {
    if (!selectedRunId || !selectedEpisodeId) return;
    let cancelled = false;

    void Promise.all([
      readTimeline(selectedRunId, selectedEpisodeId),
      rangeApi.trainingRun(selectedRunId).catch(() => null),
    ]).then(async ([timeline, run]) => {
      const started = timeline.find((event) => event.event_type === "sqli_episode_started");
      const outcome = [...timeline].reverse().find((event) => event.event_type === "sqli_episode_outcome");
      const runRecord = object(run);
      const runConfig = object(runRecord.config);
      const checkpointId = [
        started?.payload.checkpoint_run_id,
        outcome?.payload.checkpoint_run_id,
        runConfig.checkpoint_run_id,
      ].find((value): value is string => typeof value === "string" && value.length > 0);
      const checkpoint = checkpointId && checkpointId !== selectedRunId
        ? await rangeApi.trainingRun(checkpointId).catch(() => null)
        : null;
      if (cancelled) return;
      setTimelineSnapshot({ runId: selectedRunId, episodeId: selectedEpisodeId, items: timeline });
      setRunSnapshot({
        runId: selectedRunId,
        record: {
          ...runRecord,
          checkpoint_source: checkpoint ? checkpoint as unknown as Record<string, unknown> : null,
        },
      });
    }).catch((reason) => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : "Could not load persisted episode events.");
    });

    return () => {
      cancelled = true;
    };
  }, [selectedEpisodeId, selectedRunId, refreshVersion]);

  useEffect(() => {
    if (!selectedRunId || !selectedEpisodeId || selectedStatus !== "running") return;
    const timer = window.setInterval(() => {
      void readTimeline(selectedRunId, selectedEpisodeId)
        .then((timeline) => setTimelineSnapshot({ runId: selectedRunId, episodeId: selectedEpisodeId, items: timeline }))
        .catch(() => {});
      void fetchIndex(cursor, true);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [cursor, fetchIndex, selectedEpisodeId, selectedRunId, selectedStatus]);

  useEffect(() => {
    if (!selectedEpisodeId || selectedPhase !== "training" || selectedRunMode !== "train"
      || !blueUpdate || (blueUpdate.payload.updated !== true && blueUpdate.payload.update_status !== "applied")) {
      return;
    }
    let cancelled = false;
    void rangeApi.episodes(selectedRunId, { afterIndex: selectedEpisodeIndex }, 1).then(async (result) => {
      const next = result.items.find((item) => item.episode_index > selectedEpisodeIndex);
      if (!next) return;
      const firstPage = await rangeApi.episodeTimeline(selectedRunId, next.episode_id, 0, 50);
      const started = firstPage.items.find((event) => event.event_type === "sqli_episode_started");
      const version = num(object(started?.payload.policy_versions).blue);
      if (!cancelled) setNextBlueSnapshot({
        episodeId: selectedEpisodeId,
        value: { episodeIndex: next.episode_index, policyVersion: version },
      });
    }).catch(() => {
      if (!cancelled) setNextBlueSnapshot({ episodeId: selectedEpisodeId, value: null });
    });
    return () => {
      cancelled = true;
    };
  }, [blueUpdate, selectedEpisodeId, selectedEpisodeIndex, selectedPhase, selectedRunId, selectedRunMode]);

  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const timer = window.setTimeout(() => setPlayback((current) => {
      if (current.episodeId !== selectedEpisodeId) return current;
      if (current.index >= frames.length - 1) return { ...current, playing: false };
      return { ...current, index: current.index + 1 };
    }), 1000 / speed);
    return () => window.clearTimeout(timer);
  }, [frameIndex, frames.length, playing, selectedEpisodeId, speed]);

  const changeFilter = (key: keyof Filters, value: Filters[keyof Filters]) => {
    setFilters((current) => ({ ...current, [key]: value }));
    setCursor(null);
    setCursorHistory([]);
    setSelectedEpisodeId("");
  };

  const previousPage = () => {
    if (!cursorHistory.length) return;
    const previous = cursorHistory.at(-1) ?? null;
    setCursorHistory((history) => history.slice(0, -1));
    setCursor(previous);
  };

  const nextPage = () => {
    if (!page?.hasMore || !page.nextCursor) return;
    setCursorHistory((history) => [...history, cursor]);
    setCursor(page.nextCursor);
  };

  const onPlayerKeyDown = (event: React.KeyboardEvent<HTMLElement>) => {
    if (event.target instanceof HTMLElement && event.target.closest("button, input, select, summary")) return;
    if (event.key === "ArrowRight") {
      event.preventDefault();
      setPlayback((current) => ({ episodeId: selectedEpisodeId, index: Math.min(frames.length - 1, (current.episodeId === selectedEpisodeId ? current.index : 0) + 1), playing: false }));
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      setPlayback((current) => ({ episodeId: selectedEpisodeId, index: Math.max(0, (current.episodeId === selectedEpisodeId ? current.index : 0) - 1), playing: false }));
    } else if (event.key === " " || event.key.toLowerCase() === "p") {
      event.preventDefault();
      setPlayback((current) => ({
        episodeId: selectedEpisodeId,
        index: current.episodeId === selectedEpisodeId ? current.index : 0,
        playing: !(current.episodeId === selectedEpisodeId && current.playing),
      }));
    }
  };

  return <div className="episode-page">
    <header className="page-header">
      <div>
        <div className="eyebrow">Persisted SQLi lab record · read-only reconstruction</div>
        <h1 className="page-title">Episodes</h1>
        <p className="page-deck">Browse every saved training, evaluation, challenge, and other SQLi episode. Replay follows recorded server events; it never contacts the target.</p>
      </div>
      <button className="button button-secondary" onClick={() => setRefreshVersion((value) => value + 1)} disabled={loadingIndex || loadingTimeline}>
        <RefreshCw size={14} />Refresh archive
      </button>
    </header>

    {error && <div className="error-banner" role="alert"><AlertTriangle size={15} />{error}</div>}

    <section className="episode-filter-panel panel" aria-label="Episode filters">
      <label>Run ID
        <input value={runIdDraft} onChange={(event) => {
          const value = event.target.value.trim();
          setRunIdDraft(event.target.value);
          if (!value || /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value)) {
            changeFilter("runId", value);
          }
        }} placeholder="Optional exact run UUID" aria-label="Filter by exact run ID" />
      </label>
      <label>Phase
        <select value={filters.phase} onChange={(event) => changeFilter("phase", event.target.value as Filters["phase"])}>
          <option value="all">All phases</option><option value="training">Training</option><option value="evaluation">Evaluation</option><option value="challenge">Challenge</option><option value="other">Other SQLi</option>
        </select>
      </label>
      <label>Status
        <select value={filters.status} onChange={(event) => changeFilter("status", event.target.value as Filters["status"])}>
          <option value="all">All statuses</option><option value="running">Running</option><option value="completed">Completed</option><option value="incomplete">Incomplete</option><option value="failed">Failed</option>
        </select>
      </label>
      <label>Outcome
        <select value={filters.outcome} onChange={(event) => changeFilter("outcome", event.target.value as Filters["outcome"])}>
          <option value="all">All outcomes</option><option value="red">Red win</option><option value="blue">Blue win</option><option value="tie">Tie</option><option value="no_winner">No winner</option><option value="unknown">Not recorded</option>
        </select>
      </label>
      <div className="episode-index-pager">
        <button className="button button-secondary button-small" onClick={previousPage} disabled={!cursorHistory.length || loadingIndex}><ArrowLeft size={13} />Newer</button>
        <span className="pill">{page?.items.length ?? 0} episodes on this page</span>
        <button className="button button-secondary button-small" onClick={nextPage} disabled={!page?.hasMore || loadingIndex}>Older<ArrowRight size={13} /></button>
      </div>
    </section>

    <div className="episode-browser-grid">
      <aside className="panel episode-index-panel" aria-label="Persisted episode index">
        <div className="panel-head"><div><span className="panel-kicker">Newest first · cursor paginated</span><h2 className="panel-title">Episode archive</h2></div><span className="pill">SQLi</span></div>
        <div className="episode-index-list">
          {page?.items.map((item) => <EpisodeRow key={`${item.run_id}:${item.episode_id}`} item={item} selected={item.episode_id === selectedEpisodeId} onSelect={() => {
            setSelectedEpisodeId(item.episode_id);
            setPlayback({ episodeId: item.episode_id, index: 0, playing: false });
          }} />)}
          {!page?.items.length && <div className="episode-empty">{loadingIndex ? "Loading persisted SQLi episodes…" : "No episodes match these filters."}</div>}
        </div>
      </aside>

      <main className="episode-detail-column">
        {selected ? <>
          <section className="episode-summary panel">
            <div className="episode-summary-heading"><div><span className="panel-kicker">{selected.phase} · run {selected.run_id.slice(0, 8)}</span><h2>Episode #{selected.episode_index + 1}</h2></div><span className={`pill ${selected.status === "completed" ? "pill-green" : "pill-amber"}`}>{selected.status}</span></div>
            <div className="episode-data-grid">
              <DataValue label="Run ID" value={selected.run_id} />
              <DataValue label="Run mode / type" value={`${selected.run_mode} / ${selected.run_type}`} />
              <DataValue label="Run state" value={selected.run_status} />
              <DataValue label="Seed / timestamp" value={`${selected.seed ?? "—"} · ${formatTimestamp(selected.timestamp, false)}`} />
              <DataValue label="Outcome" value={selected.outcome?.winner ?? (selected.outcome_key === "unknown" ? "not recorded" : "no winner")} />
              <DataValue label="Reason" value={selected.outcome?.reason} />
            </div>
            {!selected.outcome && <div className="episode-missing">This episode has no saved outcome record. Events below show the partial evidence that was persisted.</div>}
          </section>

          {selected.phase === "training" && <section className="episode-policy-link panel">
            <div><span className="panel-kicker">Blue policy handoff</span><strong>Post-episode v{str(blueUpdate?.payload.policy_version)}</strong></div>
            {nextBlueUse
              ? <p>Next episode #{nextBlueUse.episodeIndex + 1} records Blue v{nextBlueUse.policyVersion ?? "—"} at start. {nextBlueUse.policyVersion !== null && nextBlueUse.policyVersion === num(blueUpdate?.payload.policy_version) ? "The recorded version matches this update." : "The recorded versions differ or the next episode start omitted its policy version."}</p>
              : <p>The next episode policy link is unavailable until a following episode start is persisted.</p>}
          </section>}

          <section className="episode-player-panel panel" onKeyDown={onPlayerKeyDown} tabIndex={0} aria-label="Episode event replay. Use left and right arrows to step, Space or P to play and pause.">
            <div className="panel-head"><div><span className="panel-kicker">Saved event sequence</span><h2 className="panel-title">Replay</h2></div><span className={`pill ${loadingTimeline ? "pill-amber" : ""}`}>{loadingTimeline ? "Loading timeline" : `${frames.length} recorded events`}</span></div>
            {frames.length ? <>
              {(unansweredRequests.length > 0 || !hasVerificationRecord || (selected.outcome && !hasOutcomeRecord)) && <div className="episode-missing episode-gap-notice">
                {unansweredRequests.length > 0 && <p>{unansweredRequests.length} saved request submission{unansweredRequests.length === 1 ? "" : "s"} have no correlated target response. Their response bodies are unavailable.</p>}
                {!hasVerificationRecord && <p>{selected.status === "running" ? "Independent verification has not been persisted yet." : "No independent verification event is present in this saved timeline."}</p>}
                {selected.outcome && !hasOutcomeRecord && <p>The index has an outcome summary, but the episode timeline does not include its outcome event.</p>}
              </div>}
              <div className="episode-player-controls">
                <button className="button button-secondary button-small" onClick={() => setPlayback({ episodeId: selectedEpisodeId, index: 0, playing: false })} disabled={frameIndex === 0} aria-label="Go to first event"><SkipBack size={14} /></button>
                <button className="button button-secondary button-small" onClick={() => setPlayback((current) => ({ episodeId: selectedEpisodeId, index: Math.max(0, (current.episodeId === selectedEpisodeId ? current.index : 0) - 1), playing: false }))} disabled={frameIndex === 0} aria-label="Previous event"><ArrowLeft size={14} />Previous</button>
                <button className="button button-red button-small" onClick={() => setPlayback((current) => ({ episodeId: selectedEpisodeId, index: current.episodeId === selectedEpisodeId ? current.index : 0, playing: !(current.episodeId === selectedEpisodeId && current.playing) }))} disabled={frames.length < 2} aria-label={playing ? "Pause replay" : "Play replay"}>
                  {playing ? <Pause size={13} /> : <Play size={13} />}{playing ? "Pause" : "Play"}
                </button>
                <button className="button button-secondary button-small" onClick={() => setPlayback((current) => ({ episodeId: selectedEpisodeId, index: Math.min(frames.length - 1, (current.episodeId === selectedEpisodeId ? current.index : 0) + 1), playing: false }))} disabled={frameIndex >= frames.length - 1} aria-label="Next event">Next<ArrowRight size={14} /></button>
                <button className="button button-secondary button-small" onClick={() => setPlayback({ episodeId: selectedEpisodeId, index: frames.length - 1, playing: false })} disabled={frameIndex >= frames.length - 1} aria-label="Go to last event"><SkipForward size={14} /></button>
                <label className="episode-speed">Speed<select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option><option value={4}>4×</option></select></label>
              </div>
              <div className="episode-scrubber-row"><label htmlFor="episode-scrubber">Scrub event</label><input id="episode-scrubber" type="range" min={0} max={frames.length - 1} value={frameIndex} onChange={(event) => setPlayback({ episodeId: selectedEpisodeId, index: Number(event.target.value), playing: false })} /><span>{frameIndex + 1} / {frames.length}</span></div>
              {currentFrame && <article className="episode-event-frame" aria-live="polite">
                <header className="episode-event-header">
                  <div><span className="episode-event-seq">EVENT {String(currentFrame.seq).padStart(4, "0")}</span><h3>{eventLabel(currentFrame)}</h3></div>
                  <time>{formatTimestamp(currentFrame.occurred_at)}</time>
                </header>
                <div className="episode-correlation-row">
                  <span>event {currentFrame.event_id}</span>
                  {correlationIds(currentFrame).map(([key, value]) => <span key={key}>{key.replaceAll("_", " ")} {value}</span>)}
                </div>
                <EventDetails event={currentFrame} derived={derivedTransitions.get(currentFrame.event_id)} settings={settings} />
                <JsonDetails label="Exact persisted event payload · challenge phrases redacted" value={currentFrame.payload} />
              </article>}
            </> : <div className="episode-empty">{loadingTimeline ? "Reading the persisted timeline…" : "No episode events were recorded."}</div>}
            {frames.length > 0 && <p className="episode-key-help">Keyboard: ← previous · → next · Space or P play/pause. The scrubber also supports arrow keys.</p>}
          </section>

          <section className="episode-learning-note panel">
            <div><span className="panel-kicker">Recorded update provenance</span><h2 className="panel-title">How to read the learning values</h2></div>
            {settings
              ? <>
                <p>Settings source: {settings.provenance}. The actor uses clipped PPO with γ {settings.gamma}, GAE λ {settings.gae_lambda}, clip range ±{settings.clip_range}, policy/value/entropy weights {settings.policy_loss_coefficient} / {settings.value_loss_coefficient} / {settings.entropy_coefficient}, {settings.optimizer} at {settings.learning_rate === null ? "an unavailable learning rate" : settings.learning_rate}, gradient norm cap {settings.gradient_clip_max_norm}, and {settings.optimization_passes} passes.</p>
                <p>For current runs, advantages, returns, and old/new probabilities are read from the optimizer&apos;s persisted transition diagnostics. Older events can show recomputed values and are labeled as such. Loss, entropy, gradient norm, KL, and clipping metrics come directly from the update event. A same-observation probability probe describes one state only, not general performance.</p>
                <details className="episode-json-details">
                  <summary>Exact PPO provenance values</summary>
                  <pre>{pretty({
                    algorithm: settings.algorithm,
                    actor_critic: settings.actor_critic,
                    gamma: settings.gamma,
                    gae_lambda: settings.gae_lambda,
                    clip_range: settings.clip_range,
                    policy_loss_coefficient: settings.policy_loss_coefficient,
                    value_loss_coefficient: settings.value_loss_coefficient,
                    entropy_coefficient: settings.entropy_coefficient,
                    optimizer: settings.optimizer,
                    learning_rate: settings.learning_rate,
                    gradient_clip_max_norm: settings.gradient_clip_max_norm,
                    optimization_passes: settings.optimization_passes,
                    advantage_normalization: settings.advantage_normalization,
                    provenance: settings.provenance,
                  })}</pre>
                </details>
              </>
              : <p>PPO settings were not saved for this historical run, so derived advantages and return targets are marked unavailable. The page still shows any update metrics that were persisted.</p>}
          </section>
        </> : <section className="panel episode-empty episode-detail-empty">{loadingIndex ? "Loading episode archive…" : "Choose an episode to inspect its persisted event replay."}</section>}
      </main>
    </div>
  </div>;
}
