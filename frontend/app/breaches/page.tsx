"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, ChevronLeft, ChevronRight, RefreshCw, Server, ShieldCheck } from "lucide-react";
import { rangeApi, type RangeEvent, type SqliTargetStatus, type TrainingRun } from "../../lib/range-api";

type Episode = {
  episode_id: string;
  episode_index: number;
  seed: number | null;
  status: string;
  config: Record<string, unknown>;
};
type EpisodePage = {
  runId: string;
  items: Episode[];
  hasOlder: boolean;
  hasNewer: boolean;
};
type Timeline = { runId: string; episodeId: string; items: RangeEvent[] };
type VerifiedDisclosure = {
  step: number;
  seq: number;
  occurred_at: string;
  policy_id: string;
  policy_version: number | null;
  field: string;
  base_term: string;
  close_quote: boolean;
  connector: string;
  predicate: string;
  comment: boolean;
  query_value: string;
  request_id: string;
  target_event_id: string;
  http_status: number | null;
  returned_rows: unknown[];
  unauthorized_rows: number;
};
type Breach = {
  key: string;
  run_id: string;
  episode_id: string;
  seq: number;
  occurred_at: string;
  payload: Record<string, unknown>;
  disclosure: VerifiedDisclosure;
};
type BreachGroup = { key: string; label: string; occurrences: Breach[] };
type NextBlueUse = {
  episodeIndex: number;
  policyVersion: number | null;
  action: string | null;
} | null;

const RED_HEADS = ["field", "base term", "quote", "connector", "predicate", "comment"];
const object = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const str = (value: unknown, fallback = "—"): string =>
  typeof value === "string" && value.length ? value : fallback;
const num = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
const formatTimestamp = (value: string): string => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "timestamp unavailable" : `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`;
};
function redactSecrets(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redactSecrets);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([key, item]) => [
      key,
      /phrase/i.test(key) ? "[redacted]" : redactSecrets(item),
    ]));
  }
  return value;
}

const pretty = (value: unknown): string => JSON.stringify(redactSecrets(value ?? {}), null, 2);
const stepFor = (event: RangeEvent): number | null => {
  const payload = event.payload;
  return num(payload.step_id) ?? num(object(payload.observation).step_id);
};

function isVerifiedTrainingDisclosure(event: RangeEvent): boolean {
  const payload = event.payload;
  const verification = object(payload.exposure_verification);
  return event.event_type === "sqli_episode_outcome"
    && payload.phase === "training"
    && payload.status === "completed"
    && payload.kind !== "evaluation_comparator"
    && payload.winner === "red"
    && payload.verified_exposure === true
    && verification.evidence_verified === true
    && verification.verified_exposure === true;
}

function disclosuresFromTimeline(events: RangeEvent[]): VerifiedDisclosure[] {
  const ordered = [...events].sort((left, right) => left.seq - right.seq);
  const responses = ordered.filter((event) => {
    const telemetry = object(event.payload.telemetry);
    return event.event_type === "sqli_target_response"
      && event.payload.purpose === "policy"
      && event.payload.phase !== "request_submitted"
      && telemetry.evidence_status === "verified"
      && Boolean(event.payload.target_event_id)
      && Number(telemetry.unauthorized_rows) > 0;
  });
  const disclosures: VerifiedDisclosure[] = [];
  for (const responseEvent of responses) {
    const redEvent = ordered.find((event) => event.event_type === "sqli_red_decision"
      && event.payload.decision_id === responseEvent.payload.decision_id);
    if (!redEvent) continue;
    const action = object(redEvent.payload.action ?? redEvent.payload.components);
    if (
      typeof action.field !== "string"
      || typeof action.base_term !== "string"
      || typeof action.close_quote !== "boolean"
      || typeof action.connector !== "string"
      || typeof action.predicate !== "string"
      || typeof action.comment !== "boolean"
    ) continue;
    const response = object(responseEvent.payload.response);
    const telemetry = object(responseEvent.payload.telemetry);
    const request = object(responseEvent.payload.request);
    const rows = Array.isArray(response.rows) ? response.rows : [];
    const step = num(responseEvent.payload.step_id) ?? stepFor(redEvent) ?? 0;
    disclosures.push({
      step,
      seq: responseEvent.seq,
      occurred_at: responseEvent.occurred_at,
      policy_id: str(redEvent.payload.policy_id),
      policy_version: num(redEvent.payload.policy_version),
      field: str(action.field),
      base_term: str(action.base_term),
      close_quote: action.close_quote === true,
      connector: str(action.connector, "none"),
      predicate: str(action.predicate, "none"),
      comment: action.comment === true,
      query_value: str(request.query_value, "composed value unavailable"),
      request_id: str(responseEvent.payload.request_id),
      target_event_id: str(responseEvent.payload.target_event_id),
      http_status: num(responseEvent.payload.http_status),
      returned_rows: rows,
      unauthorized_rows: num(telemetry.unauthorized_rows) ?? 0,
    });
  }
  return disclosures;
}

function disclosureGroupKey(disclosure: VerifiedDisclosure): string {
  return JSON.stringify([
    disclosure.field,
    disclosure.base_term,
    disclosure.close_quote,
    disclosure.connector,
    disclosure.predicate,
    disclosure.comment,
  ]);
}

function disclosureGroupLabel(disclosure: VerifiedDisclosure): string {
  return [
    disclosure.field,
    `base “${disclosure.base_term}”`,
    disclosure.close_quote ? "closed quote" : "open quote",
    disclosure.connector,
    disclosure.predicate,
    disclosure.comment ? "comment" : "no comment",
  ].join(" · ");
}

function groupBreaches(occurrences: Breach[]): BreachGroup[] {
  const groups = new Map<string, BreachGroup>();
  for (const occurrence of occurrences) {
    const key = disclosureGroupKey(occurrence.disclosure);
    const existing = groups.get(key);
    if (existing) existing.occurrences.push(occurrence);
    else groups.set(key, {
      key,
      label: disclosureGroupLabel(occurrence.disclosure),
      occurrences: [occurrence],
    });
  }
  return [...groups.values()].sort((left, right) =>
    right.occurrences.length - left.occurrences.length
    || Date.parse(right.occurrences[0].occurred_at) - Date.parse(left.occurrences[0].occurred_at));
}

async function getAllRunSummaryEvents(runId: string): Promise<RangeEvent[]> {
  const events: RangeEvent[] = [];
  let cursor = 0;
  for (;;) {
    const page = await rangeApi.sqliSummary(runId, cursor, 1000);
    events.push(...page.items);
    if (!page.has_more || !page.items.length || page.next_after_seq <= cursor) break;
    cursor = page.next_after_seq;
  }
  return events;
}

async function getEpisodeTimeline(runId: string, episodeId: string): Promise<RangeEvent[]> {
  const events: RangeEvent[] = [];
  let cursor = 0;
  for (;;) {
    const page = await rangeApi.episodeTimeline(runId, episodeId, cursor, 500);
    events.push(...page.items);
    if (!page.has_more || !page.items.length || page.next_after_seq <= cursor) break;
    cursor = page.next_after_seq;
  }
  return events;
}

function DataBlock({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="server-data-block"><div className="server-data-label">{label}</div>{children}</div>;
}

function Panel({ title, children, tone = "" }: {
  title: string;
  children: React.ReactNode;
  tone?: string;
}) {
  return <section className={`panel server-card ${tone}`}>
    <div className="panel-head"><h3 className="panel-title">{title}</h3></div>
    <div className="panel-body">{children}</div>
  </section>;
}

function JsonData({ label, value }: { label: string; value: unknown }) {
  return <DataBlock label={label}><pre>{pretty(value)}</pre></DataBlock>;
}

function ProbabilityHeads({ distributions, selected, names = RED_HEADS }: {
  distributions: unknown;
  selected: unknown;
  names?: string[];
}) {
  if (!Array.isArray(distributions)) return <p className="micro text-muted">Probability sample is missing.</p>;
  const selections = Array.isArray(selected) ? selected : [];
  return <div className="server-probabilities">{distributions.map((head, index) => {
    const values = Array.isArray(head) ? head.filter((item): item is number => typeof item === "number") : [];
    const picked = num(selections[index]);
    return <div className="server-probability" key={index}>
      <span>{names[index] ?? `action head ${index + 1}`}</span>
      <strong>{picked === null ? "selected probability unavailable" : `selected ${picked.toFixed(3)}`}</strong>
      <div className="server-probability-bars">{values.map((value, choice) => <i
        key={choice}
        title={`${(value * 100).toFixed(1)}%`}
        style={{ height: `${Math.max(3, Math.round(value * 42))}px` }}
      />)}</div>
    </div>;
  })}</div>;
}

function TargetResponse({ event }: { event: RangeEvent }) {
  const payload = event.payload;
  const response = object(payload.response);
  const telemetry = object(payload.telemetry);
  const request = object(payload.request);
  const rows = Array.isArray(response.rows) ? response.rows : [];
  const httpStatus = num(payload.http_status);
  return <div className="server-response">
    <div className="server-response-topline">
      <span className="pill">{str(payload.purpose, "target request")}</span>
      <strong>{str(payload.method, "POST")} {str(payload.route, "/api/lookup")}</strong>
      <span className={`pill ${httpStatus !== null && httpStatus < 400 ? "pill-green" : "pill-amber"}`}>
        HTTP {httpStatus ?? "unknown"}
      </span>
    </div>
    <div className="server-evidence-ids">
      <span>request {str(payload.request_id)}</span>
      <span>target event {str(payload.target_event_id, "missing")}</span>
      <span>executor {str(payload.status)}</span>
      <span>evidence {str(payload.evidence_status, "missing")}</span>
    </div>
    <div className="server-signal-row">
      <span>syntax signal <b>{String(Boolean(telemetry.query_syntax_signal))}</b></span>
      <span>query rejected <b>{String(Boolean(telemetry.query_rejected))}</b></span>
      <span>unauthorized rows <b>{str(telemetry.unauthorized_rows, "unknown")}</b></span>
      <span>returned rows <b>{rows.length}</b></span>
    </div>
    <details className="server-payload-details">
      <summary>Request and returned row evidence</summary>
      <JsonData label="HTTP request body" value={request} />
      <JsonData label="HTTP response body" value={response} />
      <JsonData label="Verified target telemetry" value={telemetry} />
    </details>
  </div>;
}

function TargetSubmission({ event }: { event: RangeEvent }) {
  const payload = event.payload;
  const components = object(payload.components);
  const actor = payload.purpose === "verifier_legitimate" ? "researcher" : "guest";
  return <div className="server-response server-pending-request">
    <div className="server-response-topline">
      <span className="pill pill-amber">Request submitted · awaiting target evidence</span>
      <strong>POST {str(payload.route, "/api/lookup")}</strong>
    </div>
    <div className="server-evidence-ids">
      <span>purpose {str(payload.purpose)}</span>
      <span>request {str(payload.request_id)}</span>
      <span>executor job {str(payload.job_id)}</span>
    </div>
    <JsonData label="Composed request awaiting target response" value={{
      actor_ref: actor,
      field: components.field,
      query_value: payload.composed_query_value,
    }} />
  </div>;
}

function IncidentSequence({ events, legacyRedFirst }: {
  events: RangeEvent[];
  legacyRedFirst: boolean;
}) {
  const sequence = [...events].filter((event) =>
    event.event_type === "sqli_red_decision"
    || event.event_type === "sqli_blue_decision"
    || event.event_type === "sqli_control_acknowledged"
    || (event.event_type === "sqli_target_response" && event.payload.phase !== "request_submitted")
    || event.event_type === "sqli_verification",
  ).sort((left, right) => left.seq - right.seq);
  if (!sequence.length) return <p className="server-missing">The correlated event sequence is unavailable.</p>;

  const title = (event: RangeEvent): string => {
    const payload = event.payload;
    if (event.event_type === "sqli_red_decision") return "Red chose a request from its round-start observation";
    if (event.event_type === "sqli_blue_decision") {
      return `Blue chose ${str(object(payload.selected_action).operation, "a control")} from its round-start observation`;
    }
    if (event.event_type === "sqli_control_acknowledged") {
      const ack = object(payload.acknowledgement);
      return `Blue control ${str(payload.action)} was acknowledged as ${str(ack.status)}`;
    }
    if (event.event_type === "sqli_verification") {
      const verified = payload.evidence_verified === true;
      return verified
        ? `Verifier confirmed ${payload.verified_prevention === true ? "prevention" : `${num(payload.unauthorized_rows_returned) ?? "unknown"} unauthorized rows`} and researcher access ${payload.legitimate_query_served === true ? "was available" : "was unavailable"}`
        : "Verifier evidence is incomplete";
    }
    const purpose = str(payload.purpose, "target request");
    const response = object(payload.response);
    const telemetry = object(payload.telemetry);
    const rowCount = Array.isArray(response.rows) ? response.rows.length : num(telemetry.row_count);
    if (purpose === "policy") {
      return `Target answered Red's request · HTTP ${str(payload.http_status, "unknown")} · ${rowCount ?? "unknown"} rows · ${num(telemetry.unauthorized_rows) ?? "unknown"} unauthorized`;
    }
    if (purpose === "verifier_attack") {
      return `Independent attack probe · HTTP ${str(payload.http_status, "unknown")} · ${num(telemetry.unauthorized_rows) ?? "unknown"} unauthorized rows`;
    }
    return `Researcher lookup probe · HTTP ${str(payload.http_status, "unknown")} · ${rowCount ?? "unknown"} rows`;
  };

  return <ol className="incident-sequence" aria-label="Correlated request and verification sequence">
    {sequence.map((event) => <li key={event.event_id}>
      <span className={`incident-sequence-dot ${event.event_type.includes("blue") || event.event_type.includes("control") ? "is-blue" : event.event_type.includes("red") ? "is-red" : ""}`} />
      <div><strong>{title(event)}</strong><small>event #{event.seq}{event.payload.request_id ? ` · request ${str(event.payload.request_id)}` : ""}{event.payload.target_event_id ? ` · target ${str(event.payload.target_event_id)}` : ""}</small></div>
    </li>)}
    {legacyRedFirst && <li className="incident-sequence-note"><span className="incident-sequence-dot is-legacy" /><div><strong>Legacy ordering: Red&apos;s request reached the target before Blue acted.</strong><small>Blue&apos;s later control could not reverse an earlier disclosure.</small></div></li>}
  </ol>;
}

function Round({ step, events }: { step: number; events: RangeEvent[] }) {
  const redEvent = events.find((event) => event.event_type === "sqli_red_decision");
  const blueEvent = events.find((event) => event.event_type === "sqli_blue_decision");
  const red = object(redEvent?.payload);
  const blue = object(blueEvent?.payload);
  const redObservation = object(red.observation);
  const blueObservation = object(blue.observation);
  const action = object(red.action);
  const redDecisionId = str(red.decision_id, "");
  const blueDecisionId = str(blue.decision_id, "");
  const targetEvents = events.filter((event) => event.event_type === "sqli_target_response");
  const submissions = targetEvents.filter((event) => event.payload.phase === "request_submitted");
  const responses = targetEvents.filter((event) => event.payload.phase !== "request_submitted");
  const policyRequest = responses.find((event) => event.payload.purpose === "policy");
  const control = events.find((event) => event.event_type === "sqli_control_acknowledged"
    && (str(event.payload.decision_id, "") === blueDecisionId || stepFor(event) === step));
  const verificationEvent = events.find((event) => event.event_type === "sqli_verification");
  const verification = object(verificationEvent?.payload);
  const rewardEvent = events.find((event) => event.event_type === "sqli_reward");
  const reward = object(rewardEvent?.payload);
  const redReward = object(reward.red);
  const blueReward = object(reward.blue);
  const transitions = events.filter((event) => event.event_type === "sqli_transition");
  const hasTargetEvidence = ["policy", "verifier_attack", "verifier_legitimate"].every((purpose) => {
    const response = responses.find((event) => event.payload.purpose === purpose);
    return response?.payload.evidence_status === "verified" && Boolean(response.payload.target_event_id);
  });
  const verificationComplete = verification.evidence_verified === true;
  const monitorNeedsNoControl = str(object(blue.selected_action).operation) === "MONITOR";
  const hasControlEvidence = Boolean(control) || monitorNeedsNoControl;
  const complete = Boolean(redEvent && blueEvent && rewardEvent && verificationEvent
    && hasControlEvidence && hasTargetEvidence && verificationComplete);
  const ack = object(control?.payload);
  const acknowledgement = object(ack.acknowledgement);
  const activeControls = object(acknowledgement.controls);
  const selected = Array.isArray(red.selected_component_probabilities)
    ? red.selected_component_probabilities : [];
  const quote = action.close_quote === true ? "close quote" : "leave quote open";
  const composedSummary = [
    str(action.field), str(action.base_term), quote, str(action.connector, "none"),
    str(action.predicate, "none"), action.comment === true ? "comment on" : "comment off",
  ].join(" · ");
  const attackVerified = verification.verified_prevention === true;
  const legitimateAvailable = verification.legitimate_query_served === true;
  const summary = `${redEvent ? composedSummary : "Red decision missing"} / ${str(object(blue.selected_action).operation, "Blue decision missing")}`;

  return <details className={`panel server-round ${complete ? "is-complete" : "is-partial"}`} open={step === 0}>
    <summary className="server-round-summary">
      <span className="server-round-number">ROUND {String(step + 1).padStart(2, "0")}</span>
      <span className="server-round-brief">{summary}</span>
      <span className={`pill ${complete ? "pill-green" : "pill-amber"}`}>
        {complete ? "Evidence complete" : "Partial / incomplete"}
      </span>
    </summary>
    <div className="server-round-grid">
      <Panel title="RED · observation and request" tone="server-red">
        <JsonData label="Policy-visible observation" value={redObservation} />
        <DataBlock label="Selected SQL grammar fields">
          <div className="server-action-fields">
            <span>Field <b>{str(action.field)}</b></span>
            <span>Base term <b>{str(action.base_term)}</b></span>
            <span>Quote <b>{quote}</b></span>
            <span>Connector <b>{str(action.connector, "none")}</b></span>
            <span>Predicate <b>{str(action.predicate, "none")}</b></span>
            <span>Comment <b>{action.comment === true ? "on" : action.comment === false ? "off" : "unknown"}</b></span>
          </div>
          <div className="server-composed-request"><span>Composed selection</span><strong>{composedSummary}</strong></div>
        </DataBlock>
        <DataBlock label="Selected action probabilities">
          <ProbabilityHeads distributions={red.component_distributions} selected={selected} />
        </DataBlock>
        <JsonData label="Composed HTTP request sent to target" value={policyRequest ? {
          method: policyRequest.payload.method ?? "POST",
          route: policyRequest.payload.route ?? "/api/lookup",
          body: policyRequest.payload.request,
        } : { status: "target request missing" }} />
        <p className="micro text-muted">Decision {str(red.decision_id)} · policy v{str(red.policy_version)}</p>
        {transitions.find((event) => event.payload.role === "red") && <JsonData
          label="Accepted transition"
          value={(() => { const transition = transitions.find((event) => event.payload.role === "red"); const p = object(transition?.payload); return { accepted: p.accepted, learning_eligible: p.learning_eligible, reward: p.reward, reward_components: p.reward_components, old_log_probability: p.old_log_probability, value_estimate: p.value_estimate }; })()}
        />}
      </Panel>

      <Panel title="TARGET · actual HTTP exchange">
        {responses.length ? <div className="server-response-list">{responses.map((event) =>
          <TargetResponse key={event.event_id} event={event} />)}</div>
          : submissions.length ? <div className="server-response-list">{submissions.map((event) =>
            <TargetSubmission key={event.event_id} event={event} />)}</div>
            : <div className="server-missing">No target request or response was recorded for this round.</div>}
        {redDecisionId && <p className="micro text-muted">Red decision correlation ID: {redDecisionId}</p>}
      </Panel>

      <Panel title="BLUE · control and verification" tone="server-blue">
        <JsonData label="Policy-visible observation" value={blueObservation} />
        <DataBlock label="Selected control">
          <strong className="server-control-name">{str(object(blue.selected_action).operation, "Blue decision missing")}</strong>
          <p className="micro">Decision {str(blue.decision_id)} · policy v{str(blue.policy_version)}</p>
          <ProbabilityHeads distributions={blue.component_distributions} selected={blue.selected_component_probabilities} names={["Blue control"]} />
        </DataBlock>
        <DataBlock label="Control acknowledgment">
          {control ? <>
            <div className="server-signal-row"><span>action <b>{str(ack.action)}</b></span><span>status <b>{str(acknowledgement.status)}</b></span></div>
            <JsonData label="Active target settings" value={activeControls} />
          </> : monitorNeedsNoControl ? <p>No command required for MONITOR. Older records may not include a persisted no-op acknowledgment.</p>
            : <div className="server-missing">Control acknowledgment is missing.</div>}
        </DataBlock>
        <DataBlock label="Independent verification">
          {verificationEvent ? <>
            <div className="server-verification-grid">
              <div><span>Attack probe</span><b>HTTP {str(verification.attack_probe_status)} · {attackVerified ? "prevention verified" : "not prevented"}</b></div>
              <div><span>Unauthorized rows</span><b>{str(verification.unauthorized_rows_returned, "unknown")}</b></div>
              <div><span>Legitimate lookup</span><b>HTTP {str(verification.legitimate_query_status)} · {legitimateAvailable ? "available" : "not verified"}</b></div>
              <div><span>Evidence join</span><b>{verificationComplete ? "both probes verified" : "partial / missing"}</b></div>
            </div>
            <JsonData label="Verification record" value={verification} />
          </> : <div className="server-missing">Verifier evidence is missing; outcome remains incomplete.</div>}
        </DataBlock>
      </Panel>

      <Panel title="REWARD · round learning signal">
        {rewardEvent ? <>
          <div className="server-reward-totals">
            <div><span>Red</span><strong>{num(redReward.total)?.toFixed(3) ?? "—"}</strong></div>
            <div><span>Blue</span><strong>{num(blueReward.total)?.toFixed(3) ?? "—"}</strong></div>
            <small>{str(redReward.reward_version, str(rewardEvent.payload.schema_id))}</small>
          </div>
          <div className="server-learning-grid">
            <JsonData label="Red reward components" value={redReward.components} />
            <JsonData label="Blue reward components" value={blueReward.components} />
          </div>
          {transitions.map((event) => {
            const transition = event.payload;
            return <div className="server-transition" key={event.event_id}>
              <strong>{str(transition.role)} transition · {transition.accepted === true ? "accepted" : "not accepted"}</strong>
              <span>{transition.learning_eligible === true ? "included in PPO batch" : "excluded from PPO batch"} · reward {num(transition.reward)?.toFixed(3) ?? "—"}</span>
            </div>;
          })}
        </> : <div className="server-missing">Reward breakdown is missing for this round.</div>}
      </Panel>
    </div>
  </details>;
}

function ProbabilitySnapshot({ event }: { event: RangeEvent }) {
  const payload = event.payload;
  return <Panel title={`${str(payload.role)} · same-reference policy probe`}>
    <p className="micro">Policy v{str(payload.policy_version_before)} → v{str(payload.policy_version_after)} · same observation: {String(payload.same_reference_observation === true)}</p>
    <JsonData label="Reference observation features" value={payload.reference_observation} />
    <JsonData label="Before → after action probabilities" value={{
      before: payload.before_action_probabilities,
      after: payload.after_action_probabilities,
    }} />
  </Panel>;
}

function UpdateReport({ event }: { event: RangeEvent }) {
  const metrics = object(event.payload.metrics);
  return <Panel title={`${str(event.payload.role)} PPO · ${str(event.payload.update_status)}`}>
    <p className="micro">Policy {str(event.payload.policy_id)} · v{str(event.payload.policy_version_before)} → v{str(event.payload.policy_version)}</p>
    <div className="server-update-stats">
      <div><span>Accepted transitions</span><b>{num(metrics.accepted_transitions) ?? num(metrics.sample_count) ?? 0}</b></div>
      <div><span>Optimization passes</span><b>{num(event.payload.optimization_passes) ?? num(metrics.optimizer_steps) ?? 0}</b></div>
      <div><span>Policy version</span><b>v{str(event.payload.policy_version)}</b></div>
      <div><span>Checkpoint SHA-256</span><b className="server-digest">{str(event.payload.checkpoint_sha256, "not recorded")}</b></div>
    </div>
    <JsonData label="Post-update PPO metrics" value={{
      loss: metrics.loss,
      policy_loss: metrics.policy_loss,
      value_loss: metrics.value_loss,
      entropy: metrics.entropy,
      gradient_norm: metrics.gradient_norm,
      post_update_kl: metrics.post_update_approx_kl ?? metrics.approx_kl,
      post_update_clip_fraction: metrics.post_update_clip_fraction ?? metrics.clip_fraction,
    }} />
  </Panel>;
}

async function getAllTrainingRuns(): Promise<TrainingRun[]> {
  const all: TrainingRun[] = [];
  let offset = 0;
  for (;;) {
    const page = await rangeApi.trainingRuns(100, offset);
    all.push(...page.items);
    if (!page.has_more || page.next_offset <= offset) break;
    offset = page.next_offset;
  }
  return all;
}

export default function ServerPage() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [breaches, setBreaches] = useState<Breach[]>([]);
  const [runId, setRunId] = useState("");
  const [episodePage, setEpisodePage] = useState<EpisodePage | null>(null);
  const [episodeId, setEpisodeId] = useState("");
  const [selectedBreachKey, setSelectedBreachKey] = useState("");
  const [selectedRecipeKey, setSelectedRecipeKey] = useState("");
  const [patternFilter, setPatternFilter] = useState("all");
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [nextBlueUse, setNextBlueUse] = useState<NextBlueUse>(null);
  const [target, setTarget] = useState<SqliTargetStatus | null>(null);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [loadingBreaches, setLoadingBreaches] = useState(true);
  const [loadingEpisodes, setLoadingEpisodes] = useState(false);
  const [unclassifiedCount, setUnclassifiedCount] = useState(0);
  const [error, setError] = useState("");
  const [refreshVersion, setRefreshVersion] = useState(0);

  const run = useMemo(() => runs.find((item) => item.training_run_id === runId) ?? null, [runs, runId]);
  const selectedBreach = breaches.find((item) => item.key === selectedBreachKey) ?? null;
  const episode = episodePage?.runId === runId
    ? episodePage.items.find((item) => item.episode_id === episodeId)
      ?? (selectedBreach?.run_id === runId && selectedBreach.episode_id === episodeId
        ? {
          episode_id: selectedBreach.episode_id,
          episode_index: Math.max(0, (num(selectedBreach.payload.episode_index) ?? 1) - 1),
          seed: num(selectedBreach.payload.seed),
          status: str(selectedBreach.payload.status, "completed"),
          config: {},
        }
        : null)
    : null;
  const events = useMemo(() => timeline?.runId === runId && timeline.episodeId === episodeId
    ? timeline.items : [], [timeline, runId, episodeId]);
  const rounds = useMemo(() => {
    const grouped = new Map<number, RangeEvent[]>();
    for (const event of events) {
      const step = stepFor(event);
      if (step === null || step < 0) continue;
      grouped.set(step, [...(grouped.get(step) ?? []), event]);
    }
    return [...grouped.entries()].sort(([left], [right]) => left - right);
  }, [events]);
  const probes = events.filter((event) => event.event_type === "sqli_policy_probe");
  const updates = events.filter((event) => event.event_type === "sqli_ppo_update");
  const outcome = [...events].reverse().find((event) => event.event_type === "sqli_episode_outcome");
  const summary = [...events].reverse().find((event) => event.event_type === "sqli_episode_summary");
  const outcomePayload = object(outcome?.payload ?? summary?.payload);
  const cursorExhausted = timeline?.runId === runId && timeline.episodeId === episodeId;
  const loadingTimeline = Boolean(runId && episodeId && !cursorExhausted);
  const runArchiveKey = runs.map((item) => `${item.training_run_id}:${item.status}`).join("|");
  const breachGroups = useMemo(() => groupBreaches(breaches), [breaches]);
  const filteredBreachGroups = patternFilter === "all"
    ? breachGroups
    : breachGroups.filter((item) => item.key === patternFilter);
  const episodeStart = events.find((event) => event.event_type === "sqli_episode_started");
  const protocolId = str(
    episodeStart?.payload.interaction_protocol_schema_id,
    str(run?.config.interaction_protocol_schema_id, ""),
  );
  const legacyRedFirst = protocolId !== "sqli.round_protocol.v2";
  const selectedDisclosure = selectedBreach?.run_id === runId && selectedBreach.episode_id === episodeId
    ? selectedBreach.disclosure : null;
  const selectedStepEvents = selectedDisclosure
    ? rounds.find(([step]) => step === selectedDisclosure.step)?.[1] ?? []
    : [];
  const selectedBlueDecision = selectedStepEvents.find((event) => event.event_type === "sqli_blue_decision");
  const selectedControlAck = selectedStepEvents.find((event) => event.event_type === "sqli_control_acknowledged");
  const selectedPolicyResponse = selectedStepEvents.find((event) =>
    event.event_type === "sqli_target_response"
    && event.payload.purpose === "policy"
    && str(event.payload.target_event_id, "") === selectedDisclosure?.target_event_id,
  );
  const selectedVerification = selectedStepEvents.find((event) => event.event_type === "sqli_verification");
  const controlStatus = str(object(selectedControlAck?.payload.acknowledgement).status, "unavailable");
  const controlBeforeRequest = Boolean(
    selectedControlAck && selectedPolicyResponse && selectedControlAck.seq < selectedPolicyResponse.seq,
  );
  const selectedRunId = run?.training_run_id ?? null;
  const selectedEpisodeId = episode?.episode_id ?? null;
  const selectedEpisodeIndex = episode?.episode_index ?? null;

  const loadRuns = useCallback(async (preserveCurrent = true) => {
    setLoadingRuns(true);
    setError("");
    try {
      const all = await getAllTrainingRuns();
      const sqli = all.filter((item) => item.config.scenario_id === "sqli_lab_v1");
      setRuns(sqli);
      if (!preserveCurrent) setSelectedBreachKey("");
      setRunId((current) => preserveCurrent && sqli.some((item) => item.training_run_id === current)
        ? current : sqli[0]?.training_run_id ?? "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load the training archive.");
    } finally {
      setLoadingRuns(false);
    }
  }, []);

  const loadEpisodePage = useCallback(async (
    selectedRunId: string,
    cursor: { afterIndex?: number; beforeIndex?: number } = {},
    preserveEpisode = false,
  ) => {
    if (!selectedRunId) return;
    setLoadingEpisodes(true);
    try {
      const result = await rangeApi.episodes(selectedRunId, cursor, 100);
      const items = result.items;
      setEpisodePage({
        runId: selectedRunId,
        items,
        hasOlder: result.has_older,
        hasNewer: result.has_newer,
      });
      setEpisodeId((current) => preserveEpisode && current
        ? current : items.at(-1)?.episode_id ?? "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load episodes for this run.");
    } finally {
      setLoadingEpisodes(false);
    }
  }, []);

  useEffect(() => {
    if (!runs.length) {
      const timer = window.setTimeout(() => {
        setBreaches([]);
        setLoadingBreaches(false);
      }, 0);
      return () => window.clearTimeout(timer);
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void (async () => {
        setLoadingBreaches(true);
        try {
          const trainingRuns = runs.filter((item) => item.mode === "train");
          const summaryPages = await Promise.all(trainingRuns.map(async (item) => ({
            runId: item.training_run_id,
            events: await getAllRunSummaryEvents(item.training_run_id),
          })));
          const candidates = summaryPages.flatMap(({ runId: sourceRunId, events: summaryEvents }) =>
            summaryEvents.filter(isVerifiedTrainingDisclosure).map((event) => ({
              key: `${sourceRunId}:${event.event_id}`,
              run_id: sourceRunId,
              episode_id: event.episode_id ?? str(event.payload.episode_id, ""),
              seq: event.seq,
              occurred_at: event.occurred_at,
              payload: event.payload,
            })).filter((item) => item.episode_id.length > 0));
          const hydratedPages = await Promise.all(candidates.map(async (candidate) => {
            try {
              const disclosures = disclosuresFromTimeline(
                await getEpisodeTimeline(candidate.run_id, candidate.episode_id),
              );
              return disclosures.map((disclosure) => ({
                ...candidate,
                key: `${candidate.run_id}:${candidate.episode_id}:${disclosure.target_event_id}`,
                seq: disclosure.seq,
                occurred_at: disclosure.occurred_at,
                disclosure,
              }));
            } catch {
              return [];
            }
          }));
          const hydrated = hydratedPages.flat();
          if (cancelled) return;
          hydrated.sort((left, right) =>
            Date.parse(right.occurred_at) - Date.parse(left.occurred_at) || right.seq - left.seq);
          setBreaches(hydrated);
          setUnclassifiedCount(candidates.length - hydratedPages.filter((items) => items.length > 0).length);
          const nextGroups = groupBreaches(hydrated);
          setSelectedRecipeKey((current) => nextGroups.some((item) => item.key === current)
            ? current : nextGroups[0]?.key ?? "");
          setSelectedBreachKey((current) => hydrated.some((item) => item.key === current)
            ? current : nextGroups[0]?.occurrences[0]?.key ?? "");
        } catch (reason) {
          if (!cancelled) setError(reason instanceof Error ? reason.message : "Could not classify verified disclosures.");
        } finally {
          if (!cancelled) setLoadingBreaches(false);
        }
      })();
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [runArchiveKey, refreshVersion, runs]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadRuns(false), 0);
    return () => window.clearTimeout(timer);
  }, [loadRuns]);

  useEffect(() => {
    if (!runId) return;
    const preserveBreachEpisode = selectedBreach?.run_id === runId;
    const timer = window.setTimeout(() => void loadEpisodePage(runId, {}, preserveBreachEpisode), 0);
    return () => window.clearTimeout(timer);
  }, [runId, loadEpisodePage, refreshVersion, selectedBreachKey, selectedBreach?.run_id]);

  useEffect(() => {
    if (!runId || !episodeId) return;
    let cancelled = false;
    let cursor = 0;
    let accumulated: RangeEvent[] = [];
    let pollTimer: number | undefined;
    const loadTimeline = async (afterSeq: number, append: boolean) => {
      let nextCursor = afterSeq;
      const newItems: RangeEvent[] = [];
      for (;;) {
        const page = await rangeApi.episodeTimeline(runId, episodeId, nextCursor, 500);
        if (cancelled) return;
        newItems.push(...page.items);
        if (!page.has_more || !page.items.length || page.next_after_seq <= nextCursor) {
          nextCursor = page.next_after_seq;
          break;
        }
        nextCursor = page.next_after_seq;
      }
      cursor = Math.max(cursor, nextCursor);
      accumulated = append
        ? [...accumulated, ...newItems.filter((item) => !accumulated.some((old) => old.event_id === item.event_id))]
        : newItems;
      setTimeline({ runId, episodeId, items: accumulated });
    };

    void loadTimeline(0, false).then(() => {
      if (cancelled) return;
      if (run?.status === "running" || run?.status === "queued") {
        pollTimer = window.setInterval(() => {
          void loadTimeline(cursor, true).catch((reason) => {
            if (!cancelled) setError(reason instanceof Error ? reason.message : "Live timeline reconnect failed.");
          });
        }, 4000);
      }
    }).catch((reason) => {
      if (!cancelled) {
        setError(reason instanceof Error ? reason.message : "Could not load the episode timeline.");
      }
    });
    return () => {
      cancelled = true;
      if (pollTimer !== undefined) window.clearInterval(pollTimer);
    };
  }, [runId, episodeId, run?.status]);

  useEffect(() => {
    let cancelled = false;
    if (!selectedRunId || selectedEpisodeId === null || selectedEpisodeIndex === null) {
      const timer = window.setTimeout(() => {
        if (!cancelled) setNextBlueUse(null);
      }, 0);
      return () => { cancelled = true; window.clearTimeout(timer); };
    }
    const loadNextBlueUse = async () => {
      try {
        const page = await rangeApi.episodes(selectedRunId, { afterIndex: selectedEpisodeIndex }, 1);
        const nextEpisode = page.items.find((item) => item.episode_index > selectedEpisodeIndex);
        if (!nextEpisode) {
          if (!cancelled) setNextBlueUse(null);
          return;
        }
        const nextEvents = await getEpisodeTimeline(selectedRunId, nextEpisode.episode_id);
        const startEvent = nextEvents.find((event) => event.event_type === "sqli_episode_started");
        const blueDecision = nextEvents.find((event) => event.event_type === "sqli_blue_decision");
        if (!cancelled) setNextBlueUse({
          episodeIndex: nextEpisode.episode_index,
          policyVersion: num(object(startEvent?.payload.policy_versions).blue)
            ?? num(blueDecision?.payload.policy_version),
          action: blueDecision
            ? str(object(blueDecision.payload.selected_action).operation, "action unavailable")
            : null,
        });
      } catch {
        if (!cancelled) setNextBlueUse(null);
      }
    };
    void loadNextBlueUse();
    return () => { cancelled = true; };
  }, [selectedRunId, selectedEpisodeId, selectedEpisodeIndex]);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await rangeApi.sqliTargetStatus();
        if (!cancelled) setTarget(status);
        if (runId) {
          const latest = await rangeApi.trainingRun(runId);
          if (!cancelled) setRuns((current) => current.map((item) =>
            item.training_run_id === runId ? latest : item));
        }
      } catch {
        if (!cancelled) setError("Connection interrupted. Refresh to reconnect to the archive.");
      }
    };
    const timer = window.setInterval(() => void poll(), 5000);
    const initial = window.setTimeout(() => void poll(), 0);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      window.clearTimeout(initial);
    };
  }, [runId]);

  async function refreshArchive() {
    const selectedRun = runId;
    const selectedEpisode = episodeId;
    await loadRuns(true);
    if (selectedRun) {
      await loadEpisodePage(selectedRun, {}, true);
      if (selectedEpisode) setEpisodeId(selectedEpisode);
    }
    setRefreshVersion((value) => value + 1);
  }

  async function olderEpisodes() {
    const first = episodePage?.items[0];
    if (runId && first) await loadEpisodePage(runId, { beforeIndex: first.episode_index });
  }

  async function newerEpisodes() {
    const last = episodePage?.items.at(-1);
    if (runId && last) await loadEpisodePage(runId, { afterIndex: last.episode_index });
  }

  function openBreach(breach: Breach) {
    const recipeKey = disclosureGroupKey(breach.disclosure);
    setPatternFilter(recipeKey);
    setSelectedRecipeKey(recipeKey);
    setSelectedBreachKey(breach.key);
    setRunId(breach.run_id);
    setEpisodeId(breach.episode_id);
  }

  function selectRecipe(group: BreachGroup) {
    setSelectedRecipeKey(group.key);
    const newest = group.occurrences[0];
    if (newest) openBreach(newest);
  }

  return <div className="server-page">
    <header className="page-header">
      <div>
        <div className="eyebrow">Training archive · target verified disclosures</div>
        <h1 className="page-title">Breaches</h1>
        <p className="page-deck">Read each verified disclosure as an incident report: Red&apos;s request, the target response, the verified impact, and when Blue&apos;s control took effect. Older Red-first episodes are labeled.</p>
      </div>
      <button className="button button-secondary" onClick={() => void refreshArchive()} disabled={loadingRuns || loadingEpisodes}>
        <RefreshCw size={14} />Refresh archive
      </button>
    </header>

    {error && <div className="error-banner" role="alert"><AlertTriangle size={15} />{error}</div>}

    <section className="server-console-status panel" aria-label="Target and archive status">
      <div className="server-console-mark"><Server size={17} /><span>RANGE / 02</span></div>
      <div><span>SQLi target</span><strong className={target?.status === "available" ? "is-healthy" : "is-unhealthy"}>{target?.status ?? "checking"}</strong></div>
      <div><span>Target version</span><strong>{str(target?.target_version, "sqli_target_v1")}</strong></div>
      <div><span>Executor health age</span><strong>{target?.last_health_check_age_seconds == null ? "—" : `${target.last_health_check_age_seconds.toFixed(1)}s`}</strong></div>
      <div><span>Selected run</span><strong>{run?.status ?? (loadingRuns ? "loading" : "choose a run")}</strong></div>
      <div><span>Archive mode</span><strong>Read only</strong></div>
    </section>

    <section className="panel breach-index" aria-label="Verified disclosure archive">
      <div className="panel-head">
        <div><span className="panel-kicker">Source: persisted episode outcomes and target evidence</span><h2 className="panel-title">Verified Red disclosures</h2></div>
        <span className="pill pill-red">{loadingBreaches ? "classifying" : `${breachGroups.length} recipes · ${breaches.length} occurrences`}</span>
      </div>
      <div className="breach-index-note">Only completed training episodes with a Red win and verified unauthorized target rows appear here. Evaluation matches, challenge episodes, incomplete episodes, and unverified responses are excluded.</div>
      <div className="breach-filter-row" role="group" aria-label="Group disclosures by SQL grammar">
        <button className={`breach-filter ${patternFilter === "all" ? "is-selected" : ""}`} onClick={() => setPatternFilter("all")}>
          All recipes <strong>{breachGroups.length}</strong>
        </button>
        {breachGroups.map((group) => <button
          className={`breach-filter ${patternFilter === group.key ? "is-selected" : ""}`}
          key={group.key}
          onClick={() => setPatternFilter(group.key)}
        >{group.label}<strong>{group.occurrences.length}</strong></button>)}
      </div>
      {filteredBreachGroups.length ? <div className="breach-list" role="list" aria-label="Attack recipe index">
        {filteredBreachGroups.map((group) => {
          const selected = selectedRecipeKey === group.key;
          const newest = group.occurrences[0];
          return <article className={`breach-recipe-card ${selected ? "is-selected" : ""}`} key={group.key} role="listitem">
            <button className="breach-recipe-button" onClick={() => selectRecipe(group)} aria-pressed={selected} aria-label={`Select recipe ${group.label}`}>
              <span className="breach-recipe-name"><strong className="breach-pattern">{group.label}</strong><small>six-field attack recipe</small></span>
              <span className="breach-card-meta">{group.occurrences.length} verified occurrence{group.occurrences.length === 1 ? "" : "s"}</span>
              <span className="breach-card-meta">Newest · episode #{num(newest.payload.episode_index) ?? "—"} · HTTP {newest.disclosure.http_status ?? "—"}</span>
              <time>{formatTimestamp(newest.occurred_at)}</time>
            </button>
            {selected && <div className="breach-occurrence-list" aria-label={`Occurrences of ${group.label}`}>
              {group.occurrences.map((breach, index) => <button
                className={`breach-occurrence-link ${selectedBreachKey === breach.key ? "is-selected" : ""}`}
                key={breach.key}
                onClick={() => openBreach(breach)}
                aria-current={selectedBreachKey === breach.key ? "true" : undefined}
              >
                <span>{index === 0 ? "Newest" : `Occurrence ${index + 1}`} · episode #{num(breach.payload.episode_index) ?? "—"}</span>
                <small>{formatTimestamp(breach.occurred_at)} · policy v{breach.disclosure.policy_version ?? "—"}</small>
                <small>run {breach.run_id.slice(0, 8)} · request {breach.disclosure.request_id.slice(0, 8)}</small>
              </button>)}
            </div>}
          </article>;
        })}
      </div> : <div className="server-empty">{loadingBreaches ? "Loading persisted disclosure outcomes…" : "No target-verified training disclosures are recorded."}</div>}
      {unclassifiedCount > 0 && <div className="server-missing breach-index-note">{unclassifiedCount} verified outcome{unclassifiedCount === 1 ? "" : "s"} lack enough persisted action and target-response detail to assign an exact six-field recipe.</div>}
    </section>

    <section className="server-toolbar panel" aria-label="Training archive selectors">
      <label>Training run
        <select value={runId} onChange={(event) => {
          setSelectedBreachKey("");
          setEpisodeId("");
          setRunId(event.target.value);
        }}>
          <option value="">Select a run</option>
          {runs.map((item) => <option value={item.training_run_id} key={item.training_run_id}>
            {!item.config.interaction_protocol_schema_id ? "Legacy Red-first · " : ""}
            {item.config.campaign_revision ? `${String(item.config.campaign_revision)} · ` : ""}
            {item.status} · {item.created_at.slice(0, 16).replace("T", " ")} · {item.training_run_id.slice(0, 8)}
          </option>)}
        </select>
      </label>
      <label>Episode archive · newest page first
        <select value={episodeId} onChange={(event) => {
          setSelectedBreachKey("");
          setEpisodeId(event.target.value);
        }} disabled={!episodePage?.items.length}>
          {!episodePage?.items.length && <option value="">{loadingEpisodes ? "Loading episodes…" : "No episodes yet"}</option>}
          {episode && episodePage?.items.every((item) => item.episode_id !== episode.episode_id) && <option value={episode.episode_id}>
            #{episode.episode_index + 1} · selected verified breach · seed {episode.seed ?? "—"}
          </option>}
          {episodePage?.items.map((item) => <option value={item.episode_id} key={item.episode_id}>
            #{item.episode_index + 1} · {item.status} · seed {item.seed ?? "—"}
          </option>)}
        </select>
      </label>
      <div className="server-archive-controls">
        <button className="button button-secondary button-small" onClick={() => void olderEpisodes()} disabled={!episodePage?.hasOlder || loadingEpisodes}>
          <ChevronLeft size={13} />Older
        </button>
        <span className="pill">{episodePage?.items.length
          ? `episodes ${episodePage.items[0].episode_index + 1}–${episodePage.items.at(-1)!.episode_index + 1}`
          : "episode index unavailable"}</span>
        <button className="button button-secondary button-small" onClick={() => void newerEpisodes()} disabled={!episodePage?.hasNewer || loadingEpisodes}>
          Newer<ChevronRight size={13} />
        </button>
      </div>
    </section>

    {run && episode && <>
      <section className="server-meta panel">
        <div><span>RUN ID</span><strong>{run.training_run_id}</strong></div>
        <div><span>RUN STATE</span><strong>{run.status}{run.stop_requested ? " · stop requested" : ""}</strong></div>
        <div><span>EPISODE / SEED</span><strong>#{episode.episode_index + 1} · {episode.seed ?? "—"}</strong></div>
        <div><span>EPISODE STATE</span><strong>{outcomePayload.status ? str(outcomePayload.status) : episode.status}</strong></div>
      </section>

      {selectedDisclosure && <section className={`panel incident-report ${legacyRedFirst ? "is-legacy" : ""}`} aria-label="Disclosure incident report">
        <div className="incident-report-heading">
          <div><span className="panel-kicker">Incident report · round {selectedDisclosure.step + 1}</span><h2>{legacyRedFirst ? "Legacy Red-first episode" : "Blue control before Red request"}</h2></div>
          <span className={`pill ${legacyRedFirst ? "pill-amber" : "pill-green"}`}>
            {legacyRedFirst ? "Older protocol" : "Round-start decisions"}
          </span>
        </div>
        <div className="incident-narrative-grid">
          <article><span>Red&apos;s request</span><p>Red sent a guest lookup for <b>{selectedDisclosure.field}</b> using “{selectedDisclosure.query_value}”.</p></article>
          <article><span>Target response and impact</span><p>The target returned HTTP <b>{selectedDisclosure.http_status ?? "unknown"}</b> with <b>{selectedDisclosure.returned_rows.length}</b> rows. Correlated target evidence verified <b>{selectedDisclosure.unauthorized_rows}</b> unauthorized rows.</p></article>
          <article><span>Blue&apos;s control timing</span><p>{legacyRedFirst
            ? `Blue chose ${str(object(selectedBlueDecision?.payload.selected_action).operation, "a control")} after Red's request had already reached the target. The later control could not undo that disclosure.`
            : selectedControlAck
              ? `Blue chose ${str(object(selectedBlueDecision?.payload.selected_action).operation, "a control")} from its round-start observation. The target acknowledged it as ${controlStatus} ${controlBeforeRequest ? "before Red's request" : "with timing that does not precede Red's request"}.`
              : "Blue's control acknowledgement is unavailable, so its effect timing cannot be confirmed."}</p></article>
          <article><span>Independent verification</span><p>{selectedVerification
            ? object(selectedVerification.payload).evidence_verified === true
              ? `The verifier recorded ${selectedVerification.payload.verified_prevention === true ? "prevention" : `${num(selectedVerification.payload.unauthorized_rows_returned) ?? "unknown"} unauthorized rows`} and researcher lookup ${selectedVerification.payload.legitimate_query_served === true ? "available" : "unavailable"}.`
              : "Verifier evidence is incomplete; the verified impact above comes from the correlated Red request response."
            : "Verifier evidence is unavailable for this round."}</p></article>
        </div>
        <div className="incident-sequence-heading"><span className="panel-kicker">Correlated HTTP and verification evidence · persisted order</span></div>
        <IncidentSequence events={selectedStepEvents} legacyRedFirst={legacyRedFirst} />
      </section>}

      {selectedBreach?.disclosure && <section className="panel server-selected-disclosure">
        <div className="panel-head"><div><span className="panel-kicker">Selected verified occurrence · episode #{num(selectedBreach.payload.episode_index) ?? "—"}</span><h2 className="panel-title">{disclosureGroupLabel(selectedBreach.disclosure)}</h2></div><span className="pill pill-red">{selectedBreach.disclosure.unauthorized_rows} unauthorized rows</span></div>
        <div className="server-selected-disclosure-grid">
          <div><span>Red policy</span><strong>{selectedBreach.disclosure.policy_id} · v{selectedBreach.disclosure.policy_version ?? "—"}</strong></div>
          <div><span>HTTP request</span><strong>POST /api/lookup · {selectedBreach.disclosure.field} = {selectedBreach.disclosure.query_value}</strong></div>
          <div><span>Request ID</span><strong>{selectedBreach.disclosure.request_id}</strong></div>
          <div><span>Target event</span><strong>{selectedBreach.disclosure.target_event_id} · HTTP {selectedBreach.disclosure.http_status ?? "—"}</strong></div>
        </div>
        <details className="server-payload-details"><summary>Returned target rows</summary><pre>{pretty(selectedBreach.disclosure.returned_rows)}</pre></details>
      </section>}

      <section className={`server-outcome-banner ${outcomePayload.status === "completed" ? "outcome-complete" : "outcome-partial"}`}>
        <div><span className="panel-kicker">Verified episode outcome</span>
          <strong>{str(outcomePayload.winner, outcomePayload.status === "incomplete" ? "Incomplete · no winner" : "Outcome not recorded")}</strong>
        </div>
        <div><span>Reason</span><b>{str(outcomePayload.reason, "Target or verifier evidence is pending")}</b></div>
        <div><span>Rounds</span><b>{num(outcomePayload.steps) ?? rounds.length} / {num(outcomePayload.max_steps) ?? num(run.config.max_steps) ?? 40}</b></div>
        <div><span>Red disclosure</span><b>{outcomePayload.verified_exposure === true || outcomePayload.challenge_disclosed === true ? "verified" : outcomePayload.status === "completed" ? "none verified" : "unknown / partial"}</b></div>
        <div><span>Blue prevention</span><b>{num(outcomePayload.verified_prevention_rate) === null ? "unknown" : `${Math.round((num(outcomePayload.verified_prevention_rate) ?? 0) * 100)}% verified`}</b></div>
        <div><span>Legitimate lookup</span><b>{outcomePayload.legitimate_query_served === true ? "verified available" : outcomePayload.legitimate_query_served === false ? "not available at finish" : "unknown"}</b></div>
      </section>

      <section className="panel server-learning">
        <div className="panel-head">
          <div><span className="panel-kicker">After-episode learning report</span><h2 className="panel-title">Recorded PPO update and policy shift</h2></div>
          <span className="pill pill-amber"><ShieldCheck size={12} />One reference observation</span>
        </div>
        <div className="panel-body">
          <p className="micro server-probe-explanation">Before/after probabilities use the same reference observation for each role. This probe describes that one state; it does not measure general performance.</p>
          <div className="server-learning-grid">
            {probes.map((event) => <ProbabilitySnapshot key={event.event_id} event={event} />)}
            {updates.map((event) => <UpdateReport key={event.event_id} event={event} />)}
          </div>
          {Boolean(outcomePayload.checkpoint_sha256) && <div className="server-checkpoint-line">
            Checkpoint digest <code>{str(outcomePayload.checkpoint_sha256)}</code>
          </div>}
          {!updates.length && <div className="server-missing server-learning-missing">PPO update evidence is missing or this episode has not reached its update boundary yet.</div>}
          <div className="next-blue-report" aria-label="Blue policy at the next episode">
            <div><span className="panel-kicker">Next episode handoff</span><strong>{nextBlueUse
              ? `Episode #${nextBlueUse.episodeIndex + 1} · Blue v${nextBlueUse.policyVersion ?? "unavailable"}`
              : "Next Blue version and action"}</strong></div>
            <p>{nextBlueUse
              ? nextBlueUse.action
                ? `The next episode started with Blue v${nextBlueUse.policyVersion ?? "unavailable"}; its first recorded action was ${nextBlueUse.action}.`
                : "The next episode exists, but its Blue decision is not available in persisted evidence."
              : "Unavailable until a following episode start and Blue decision are persisted. No policy handoff is inferred."}</p>
          </div>
        </div>
      </section>

      <section className="server-rounds">
        <div className="server-section-title">
          <div><span className="panel-kicker">Persisted event timeline</span><h2>Round-by-round replay</h2></div>
          <span className={`pill ${loadingTimeline ? "pill-amber" : "pill-green"}`}>
            {loadingTimeline ? "Loading cursor pages" : `${rounds.length} rounds · ${cursorExhausted ? "timeline exhausted" : "waiting"}`}
          </span>
        </div>
        {rounds.length ? rounds.map(([step, grouped]) => <Round key={step} step={step} events={grouped} />)
          : <section className="panel server-empty">{loadingTimeline ? "Loading persisted events…" : "No round events are recorded for this episode."}</section>}
        {!loadingTimeline && rounds.some(([, grouped]) => !grouped.some((event) => event.event_type === "sqli_verification"))
          && <div className="server-missing">At least one round has no verifier record. The episode must remain partial or incomplete.</div>}
      </section>
    </>}
    {!loadingRuns && runs.length === 0 && <section className="panel server-empty">No SQLi training runs have been recorded.</section>}
  </div>;
}
