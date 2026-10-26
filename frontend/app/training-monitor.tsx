"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Bug,
  CircleStop,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import {
  rangeApi,
  type RangeEvent,
  type SqliTargetStatus,
  type TrainingRun,
} from "../lib/range-api";

type Numeric = number | null;
type Outcome = RangeEvent & { payload: Record<string, unknown> };
type ChartLine = { label: string; color: string; values: Numeric[] };

function object(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function text(value: unknown, fallback = "—"): string {
  return typeof value === "string" && value.length ? value : fallback;
}

function pct(value: Numeric): string {
  return value === null ? "—" : `${Math.round(value * 100)}%`;
}

function movingAverage(values: Numeric[], width = 100): Numeric[] {
  return values.map((_value, index) => {
    const window = values.slice(Math.max(0, index - width + 1), index + 1)
      .filter((item): item is number => item !== null);
    return window.length ? window.reduce((sum, item) => sum + item, 0) / window.length : null;
  });
}

function downsample(values: Numeric[], limit = 64): Numeric[] {
  if (values.length <= limit) return values;
  return Array.from({ length: limit }, (_item, index) => {
    const sourceIndex = Math.round(index * (values.length - 1) / (limit - 1));
    return values[sourceIndex] ?? null;
  });
}

function TrendChart({ lines, labels, fixedRange }: {
  lines: ChartLine[];
  labels?: [string, string];
  fixedRange?: [number, number];
}) {
  const available = lines.some((line) => line.values.some((value) => value !== null));
  if (!available) return <div className="chart-empty">Waiting for the first verified episode</div>;

  const sampled = lines.map((line) => ({ ...line, values: downsample(line.values) }));
  const all = sampled.flatMap((line) => line.values.filter((value): value is number => value !== null));
  const low = fixedRange?.[0] ?? Math.min(0, ...all);
  const high = fixedRange?.[1] ?? Math.max(1, ...all);
  const range = Math.max(1e-6, high - low);
  const width = 620;
  const height = 150;
  const left = 36;
  const right = 12;
  const top = 12;
  const bottom = 25;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const coords = (values: Numeric[]) => values
    .map((value, index) => value === null ? null : {
      x: left + (values.length <= 1 ? 0 : index * plotWidth / (values.length - 1)),
      y: top + (high - value) * plotHeight / range,
    });

  return (
    <div className="chart-wrap">
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={lines.map((line) => line.label).join(" and ")}>
        {[0, 1, 2, 3].map((tick) => {
          const y = top + tick * plotHeight / 3;
          const value = high - tick * range / 3;
          return <g key={tick}>
            <line className="chart-grid-line" x1={left} y1={y} x2={width - right} y2={y} />
            <text className="chart-axis-label" x={left - 7} y={y + 3} textAnchor="end">
              {fixedRange ? `${Math.round(value * 100)}%` : Number.isInteger(value) ? value : value.toFixed(1)}
            </text>
          </g>;
        })}
        {sampled.map((line) => {
          const points = coords(line.values);
          let current: string[] = [];
          const groups: string[][] = [];
          for (const point of points) {
            if (point) current.push(`${point.x},${point.y}`);
            else if (current.length) { groups.push(current); current = []; }
          }
          if (current.length) groups.push(current);
          return groups.map((group, index) => (
            <polyline key={`${line.label}-${index}`} className="chart-line" stroke={line.color} points={group.join(" ")} />
          ));
        })}
        <text className="chart-axis-label" x={left} y={height - 5}>{labels?.[0] ?? "start"}</text>
        <text className="chart-axis-label" x={width - right} y={height - 5} textAnchor="end">{labels?.[1] ?? "latest"}</text>
      </svg>
    </div>
  );
}

function Legend({ lines }: { lines: ChartLine[] }) {
  return <div className="chart-legend">{lines.map((line) => (
    <span className="legend-item" key={line.label}>
      <span className="legend-swatch" style={{ background: line.color }} />{line.label}
    </span>
  ))}</div>;
}

function updateOutcomes(events: RangeEvent[]): Outcome[] {
  return events.filter((event) => event.event_type === "sqli_episode_outcome") as Outcome[];
}

function eventKind(event: RangeEvent): { title: string; detail: string; role: string } {
  const payload = event.payload;
  if (event.event_type === "sqli_red_decision") {
    const action = object(payload.action);
    const components = object(payload.components);
    const parts = [components.field, components.base_term, components.connector, components.predicate]
      .filter((item) => typeof item === "string" && item !== "none")
      .map(String).join(" · ");
    return { title: "Red policy selected a request", detail: parts || JSON.stringify(action), role: "red" };
  }
  if (event.event_type === "sqli_target_response") {
    const response = object(payload.response);
    const telemetry = object(payload.telemetry);
    const count = number(response.row_count) ?? number(telemetry.row_count);
    return {
      title: `HTTP ${text(payload.http_status, "—")} · ${text(payload.purpose, "lookup")}`,
      detail: `${count ?? 0} rows · ${text(telemetry.evidence_status, "evidence pending")}${telemetry.challenge_disclosed ? " · phrase exposure verified" : ""}`,
      role: "green",
    };
  }
  if (event.event_type === "sqli_blue_decision") {
    const action = object(payload.selected_action);
    return { title: "Blue policy selected a control", detail: text(action.operation, text(payload.action, text(payload.phase))), role: "blue" };
  }
  if (event.event_type === "sqli_control_acknowledged") {
    const ack = object(payload.acknowledgement);
    return { title: `Blue control · ${text(payload.action)}`, detail: `target acknowledgement: ${text(ack.status)}`, role: "blue" };
  }
  if (event.event_type === "sqli_verification") {
    return {
      title: "Verifier checked target outcome",
      detail: `prevention ${payload.verified_prevention ? "confirmed" : "not confirmed"} · legitimate lookup ${payload.legitimate_query_served ? "available" : "unavailable"}`,
      role: "green",
    };
  }
  if (event.event_type === "sqli_ppo_update") {
    const metrics = object(payload.metrics);
    return {
      title: `${text(payload.role)} PPO update · v${text(payload.policy_version, "0")}`,
      detail: `${text(payload.update_status)} · loss ${number(metrics.loss)?.toFixed(3) ?? "—"} · ${number(metrics.sample_count) ?? 0} accepted transitions`,
      role: payload.role === "red" ? "red" : "blue",
    };
  }
  if (event.event_type === "sqli_episode_outcome") {
    return {
      title: `${text(payload.phase, "episode")} · ${text(payload.winner, "no winner")}`,
      detail: `${text(payload.reason, text(payload.status))} · ${number(payload.steps) ?? 0} rounds`,
      role: payload.winner === "red" ? "red" : payload.winner === "blue" ? "blue" : "amber",
    };
  }
  if (event.event_type === "sqli_episode_started") {
    return { title: "Episode started", detail: `seed ${text(payload.seed)} · ${number(payload.max_steps) ?? 0} round limit`, role: "amber" };
  }
  return { title: event.event_type.replaceAll("_", " "), detail: "Persisted range event", role: "" };
}

function EventTimeline({ events, empty = "The first episode will appear here." }: { events: RangeEvent[]; empty?: string }) {
  if (!events.length) return <div className="empty-state"><Activity size={18} /><span>{empty}</span></div>;
  return <div className="timeline-scroll"><div className="timeline-list">
    {events.map((event) => {
      const display = eventKind(event);
      return <article className="timeline-item" key={`${event.event_id}-${event.seq}`}>
        <time className="timeline-time">{event.occurred_at.slice(11, 19)}<br />#{event.seq}</time>
        <div className="timeline-content">
          <div className="timeline-title"><span className={`timeline-dot ${display.role}`} />{display.title}</div>
          <div className="timeline-detail">{display.detail}</div>
          {event.event_type === "sqli_target_response" && <div className="event-response">
            request {text(event.payload.request_id, "—")} · target event {text(event.payload.target_event_id, "—")}
          </div>}
        </div>
      </article>;
    })}
  </div></div>;
}

export default function TrainingMonitor() {
  const [target, setTarget] = useState<SqliTargetStatus | null>(null);
  const [run, setRun] = useState<TrainingRun | null>(null);
  const [outcomes, setOutcomes] = useState<Outcome[]>([]);
  const [updates, setUpdates] = useState<RangeEvent[]>([]);
  const [timeline, setTimeline] = useState<RangeEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [streamState, setStreamState] = useState<"connecting" | "live" | "reconnecting">("connecting");

  const discoverCampaign = useCallback(async (restart = false) => {
    setBusy(true);
    setError("");
    try {
      const status = await rangeApi.sqliTargetStatus();
      setTarget(status);
      const campaign = await rangeApi.demoCampaign(restart);
      setRun(campaign);
      setOutcomes([]);
      setUpdates([]);
      setTimeline([]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The local lab could not start.");
    } finally {
      setBusy(false);
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void discoverCampaign(), 0);
    return () => window.clearTimeout(timer);
  }, [discoverCampaign]);

  useEffect(() => {
    if (!run?.training_run_id) return;
    let stopped = false;
    let source: EventSource | null = null;

    const addEvent = (event: RangeEvent) => {
      if (event.event_type === "sqli_episode_outcome") {
        setOutcomes((current) => {
          const next = current.filter((item) => item.event_id !== event.event_id);
          return [...next, event as Outcome].sort((a, b) => a.seq - b.seq);
        });
      } else if (event.event_type === "sqli_ppo_update") {
        setUpdates((current) => [...current.filter((item) => item.event_id !== event.event_id), event].slice(-1000));
      }
      setTimeline((current) => [event, ...current.filter((item) => item.event_id !== event.event_id)].slice(0, 24));
    };

    const connect = async () => {
      let cursor = 0;
      const summaryRows: RangeEvent[] = [];
      try {
        for (;;) {
          const page = await rangeApi.sqliSummary(run.training_run_id, cursor, 1000);
          if (stopped) return;
          summaryRows.push(...page.items);
          if (!page.items.length || page.items.length < 1000) break;
          if (page.next_after_seq <= cursor) break;
          cursor = page.next_after_seq;
        }
        if (stopped) return;
        setOutcomes(updateOutcomes(summaryRows).sort((a, b) => a.seq - b.seq));
        setUpdates(summaryRows.filter((event) => event.event_type === "sqli_ppo_update").slice(-1000));
        cursor = summaryRows.reduce((last, event) => Math.max(last, event.seq), cursor);
        const latestOutcome = updateOutcomes(summaryRows).sort((a, b) => b.seq - a.seq)[0];
        if (latestOutcome?.episode_id) {
          const timelinePage = await rangeApi.episodeTimeline(run.training_run_id, latestOutcome.episode_id);
          if (!stopped) setTimeline(timelinePage.items.slice().reverse().slice(0, 24));
        }
      } catch {
        // The stream still supplies new committed events if an older summary page is unavailable.
      }
      if (stopped) return;
      source = new EventSource(rangeApi.streamUrl(run.training_run_id, cursor));
      source.onopen = () => setStreamState("live");
      source.onerror = () => setStreamState("reconnecting");
      source.onmessage = (message) => {
        try {
          addEvent(JSON.parse(message.data) as RangeEvent);
        } catch {
          setStreamState("reconnecting");
        }
      };
    };

    void connect();
    const poll = window.setInterval(async () => {
      try {
        const [latest, targetStatus] = await Promise.all([
          rangeApi.trainingRun(run.training_run_id),
          rangeApi.sqliTargetStatus(),
        ]);
        if (!stopped) { setRun(latest); setTarget(targetStatus); }
      } catch {
        if (!stopped) setStreamState("reconnecting");
      }
    }, 5000);
    return () => {
      stopped = true;
      window.clearInterval(poll);
      source?.close();
    };
  }, [run?.training_run_id]);

  const trainingOutcomes = useMemo(() => outcomes.filter((item) => item.payload.phase === "training"), [outcomes]);
  const heldOutcomes = useMemo(() => outcomes.filter((item) => item.payload.phase === "held_out_evaluation"), [outcomes]);
  const completed = trainingOutcomes.filter((item) => item.payload.status === "completed");
  const redWins = completed.filter((item) => item.payload.winner === "red").length;
  const blueWins = completed.filter((item) => item.payload.winner === "blue").length;
  const latestOutcome = trainingOutcomes.at(-1)?.payload;
  const policyVersions = object(latestOutcome?.policy_versions ?? run?.result?.policy_versions);
  const requested = number(run?.config.episodes) ?? 10000;
  const progress = Math.min(100, completed.length * 100 / requested);
  const runActive = run?.status === "queued" || run?.status === "running";
  const targetAvailable = target?.status === "available";
  const evaluationResult = object(run?.result?.held_out_evaluation);
  const evalCompleted = number(evaluationResult.completed_matches) ?? heldOutcomes.filter((item) => item.payload.status === "completed").length;
  const heldOutTotal = number(evaluationResult.requested_matches) ?? 500;
  const redWinRate = completed.length ? redWins / completed.length : null;
  const blueWinRate = completed.length ? blueWins / completed.length : null;

  const winLines: ChartLine[] = [
    { label: "Red wins", color: "var(--red)", values: movingAverage(completed.map((item) => item.payload.winner === "red" ? 1 : 0)) },
    { label: "Blue wins", color: "var(--blue)", values: movingAverage(completed.map((item) => item.payload.winner === "blue" ? 1 : 0)) },
  ];
  const securityLines: ChartLine[] = [
    { label: "Verified exposure", color: "var(--red)", values: movingAverage(completed.map((item) => item.payload.verified_exposure ? 1 : 0)) },
    { label: "Verified prevention", color: "var(--blue)", values: movingAverage(completed.map((item) => number(item.payload.verified_prevention_rate))) },
  ];
  const availabilityLines: ChartLine[] = [
    { label: "Legitimate lookup available", color: "var(--green)", values: movingAverage(completed.map((item) => number(item.payload.legitimate_query_availability))) },
  ];
  const returnLines: ChartLine[] = [
    { label: "Red return", color: "var(--red)", values: movingAverage(completed.map((item) => number(item.payload.red_return))) },
    { label: "Blue return", color: "var(--blue)", values: movingAverage(completed.map((item) => number(item.payload.blue_return))) },
  ];
  const policyLines: ChartLine[] = [
    { label: "Red policy", color: "var(--red)", values: trainingOutcomes.map((item) => number(object(item.payload.policy_versions).red)) },
    { label: "Blue policy", color: "var(--blue)", values: trainingOutcomes.map((item) => number(object(item.payload.policy_versions).blue)) },
  ];

  async function stopCampaign() {
    if (!run) return;
    setBusy(true);
    try { setRun(await rangeApi.stopTrainingRun(run.training_run_id)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not request a stop."); }
    finally { setBusy(false); }
  }

  const statusLabel = run?.status === "completed" ? "Campaign complete"
    : run?.status === "failed" ? "Campaign interrupted"
      : run?.status === "stopped" ? "Campaign stopped"
        : run?.status === "queued" ? "Queued for trainer"
          : run?.status === "running" ? "Training in progress" : loading ? "Connecting to local lab" : "Waiting for a campaign";

  return <div>
    <header className="page-header">
      <div>
        <div className="eyebrow">Live PPO campaign · evidence-first classroom demo</div>
        <h1 className="page-title">Two policies. One HTTP target.</h1>
        <p className="page-deck">Red and Blue learn from the same isolated SQLi lab. Watch each policy decision, target response, independent verification, reward, and PPO update as it happens.</p>
      </div>
      <div className="header-actions">
        {runActive && <button className="button button-secondary" onClick={stopCampaign} disabled={busy}><CircleStop size={14} />Stop campaign</button>}
        {(run?.status === "failed" || run?.status === "stopped") && <button className="button" onClick={() => void discoverCampaign(true)} disabled={busy}><RefreshCw size={14} />Start a fresh campaign</button>}
        {!run && <button className="button" onClick={() => void discoverCampaign()} disabled={busy}><RefreshCw size={14} />Connect to lab</button>}
      </div>
    </header>

    {error && <div className="error-banner" role="alert"><AlertTriangle size={15} />
      <div><strong>{error}</strong><div className="micro" style={{ marginTop: 3 }}>Training requires the private target and its fixed executor to report healthy.</div></div>
    </div>}

    <section className="campaign-grid" aria-label="Campaign status">
      <article className="panel campaign-hero">
        <div className="campaign-topline">
          <span className="pill pill-amber">Single seed · demonstration only</span>
          <span className={`pill ${runActive ? "pill-green" : run?.status === "completed" ? "pill-blue" : ""}`}>
            <span className={`live-pip ${!targetAvailable ? "is-down" : runActive ? "is-busy" : ""}`} />{statusLabel}
          </span>
        </div>
        <h2 className="campaign-name">Joint Red + Blue PPO training</h2>
        <p className="campaign-desc">Up to 40 real request / defense rounds per episode. A joint model and both optimizer states are checkpointed after every episode for recovery.</p>
        <div className="progress-row">
          <span className="progress-count">{completed.length.toLocaleString()}</span>
          <span className="progress-total">of {requested.toLocaleString()} completed PPO-updated episodes</span>
        </div>
        <div className="progress-track" role="progressbar" aria-label="Training campaign progress" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100}>
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <div className="progress-note"><span>{progress.toFixed(1)}% complete</span><span>{text(run?.config.seed, "seed pending")}</span></div>
      </article>
      <aside className="panel campaign-meta" aria-label="Target and run details">
        <div className="meta-cell"><span className="meta-label">HTTP target</span><span className="meta-value" style={{ color: targetAvailable ? "var(--green)" : "var(--red)" }}>{target?.status ?? "checking"}</span></div>
        <div className="meta-cell"><span className="meta-label">Target version</span><span className="meta-value">{text(target?.target_version, "sqli_target_v1")}</span></div>
        <div className="meta-cell"><span className="meta-label">Checkpoint SHA-256</span><span className="meta-value">{text(run?.result?.checkpoint_sha256, runActive ? "updated each episode" : "not yet available")}</span></div>
        <div className="meta-cell"><span className="meta-label">Event stream</span><span className="meta-value" style={{ color: streamState === "live" ? "var(--green)" : "var(--amber)" }}>{streamState}</span></div>
      </aside>
    </section>

    <section className="stats-grid" aria-label="Measured training outcomes">
      <article className="panel stat-card"><span className="stat-label">Completed episodes</span><strong className="stat-value">{completed.length.toLocaleString()}</strong><span className="stat-foot">Each includes paired policy updates</span></article>
      <article className="panel stat-card stat-red"><span className="stat-label">Red wins</span><strong className="stat-value">{pct(redWinRate)}</strong><span className="stat-foot">Target-confirmed unauthorized exposure</span></article>
      <article className="panel stat-card stat-blue"><span className="stat-label">Blue wins</span><strong className="stat-value">{pct(blueWinRate)}</strong><span className="stat-foot">40 rounds, no exposure, lookup available</span></article>
      <article className="panel stat-card"><span className="stat-label">Verified exposures</span><strong className="stat-value">{completed.filter((item) => item.payload.verified_exposure).length.toLocaleString()}</strong><span className="stat-foot">Joined to target evidence by request ID</span></article>
    </section>

    <section className="chart-grid" aria-label="Training outcome charts">
      <article className="panel"><div className="panel-head"><div><span className="panel-kicker">Measured outcomes</span><h2 className="panel-title">Win rate · rolling 100</h2></div><span className="pill">Target evidence</span></div><TrendChart lines={winLines} fixedRange={[0, 1]} /><Legend lines={winLines} /></article>
      <article className="panel"><div className="panel-head"><div><span className="panel-kicker">Exposure and control</span><h2 className="panel-title">Verified exposure / prevention</h2></div><span className="pill">Per episode</span></div><TrendChart lines={securityLines} fixedRange={[0, 1]} /><Legend lines={securityLines} /></article>
      <article className="panel"><div className="panel-head"><div><span className="panel-kicker">Service continuity</span><h2 className="panel-title">Legitimate lookup availability</h2></div><span className="pill pill-green">Researcher probe</span></div><TrendChart lines={availabilityLines} fixedRange={[0, 1]} /><Legend lines={availabilityLines} /></article>
      <article className="panel"><div className="panel-head"><div><span className="panel-kicker">Agent learning</span><h2 className="panel-title">Policy version by episode</h2></div><span className="pill">Updated after episode</span></div><TrendChart lines={policyLines} labels={["episode 1", `episode ${trainingOutcomes.length.toLocaleString() || "—"}`]} /><Legend lines={policyLines} /></article>
      <article className="panel"><div className="panel-head"><div><span className="panel-kicker">Reward signal</span><h2 className="panel-title">Red / Blue return · rolling 100</h2></div><span className="pill">PPO reward</span></div><TrendChart lines={returnLines} /><Legend lines={returnLines} /></article>
      <article className="panel"><div className="panel-head"><div><span className="panel-kicker">Held-out matches · separate from training</span><h2 className="panel-title">Frozen-pair evaluation</h2></div><span className="pill pill-amber">500 seeded matches</span></div>
        <div className="panel-body">
          <div className="progress-row" style={{ marginTop: 6 }}><span className="progress-count">{evalCompleted.toLocaleString()}</span><span className="progress-total">of {heldOutTotal.toLocaleString()} complete</span></div>
          <div className="progress-track"><div className="progress-fill" style={{ width: `${Math.min(100, evalCompleted * 100 / heldOutTotal)}%`, background: "var(--green)" }} /></div>
          <div className="evaluation-box">
            <div className="panel-kicker">Measured on new target seeds · no scripted opponents</div>
            <div className="evaluation-grid">
              <div className="evaluation-stat"><span className="micro text-muted">Red win rate</span><strong>{pct(number(evaluationResult.red_win_rate))}</strong></div>
              <div className="evaluation-stat"><span className="micro text-muted">Blue win rate</span><strong>{pct(number(evaluationResult.blue_win_rate))}</strong></div>
              <div className="evaluation-stat"><span className="micro text-muted">Exposure</span><strong>{pct(number(evaluationResult.verified_exposure_rate))}</strong></div>
              <div className="evaluation-stat"><span className="micro text-muted">Availability</span><strong>{pct(number(evaluationResult.legitimate_service_availability))}</strong></div>
            </div>
          </div>
          <p className="micro text-muted" style={{ margin: "10px 0 0" }}>Single-seed classroom demonstration. These results are not a multi-seed research claim.</p>
        </div>
      </article>
    </section>

    <section className="lower-grid">
      <article className="panel">
        <div className="panel-head"><div><span className="panel-kicker">Separate actors · shared episode boundary</span><h2 className="panel-title">Current PPO policies</h2><span className="micro text-muted">Most recent {updates.length.toLocaleString()} PPO update records loaded</span></div><Link href="/challenge" className="button button-secondary button-small">Open challenge <ArrowRight size={12} /></Link></div>
        <div className="panel-body">
          <div className="policy-pair">
            <div className="policy-card red"><div className="policy-title"><span><Bug size={13} style={{ verticalAlign: "-2px", marginRight: 6 }} />Red · request policy</span><span className="pill pill-red">PPO</span></div><div className="policy-version">v{number(policyVersions.red) ?? 0}</div><div className="policy-note">bounded SQL grammar · selected probability logged</div></div>
            <div className="policy-card blue"><div className="policy-title"><span><ShieldCheck size={13} style={{ verticalAlign: "-2px", marginRight: 6 }} />Blue · control policy</span><span className="pill pill-blue">PPO</span></div><div className="policy-version">v{number(policyVersions.blue) ?? 0}</div><div className="policy-note">sanitized telemetry · typed target controls</div></div>
          </div>
          <div className="notice" style={{ marginTop: 10 }}><strong>Evidence standard.</strong> Verifier probes measure prevention and researcher access. They never choose either agent&apos;s action.</div>
        </div>
      </article>

      <article className="panel">
        <div className="panel-head"><div><span className="panel-kicker">Persisted sequence · reconnect-safe</span><h2 className="panel-title">Latest policy / HTTP timeline</h2></div><span className="pill"><span className={`live-pip ${streamState === "live" ? "" : "is-busy"}`} />{streamState}</span></div>
        <div className="panel-body"><EventTimeline events={timeline} empty="Waiting for policy decisions and target evidence." /></div>
      </article>
    </section>

    <aside className="notice">
      <strong>Runtime boundary.</strong> The executor has one fixed SQLi target and no database or public egress. The target runs on its private range network, the synthetic database is separate, and this UI is bound to localhost.
    </aside>
  </div>;
}
