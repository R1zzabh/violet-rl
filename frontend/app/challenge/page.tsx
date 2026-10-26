"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  Check,
  CircleHelp,
  FlaskConical,
  LockKeyhole,
  ShieldCheck,
  Swords,
} from "lucide-react";
import {
  rangeApi,
  type RangeEvent,
  type SqliCheckpoint,
  type SqliChallengeSession,
  type SqliTargetStatus,
} from "../../lib/range-api";

const SESSION_KEY = "violet-challenge-session";

function obj(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function str(value: unknown, fallback = "—"): string { return typeof value === "string" && value ? value : fallback; }
function num(value: unknown): number | null { return typeof value === "number" && Number.isFinite(value) ? value : null; }

function describe(event: RangeEvent): { title: string; detail: string; tone: string } {
  const payload = event.payload;
  if (event.event_type === "sqli_episode_started") {
    return { title: "Challenge episode started", detail: `seed ${str(payload.seed)} · ${num(payload.max_steps) ?? 40} round limit · frozen checkpoint`, tone: "amber" };
  }
  if (event.event_type === "sqli_red_decision") {
    const action = obj(payload.components);
    const choices = [action.field, action.base_term, action.close_quote ? "close quote" : "open quote", action.connector, action.predicate, action.comment ? "comment" : "no comment"]
      .filter((item) => typeof item === "string" && item !== "none");
    const probabilities = Array.isArray(payload.selected_component_probabilities)
      ? (payload.selected_component_probabilities as unknown[]).map((value) => `${Math.round(Number(value) * 100)}%`).join(" · ")
      : "probability unavailable";
    return { title: "Red PPO selected a request", detail: `${choices.map(String).join(" · ")} · p ${probabilities}`, tone: "red" };
  }
  if (event.event_type === "sqli_target_response") {
    const response = obj(payload.response);
    const telemetry = obj(payload.telemetry);
    const count = num(response.row_count) ?? num(telemetry.row_count) ?? 0;
    return {
      title: `Target HTTP ${str(payload.http_status)} · ${str(payload.purpose)}`,
      detail: `${count} rows · evidence ${str(telemetry.evidence_status)}${telemetry.challenge_disclosed ? " · phrase disclosure confirmed" : ""}`,
      tone: "green",
    };
  }
  if (event.event_type === "sqli_blue_decision") {
    const action = obj(payload.selected_action);
    const probabilities = Array.isArray(payload.selected_component_probabilities)
      ? (payload.selected_component_probabilities as unknown[]).map((value) => `${Math.round(Number(value) * 100)}%`).join(" · ")
      : "probability unavailable";
    return { title: "Blue PPO selected a control", detail: `${str(action.operation, str(payload.phase))} · p ${probabilities}`, tone: "blue" };
  }
  if (event.event_type === "sqli_control_acknowledged") {
    return { title: `Control applied · ${str(payload.action)}`, detail: `target acknowledgement ${str(obj(payload.acknowledgement).status)}`, tone: "blue" };
  }
  if (event.event_type === "sqli_verification") {
    return { title: "Independent target verification", detail: `prevention ${payload.verified_prevention ? "confirmed" : "not confirmed"} · legitimate lookup ${payload.legitimate_query_served ? "available" : "unavailable"}`, tone: "green" };
  }
  if (event.event_type === "sqli_reward") {
    const red = obj(payload.red);
    const blue = obj(payload.blue);
    return { title: "Evidence-linked rewards", detail: `Red ${num(red.total)?.toFixed(2) ?? "0.00"} · Blue ${num(blue.total)?.toFixed(2) ?? "0.00"}`, tone: "amber" };
  }
  if (event.event_type === "sqli_episode_outcome") {
    return { title: `Verified result · ${str(payload.winner, "incomplete")}`, detail: `${str(payload.reason, str(payload.status))} · ${num(payload.steps) ?? 0} / ${num(payload.max_steps) ?? 40} rounds`, tone: payload.winner === "red" ? "red" : payload.winner === "blue" ? "green" : "amber" };
  }
  return { title: event.event_type.replaceAll("_", " "), detail: "Committed to the redacted challenge timeline", tone: "" };
}

function ChallengeTimeline({ events }: { events: RangeEvent[] }) {
  if (!events.length) return <div className="empty-state"><FlaskConical size={18} /><span>Once the trainer claims the session, Red and Blue decisions will appear here.</span></div>;
  return <div className="timeline-scroll"><div className="timeline-list">
    {events.map((event) => {
      const item = describe(event);
      return <article className="timeline-item" key={`${event.event_id}-${event.seq}`}>
        <time className="timeline-time">{event.occurred_at.slice(11, 19)}<br />#{event.seq}</time>
        <div className="timeline-content">
          <div className="timeline-title"><span className={`timeline-dot ${item.tone}`} />{item.title}</div>
          <div className="timeline-detail">{item.detail}</div>
          {event.event_type === "sqli_target_response" && <div className="event-response">
            target event {str(event.payload.target_event_id)} · request {str(event.payload.request_id)} · phrase contents are omitted
          </div>}
        </div>
      </article>;
    })}
  </div></div>;
}

export default function ChallengePage() {
  const [phrase, setPhrase] = useState("");
  const [target, setTarget] = useState<SqliTargetStatus | null>(null);
  const [checkpoint, setCheckpoint] = useState<SqliCheckpoint | null>(null);
  const [challenge, setChallenge] = useState<SqliChallengeSession | null>(null);
  const [events, setEvents] = useState<RangeEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [streamState, setStreamState] = useState("waiting");

  const refresh = useCallback(async () => {
    try {
      const [targetStatus, latestCheckpoint] = await Promise.all([
        rangeApi.sqliTargetStatus(),
        rangeApi.sqliCheckpoint(),
      ]);
      setTarget(targetStatus);
      setCheckpoint(latestCheckpoint);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The local lab is unavailable.");
    }
  }, []);

  const restoreSession = useCallback(async (id: string) => {
    try {
      const current = await rangeApi.sqliChallenge(id);
      setChallenge(current);
    } catch {
      sessionStorage.removeItem(SESSION_KEY);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refresh();
      const stored = window.sessionStorage.getItem(SESSION_KEY);
      if (stored) void restoreSession(stored);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [refresh, restoreSession]);

  useEffect(() => {
    if (!challenge?.run_id) return;
    let stopped = false;
    let source: EventSource | null = null;
    const addEvent = (event: RangeEvent) => {
      setEvents((current) => [event, ...current.filter((item) => item.event_id !== event.event_id)].slice(0, 200));
      if (event.event_type === "sqli_episode_outcome") setStreamState("result persisted");
    };
    const connect = async () => {
      let cursor = 0;
      try {
        for (;;) {
          const page = await rangeApi.events(challenge.run_id, cursor);
          if (stopped) return;
          page.events.forEach(addEvent);
          if (!page.events.length || page.events.length < 500) break;
          const next = page.events.at(-1)?.seq ?? cursor;
          if (next <= cursor) break;
          cursor = next;
        }
      } catch {
        // SSE below retries from the committed cursor if a replay page is momentarily unavailable.
      }
      if (stopped) return;
      source = new EventSource(rangeApi.streamUrl(challenge.run_id, cursor));
      source.onopen = () => setStreamState("live");
      source.onerror = () => setStreamState("reconnecting");
      source.onmessage = (message) => {
        try { addEvent(JSON.parse(message.data) as RangeEvent); }
        catch { setStreamState("reconnecting"); }
      };
    };
    void connect();
    const timer = window.setInterval(async () => {
      try {
        const current = await rangeApi.sqliChallenge(challenge.challenge_id);
        if (!stopped) setChallenge(current);
      } catch {
        if (!stopped) setStreamState("reconnecting");
      }
    }, 2500);
    return () => {
      stopped = true;
      window.clearInterval(timer);
      source?.close();
    };
  }, [challenge?.challenge_id, challenge?.run_id]);

  const active = challenge && ["queued", "preparing", "running"].includes(challenge.status);
  const outcome = obj(challenge?.result);
  const winner = outcome.winner === "red" || outcome.winner === "blue" ? outcome.winner : null;
  const checkpointResult = obj(checkpoint?.result);
  const canStart = Boolean(checkpoint && target?.status === "available" && phrase.trim() && !active && !busy);
  const lastHttp = useMemo(() => events.find((event) => event.event_type === "sqli_target_response"), [events]);

  async function startChallenge(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canStart) return;
    setBusy(true);
    setError("");
    try {
      const created = await rangeApi.createSqliChallenge(phrase);
      setChallenge(created);
      setEvents([]);
      window.sessionStorage.setItem(SESSION_KEY, created.challenge_id);
      setPhrase("");
      setStreamState("connecting");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not start the challenge.");
    } finally { setBusy(false); }
  }

  const targetBadge = target?.status === "available" ? "pill-green" : "pill-amber";

  return <div>
    <header className="page-header">
      <div>
        <div className="eyebrow">Bounded challenge · fixed local executor</div>
        <h1 className="page-title">Can the trained pair find your phrase?</h1>
        <p className="page-deck">Submit one disposable test phrase. The latest completed Red + Blue checkpoint gets up to 40 rounds against the isolated SQLi target, then the target evidence decides the result.</p>
      </div>
      <Link href="/" className="button button-secondary"><ArrowLeft size={14} />Training monitor</Link>
    </header>

    {error && <div className="error-banner" role="alert"><AlertTriangle size={15} /><span>{error}</span></div>}

    <section className="challenge-layout">
      <div className="panel challenge-form">
        <div className="panel-head" style={{ padding: "0 0 16px" }}>
          <div><span className="panel-kicker">Session setup</span><h2 className="panel-title" style={{ marginTop: 3 }}>Enter a disposable phrase</h2></div>
          <span className={`pill ${targetBadge}`}><span className={`live-pip ${target?.status === "available" ? "" : "is-down"}`} />target {target?.status ?? "checking"}</span>
        </div>

        <form onSubmit={startChallenge} className="form-stack">
          <label className="form-label" htmlFor="secret-phrase">Test phrase
            <span className="form-hint">Use synthetic text you can safely discard. Up to 96 characters.</span>
          </label>
          <input
            id="secret-phrase"
            className="text-input"
            type="text"
            autoComplete="off"
            maxLength={96}
            value={phrase}
            onChange={(event) => setPhrase(event.target.value)}
            placeholder="e.g. classroom-canary-482"
            disabled={Boolean(active) || busy}
            required
          />
          <div className="form-meta"><span>Private in-memory target value</span><span>{phrase.length} / 96</span></div>
          <button className="button button-red" type="submit" disabled={!canStart}>
            {busy ? <><span className="spinner" />Preparing target</> : <><Swords size={14} />Run 40-round challenge</>}
          </button>
        </form>

        <div className="challenge-details">
          <div className="detail-line"><span>Checkpoint</span><strong>{checkpoint ? checkpoint.training_run_id.slice(0, 12) : "not ready"}</strong></div>
          <div className="detail-line"><span>Training episodes</span><strong>{Number(checkpointResult.completed_training_episodes ?? 0).toLocaleString()}</strong></div>
          <div className="detail-line"><span>Checkpoint digest</span><strong>{str(checkpointResult.checkpoint_sha256).slice(0, 12)}</strong></div>
          <div className="detail-line"><span>Policy versions</span><strong>{(() => { const versions = obj(checkpointResult.policy_versions); return `R${num(versions.red) ?? "—"} / B${num(versions.blue) ?? "—"}`; })()}</strong></div>
        </div>

        {!checkpoint && <div className="notice" style={{ marginTop: 16 }}>
          <strong>No completed demonstration checkpoint yet.</strong> Start the 10,000-episode campaign on the <Link href="/" style={{ color: "var(--blue)", fontWeight: 700 }}>training monitor</Link>; the challenge unlocks when its joint checkpoint completes.
        </div>}

        {challenge && <div className={`challenge-result ${winner === "red" ? "red" : winner === "blue" ? "blue" : challenge.status === "incomplete" || challenge.status === "failed" ? "incomplete" : ""}`}>
          <div className="panel-kicker">{active ? "Challenge running" : challenge.status === "completed" ? "Verified result" : challenge.status}</div>
          <h2 className="challenge-result-title">{active ? "Agents are acting on the target" : winner === "red" ? "Red disclosed the phrase" : winner === "blue" ? "Blue held the boundary" : "Outcome incomplete"}</h2>
          <p className="challenge-result-copy">
            {active ? "The phrase stays out of the persisted event stream. Follow each policy sample, HTTP response, Blue control, and verifier check below." : winner === "red" ? "Correlated target evidence confirms that the submitted phrase was returned in an unauthorized lookup." : winner === "blue" ? "Forty rounds completed without a verified disclosure, and the legitimate researcher lookup remained available." : "A target or verifier result was missing or unverified. This session is not counted as a win."}
          </p>
          {!active && <div className="event-pair">
            <div className="evidence-cell"><span className="evidence-label">Exposure verified</span><span className="evidence-value">{outcome.challenge_disclosed ? "yes · target confirmed" : "no"}</span></div>
            <div className="evidence-cell"><span className="evidence-label">Legitimate lookup</span><span className="evidence-value">{outcome.legitimate_query_served ? "available" : "not confirmed"}</span></div>
          </div>}
          {challenge.status === "completed" && <div className="micro text-muted" style={{ marginTop: 9 }}>No PPO updates are made during a challenge; the checkpoint remains frozen.</div>}
        </div>}
      </div>

      <div className="form-stack">
        <section className="panel">
          <div className="panel-head">
            <div><span className="panel-kicker">Red + Blue · up to 40 rounds</span><h2 className="panel-title">Policy and HTTP timeline</h2></div>
            <span className="pill"><span className={`live-pip ${streamState === "live" ? "" : active ? "is-busy" : ""}`} />{challenge ? streamState : "awaiting session"}</span>
          </div>
          <div className="panel-body"><ChallengeTimeline events={events} /></div>
        </section>

        <section className="panel">
          <div className="panel-head"><div><span className="panel-kicker">Evidence snapshot</span><h2 className="panel-title">Latest target response</h2></div><span className="pill pill-green"><Check size={11} />Target correlated</span></div>
          <div className="panel-body">
            {lastHttp ? <div className="form-stack" style={{ gap: 9 }}>
              <div className="detail-line"><span>HTTP status</span><strong>{str(lastHttp.payload.http_status)}</strong></div>
              <div className="detail-line"><span>Purpose</span><strong>{str(lastHttp.payload.purpose)}</strong></div>
              <div className="detail-line"><span>Target event ID</span><strong>{str(lastHttp.payload.target_event_id)}</strong></div>
              <div className="detail-line"><span>Evidence</span><strong>{str(obj(lastHttp.payload.telemetry).evidence_status)}</strong></div>
              <div className="event-response">Response rows and status are retained for replay; the phrase itself is omitted.</div>
            </div> : <div className="empty-state"><LockKeyhole size={18} /><span>Target evidence will appear with the first HTTP action.</span></div>}
          </div>
        </section>
      </div>
    </section>

    <section className="chart-grid" style={{ marginTop: 14 }}>
      <article className="panel"><div className="panel-head"><div><span className="panel-kicker">Outcome rules</span><h2 className="panel-title">Target evidence decides</h2></div><CircleHelp size={16} className="text-muted" /></div>
        <div className="panel-body form-stack" style={{ gap: 9 }}>
          <div className="panel-inset" style={{ padding: 11 }}><strong style={{ color: "var(--red)", fontSize: 10 }}>Red wins</strong><div className="tiny text-muted" style={{ marginTop: 3 }}>Target evidence confirms the exact session’s phrase was disclosed in an unauthorized response.</div></div>
          <div className="panel-inset" style={{ padding: 11 }}><strong style={{ color: "var(--blue)", fontSize: 10 }}>Blue wins</strong><div className="tiny text-muted" style={{ marginTop: 3 }}>All 40 rounds finish without disclosure and the owner lookup still works.</div></div>
          <div className="panel-inset" style={{ padding: 11 }}><strong className="text-muted" style={{ fontSize: 10 }}>Incomplete</strong><div className="tiny text-muted" style={{ marginTop: 3 }}>A target or verifier failure produces no win.</div></div>
        </div>
      </article>
      <article className="panel"><div className="panel-head"><div><span className="panel-kicker">Boundary</span><h2 className="panel-title">Local lab only</h2></div><ShieldCheck size={16} style={{ color: "var(--green)" }} /></div>
        <div className="panel-body form-stack" style={{ gap: 9 }}>
          <div className="detail-line"><span>Challenge target</span><strong>sqli-target · private network</strong></div>
          <div className="detail-line"><span>HTTP executor</span><strong>fixed destination and route</strong></div>
          <div className="detail-line"><span>Database</span><strong>synthetic · SELECT only</strong></div>
          <div className="detail-line"><span>Session phrase</span><strong>target memory · short-lived</strong></div>
          <div className="notice">The target, verifier, and Blue policy measure outcomes. Red and Blue actions come from the learned checkpoint.</div>
        </div>
      </article>
    </section>
  </div>;
}
