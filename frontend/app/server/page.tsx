"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  CirclePause,
  CirclePlay,
  Database,
  Filter,
  Radio,
  RefreshCw,
  Server,
  ShieldCheck,
  SkipForward,
} from "lucide-react";
import {
  rangeApi,
  type RangeEvent,
  type ServerTrace,
  type ServerTraceStage,
  type SqliEpisodeIndexItem,
  type SqliTargetStatus,
} from "../../lib/range-api";

type ViewMode = "live" | "recorded";
type ActivityFilter = "all" | "agent" | "legitimate" | "controls";

const object = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const text = (value: unknown, fallback = "—") => typeof value === "string" && value ? value : fallback;
const number = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;
const safeJson = (value: unknown) => JSON.stringify(value ?? {}, null, 2);
const stamp = (value: unknown) => {
  if (typeof value !== "string") return "time unavailable";
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? "time unavailable" : parsed.toISOString().slice(11, 23) + " UTC";
};

function redact(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redact);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([key, item]) => [
      key,
      /phrase|secret|token/i.test(key) ? "[redacted]" : redact(item),
    ]));
  }
  return value;
}

function isRequest(event: RangeEvent) {
  return event.event_type === "sqli_target_response" && event.payload.phase !== "request_submitted";
}

function isControl(event: RangeEvent) {
  return event.event_type === "sqli_blue_decision" || event.event_type === "sqli_control_acknowledged";
}

function matchesFilter(event: RangeEvent, filter: ActivityFilter) {
  if (filter === "all") return isRequest(event) || isControl(event);
  if (filter === "controls") return isControl(event);
  if (!isRequest(event)) return false;
  if (filter === "agent") return event.payload.purpose === "policy";
  if (filter === "legitimate") return event.payload.purpose === "verifier_legitimate";
  return false;
}

async function readEpisode(runId: string, episodeId: string) {
  const events: RangeEvent[] = [];
  let cursor = 0;
  for (let pageIndex = 0; pageIndex < 5; pageIndex += 1) {
    const page = await rangeApi.episodeTimeline(runId, episodeId, cursor, 500);
    events.push(...page.items);
    if (!page.has_more || !page.items.length || page.next_after_seq <= cursor) break;
    cursor = page.next_after_seq;
  }
  return events;
}

function statusTone(status: string) {
  if (["completed", "passed", "entered", "observed"].includes(status)) return "trace-tone-good";
  if (["blocked", "rejected", "server_error", "not_reached"].includes(status)) return "trace-tone-warn";
  return "";
}

function stageName(stage: ServerTraceStage) {
  const names: Record<string, string> = {
    request_received: "HTTP request received",
    application_route: "Application route",
    request_validation: "Request validation",
    authentication_and_access: "Authentication & access checks",
    blue_controls: "Blue server controls",
    database_operation: "Database operation",
    http_response: "HTTP response",
  };
  return names[stage.stage_id] ?? stage.stage_id.replaceAll("_", " ");
}

function DetailJson({ label, value }: { label: string; value: unknown }) {
  return <div className="server-live-data">
    <span>{label}</span>
    <pre>{safeJson(redact(value))}</pre>
  </div>;
}

function TraceDetails({ trace, loading, error, event, decision }: {
  trace: ServerTrace | null;
  loading: boolean;
  error: string;
  event: RangeEvent;
  decision: RangeEvent | null;
}) {
  const payload = event.payload;
  const policy = decision?.payload ?? {};
  const telemetry = object(payload.telemetry);
  const response = object(payload.response);
  const rows = Array.isArray(response.rows) ? response.rows : [];
  const targetEventId = text(payload.target_event_id, "missing");
  const verified = telemetry.evidence_status === "verified" && targetEventId !== "missing";
  const policyVersion = typeof policy.policy_version === "number"
    ? String(policy.policy_version)
    : text(policy.policy_version, "unknown");

  return <section className="server-live-inspector">
    <div className="server-live-inspector-head">
      <div>
        <span className="panel-kicker">Request inspection</span>
        <h2>{text(payload.method, "POST")} /api/lookup</h2>
      </div>
      <span className={`pill ${verified ? "pill-green" : "pill-amber"}`}>
        {verified ? "Target evidence verified" : "Trace evidence incomplete"}
      </span>
    </div>
    <div className="server-live-id-row">
      <span>request <code>{text(payload.request_id, "missing")}</code></span>
      <span>decision <code>{text(payload.decision_id, "missing")}</code></span>
      {typeof policy.policy_id === "string" && <span>policy <code>{policy.policy_id} · v{policyVersion}</code></span>}
      <span>target event <code>{targetEventId}</code></span>
      <span>HTTP <code>{text(payload.http_status, "unknown")}</code></span>
    </div>

    {!verified && <div className="server-live-warning"><AlertTriangle size={15} />
      <span>{targetEventId === "missing" ? "The executor has no target event ID for this request." : `The target event could not be correlated (${text(telemetry.evidence_status)}).`} No server stages are presented as observed.</span>
    </div>}
    {verified && loading && <div className="server-live-loading">Loading target-recorded stages…</div>}
    {verified && error && <div className="server-live-warning"><AlertTriangle size={15} /><span>{error}</span></div>}
    {verified && trace && <>
      <div className="server-live-pipeline" aria-label="Observed server execution stages">
        {trace.trace.map((stage, index) => <div className="server-live-stage-wrap" key={`${stage.stage_id}-${index}`}>
          <button className={`server-live-stage ${statusTone(stage.status)}`} onClick={() => {
            document.getElementById(`trace-stage-${index}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
          }}>
            <span className="server-live-stage-index">{String(index + 1).padStart(2, "0")}</span>
            <strong>{stageName(stage)}</strong>
            <span>{stage.status.replaceAll("_", " ")}</span>
          </button>
          {index < trace.trace.length - 1 && <ChevronRight className="server-live-stage-arrow" size={15} aria-hidden="true" />}
        </div>)}
      </div>
      <div className="server-live-stage-details">
        {trace.trace.map((stage, index) => <details className="server-live-stage-detail" id={`trace-stage-${index}`} key={`${stage.stage_id}-detail-${index}`} open={index === trace.trace.length - 1}>
          <summary>
            <span className={`trace-stage-mark ${statusTone(stage.status)}`} />
            <strong>{stageName(stage)}</strong>
            <span>{stamp(stage.occurred_at)}</span>
            <span>{number(stage.duration_ms) === null ? "—" : `${number(stage.duration_ms)?.toFixed(2)} ms`}</span>
            <ChevronDown size={14} aria-hidden="true" />
          </summary>
          <div className="server-live-detail-grid">
            <DetailJson label="Observed inputs" value={stage.inputs} />
            <DetailJson label="Recorded result" value={stage.result} />
          </div>
        </details>)}
      </div>
      <div className="server-live-response-grid">
        <DetailJson label="Request received by target" value={payload.request} />
        <DetailJson label="Target response" value={{ http_status: payload.http_status, response, returned_rows: rows.length, telemetry: { ...telemetry, trace: undefined } }} />
      </div>
      <p className="server-live-evidence-note">An HTTP status describes the server response. A verified outcome requires the correlated target evidence record and the independent verifier.</p>
    </>}
  </section>;
}

export default function ServerLivePage() {
  const [episodes, setEpisodes] = useState<SqliEpisodeIndexItem[]>([]);
  const [episodeId, setEpisodeId] = useState("");
  const [events, setEvents] = useState<RangeEvent[]>([]);
  const [target, setTarget] = useState<SqliTargetStatus | null>(null);
  const [mode, setMode] = useState<ViewMode>("live");
  const [filter, setFilter] = useState<ActivityFilter>("all");
  const [paused, setPaused] = useState(false);
  const [cursor, setCursor] = useState(0);
  const [selectedKey, setSelectedKey] = useState("");
  const [trace, setTrace] = useState<ServerTrace | null>(null);
  const [traceError, setTraceError] = useState<{ targetEventId: string; message: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const requestInFlight = useRef(false);

  const selectedEpisode = episodes.find((item) => item.episode_id === episodeId) ?? null;
  const filteredEvents = useMemo(
    () => events.filter((event) => matchesFilter(event, filter)),
    [events, filter],
  );
  const following = mode === "live" && !paused;
  const visibleEvents = following ? filteredEvents : filteredEvents.slice(0, cursor);
  const activeEvent = visibleEvents.find((event) => `${event.event_id}:${event.seq}` === selectedKey)
    ?? [...visibleEvents].reverse().find(isRequest)
    ?? null;
  const activeDecisionId = activeEvent?.payload.decision_id;
  const policyDecision = typeof activeDecisionId === "string"
    ? events.find((candidate) =>
      (candidate.event_type === "sqli_red_decision" || candidate.event_type === "sqli_blue_decision")
      && candidate.payload.decision_id === activeDecisionId
      && typeof candidate.payload.policy_id === "string") ?? null
    : null;
  const controlCount = events.filter(isControl).length;
  const requestCount = events.filter(isRequest).length;
  const missingEvidenceCount = events.filter((event) => isRequest(event)
    && object(event.payload.telemetry).evidence_status !== "verified").length;

  const refresh = useCallback(async (first = false) => {
    if (requestInFlight.current) return;
    requestInFlight.current = true;
    try {
      const [page, targetState] = await Promise.all([
        rangeApi.sqliEpisodes({ limit: 80 }),
        rangeApi.sqliTargetStatus(),
      ]);
      setEpisodes(page.items);
      setTarget(targetState);
      setError("");
      const preferredId = episodeId || page.items[0]?.episode_id || "";
      if (preferredId && page.items.some((item) => item.episode_id === preferredId)) {
        const current = page.items.find((item) => item.episode_id === preferredId)!;
        const timeline = await readEpisode(current.run_id, current.episode_id);
        setEvents(timeline);
        if (!episodeId) setEpisodeId(preferredId);
        if (first && timeline.length) {
          const lastRequest = [...timeline].reverse().find(isRequest);
          if (lastRequest) setSelectedKey(`${lastRequest.event_id}:${lastRequest.seq}`);
        }
        if (mode === "live" && !paused) setCursor(timeline.length);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The recorded server timeline could not be loaded.");
    } finally {
      requestInFlight.current = false;
      setLoading(false);
    }
  }, [episodeId, mode, paused]);

  const initialLoadStarted = useRef(false);
  useEffect(() => {
    if (initialLoadStarted.current) return;
    initialLoadStarted.current = true;
    const id = window.setTimeout(() => { void refresh(true); }, 0);
    return () => window.clearTimeout(id);
  }, [refresh]);

  useEffect(() => {
    if (mode !== "live") return;
    const id = window.setInterval(() => void refresh(), 2500);
    return () => window.clearInterval(id);
  }, [mode, refresh]);

  useEffect(() => {
    if (mode !== "recorded" || paused || !visibleEvents.length) return;
    timer.current = setInterval(() => setCursor((current) => {
      if (current >= filteredEvents.length) {
        setPaused(true);
        return filteredEvents.length;
      }
      return current + 1;
    }), 520);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [mode, paused, filteredEvents.length, visibleEvents.length]);

  useEffect(() => {
    if (!activeEvent || !isRequest(activeEvent)) return;
    const targetEventId = activeEvent.payload.target_event_id;
    if (typeof targetEventId !== "string" || object(activeEvent.payload.telemetry).evidence_status !== "verified") return;
    let cancelled = false;
    rangeApi.serverTrace(targetEventId).then((result) => {
      if (!cancelled) setTrace(result);
    }).catch((reason) => {
      if (!cancelled) {
        setTraceError({
          targetEventId,
          message: reason instanceof Error ? reason.message : "Target trace is unavailable.",
        });
      }
    });
    return () => { cancelled = true; };
  }, [activeEvent]);

  const selectEpisode = (value: string) => {
    setEpisodeId(value);
    setEvents([]);
    setTrace(null);
    setSelectedKey("");
    setCursor(0);
    const selected = episodes.find((item) => item.episode_id === value);
    if (selected) void readEpisode(selected.run_id, selected.episode_id).then((rows) => {
      setEvents(rows);
      const last = [...rows].reverse().find(isRequest);
      if (last) setSelectedKey(`${last.event_id}:${last.seq}`);
      if (mode === "live" && !paused) setCursor(rows.length);
    }).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Timeline unavailable."));
  };

  const setViewMode = (next: ViewMode) => {
    setMode(next);
    if (next === "recorded") {
      setPaused(true);
      setCursor(1);
    } else {
      setPaused(false);
      setCursor(filteredEvents.length);
    }
  };

  const step = () => {
    setPaused(true);
    setCursor((current) => Math.min(filteredEvents.length, Math.max(1, current + 1)));
  };

  const selectEvent = (event: RangeEvent) => setSelectedKey(`${event.event_id}:${event.seq}`);

  return <div className="server-live-page">
    <header className="page-header">
      <div>
        <p className="eyebrow">Local target observatory / request trace v1</p>
        <h1 className="page-title">Server Live</h1>
        <p className="page-deck">Follow recorded requests through the target as they actually execute. Replay reads persisted events and never sends requests to the application.</p>
      </div>
      <div className="server-live-header-mark"><span className={target?.status === "available" ? "live-pip" : "live-pip is-down"} />
        {target?.status === "available" ? "target connected" : "target unavailable"}</div>
    </header>

    <section className="server-live-status panel" aria-label="Target status">
      <div className="server-live-status-title"><Server size={17} /><span>SERVER / SQLI TARGET</span></div>
      <div><span>Application</span><strong>{text(target?.scenario_id, "sqli_lab_v1")}</strong></div>
      <div><span>Version</span><strong>{text(target?.target_version, "unknown")}</strong></div>
      <div><span>Health</span><strong className={target?.status === "available" ? "is-healthy" : "is-unhealthy"}>{text(target?.status, "checking")}</strong></div>
      <div><span>Requests in episode</span><strong>{requestCount}</strong></div>
      <div><span>Recorded controls</span><strong>{controlCount}</strong></div>
    </section>

    {target?.status !== "available" && <div className="server-live-outage"><AlertTriangle size={17} />
      <span><strong>Target outage or executor delay.</strong> New requests cannot be verified while the target is unavailable. Existing recorded events remain replayable.</span>
      <button className="button button-secondary button-small" onClick={() => void refresh()}><RefreshCw size={13} />Check again</button>
    </div>}
    {error && <div className="error-banner"><AlertTriangle size={16} /><span>{error}</span></div>}

    <section className="server-live-toolbar panel" aria-label="Live and replay controls">
      <label>Timeline
        <select value={episodeId} onChange={(event) => selectEpisode(event.target.value)}>
          {episodes.length === 0 && <option value="">{loading ? "Loading episodes…" : "No recorded episodes"}</option>}
          {episodes.map((item) => <option key={item.episode_id} value={item.episode_id}>
            {item.phase} · {item.outcome?.winner ?? item.status} · {item.episode_id.slice(0, 8)} · {item.timestamp.slice(11, 19)} UTC
          </option>)}
        </select>
      </label>
      <div className="server-live-mode" role="group" aria-label="View mode">
        <button className={mode === "live" ? "is-active" : ""} onClick={() => setViewMode("live")}><Radio size={14} />Live</button>
        <button className={mode === "recorded" ? "is-active" : ""} onClick={() => setViewMode("recorded")}><Database size={14} />Recorded replay</button>
      </div>
      <label>Activity filter
        <span className="server-live-filter"><Filter size={14} aria-hidden="true" />
          <select value={filter} onChange={(event) => { setFilter(event.target.value as ActivityFilter); if (mode === "recorded") setCursor(1); }}>
            <option value="all">All requests and controls</option>
            <option value="agent">Red policy requests</option>
            <option value="legitimate">Legitimate verifier traffic</option>
            <option value="controls">Blue decisions and controls</option>
          </select>
        </span>
      </label>
      <div className="server-live-playback">
        {mode === "recorded" && <button className="button button-secondary button-small" onClick={() => setPaused((value) => !value)} aria-label={paused ? "Play replay" : "Pause replay"}>
          {paused ? <CirclePlay size={15} /> : <CirclePause size={15} />}{paused ? "Play" : "Pause"}
        </button>}
        <button className="button button-secondary button-small" onClick={step} disabled={!filteredEvents.length}><SkipForward size={15} />Step</button>
        <span>{visibleEvents.length} / {filteredEvents.length} displayed</span>
      </div>
    </section>

    <div className="server-live-method-note"><ShieldCheck size={14} />
      <span>Live reads committed records. Pausing freezes the display only; the server and event recorder continue independently.</span>
    </div>

    <div className="server-live-layout">
      <section className="server-live-event-panel panel">
        <div className="panel-head">
          <div><span className="panel-kicker">Recorded execution</span><h2 className="panel-title">{selectedEpisode ? `Episode ${selectedEpisode.episode_index}` : "Episode timeline"}</h2></div>
          {selectedEpisode && <span className={`pill ${selectedEpisode.status === "completed" ? "pill-green" : "pill-amber"}`}>{selectedEpisode.status}</span>}
        </div>
        <div className="server-live-event-list">
          {visibleEvents.map((event) => {
            const selected = activeEvent?.event_id === event.event_id && activeEvent.seq === event.seq;
            const request = isRequest(event);
            const payload = event.payload;
            const telemetry = object(payload.telemetry);
            const purpose = text(payload.purpose, request ? "request" : "control");
            const title = request
              ? `${text(payload.method, "POST")} ${text(payload.route, "/api/lookup")}`
              : event.event_type === "sqli_blue_decision"
                ? `Blue decision · ${text(object(payload.selected_action).operation, text(payload.action, text(payload.phase)))}`
                : `Control acknowledgement · ${text(payload.action)}`;
            const status = request ? text(payload.http_status, text(payload.status, "pending")) : text(object(payload.acknowledgement).status, text(payload.phase, "recorded"));
            const roleClass = request ? purpose === "policy" ? "red" : "green" : "blue";
            return <button className={`server-live-event ${selected ? "is-selected" : ""}`} key={`${event.event_id}:${event.seq}`} onClick={() => selectEvent(event)}>
              <time>{stamp(event.occurred_at)}<small>#{event.seq} · r{number(payload.step_id) === null ? "—" : number(payload.step_id)! + 1}</small></time>
              <span className={`timeline-dot ${roleClass}`} />
              <span className="server-live-event-copy"><strong>{title}</strong><small>{purpose.replaceAll("_", " ")} · {request ? text(telemetry.evidence_status, "evidence missing") : event.event_type.replaceAll("sqli_", "").replaceAll("_", " ")}</small></span>
              <span className={`server-live-event-status ${request && Number(payload.http_status) >= 400 ? "is-warning" : ""}`}>{request && Number.isFinite(Number(status)) ? `HTTP ${status}` : status.replaceAll("_", " ")}</span>
            </button>;
          })}
          {!loading && !visibleEvents.length && <div className="server-live-empty">No events match this filter yet.</div>}
          {loading && <div className="server-live-empty">Reading the local event store…</div>}
        </div>
        <div className="server-live-event-foot">{missingEvidenceCount > 0 && <span className="server-live-missing-count">{missingEvidenceCount} request result(s) have missing or mismatched evidence.</span>}</div>
      </section>

      <aside className="server-live-detail-panel panel">
        {activeEvent && isRequest(activeEvent)
          ? <TraceDetails
            trace={trace?.target_event_id === activeEvent.payload.target_event_id ? trace : null}
            loading={typeof activeEvent.payload.target_event_id === "string"
              && object(activeEvent.payload.telemetry).evidence_status === "verified"
              && trace?.target_event_id !== activeEvent.payload.target_event_id
              && traceError?.targetEventId !== activeEvent.payload.target_event_id}
            error={traceError && traceError.targetEventId === activeEvent.payload.target_event_id ? traceError.message : ""}
            event={activeEvent}
            decision={policyDecision}
          />
          : activeEvent
            ? <section className="server-live-inspector">
              <div className="server-live-inspector-head"><div><span className="panel-kicker">Control record</span><h2>{text(activeEvent.payload.action, "Blue policy event")}</h2></div><span className="pill pill-blue">persisted</span></div>
              <div className="server-live-id-row"><span>event <code>{activeEvent.event_id}</code></span><span>episode <code>{text(activeEvent.episode_id)}</code></span></div>
              <DetailJson label="Recorded policy and control event" value={activeEvent.payload} />
            </section>
            : <div className="server-live-empty server-live-empty-detail"><ShieldCheck size={22} /><strong>Choose a recorded event</strong><span>The inspector opens with target evidence and the actual stages the application recorded.</span></div>}
      </aside>
    </div>

    <footer className="server-live-footer">
      <span><Database size={13} /> Event store sequence {visibleEvents.at(-1)?.seq ?? 0}</span>
      <span>{selectedEpisode?.outcome?.verified_exposure ? "Verified disclosure recorded" : "No verified disclosure in selected episode"}</span>
      {selectedEpisode && <span>Episode {selectedEpisode.episode_id}</span>}
    </footer>
  </div>;
}
