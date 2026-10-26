"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RefreshCw,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useParams } from "next/navigation";
import { rangeApi, type RangeEvent, type TrainingRun } from "../../../lib/range-api";

const COLORS = { red: "#f07832", blue: "#91bce2", gold: "#d4af37", green: "#91bd91", muted: "#8f7965" };

function objectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function eventLabel(event: RangeEvent): { title: string; note: string } {
  const payload = objectValue(event.payload);
  const phase = String(payload.phase ?? "");
  if (event.event_type === "sqli_red_decision") return { title: "Red policy decision", note: "PPO selected bounded request components from its current observation." };
  if (event.event_type === "sqli_target_response" && phase === "request_submitted") return { title: "HTTP request dispatched", note: "The executor accepted the composed action for the fixed SQLi endpoint." };
  if (event.event_type === "sqli_target_response") return { title: "Server response and evidence", note: "The isolated target response is joined to its correlated server event." };
  if (event.event_type === "sqli_blue_decision" && phase === "policy_decision") return { title: "Blue policy decision", note: "Blue chose a control from sanitized request telemetry." };
  if (event.event_type === "sqli_blue_decision") return { title: "Control command submitted", note: "The target received a bounded, typed defensive action." };
  if (event.event_type === "sqli_control_acknowledged") return { title: "Defensive action acknowledged", note: "The target reports the control state and version it applied." };
  if (event.event_type === "sqli_verification") return { title: "Independent verification", note: "A controlled unauthorized-read probe and a legitimate query measured security and availability." };
  if (event.event_type === "sqli_reward") return { title: "Reward attribution", note: "Red and Blue rewards are shown separately from verified performance." };
  if (event.event_type === "sqli_transition") return { title: `${String(payload.role ?? "Agent")} transition`, note: "The accepted action, reward, next observation, and GAE boundary are persisted." };
  if (event.event_type === "sqli_policy_probe") return { title: `${String(payload.role ?? "Policy")} before / after probe`, note: "Both probability vectors use the same reference observation." };
  if (event.event_type === "sqli_ppo_update") return { title: `${String(payload.role ?? "PPO")} optimizer update`, note: "The role-local PPO optimizer changed its policy after the episode." };
  if (event.event_type === "sqli_episode_summary") return { title: "Episode performance", note: "Measured outcomes are recorded independently of optimizer metrics." };
  if (event.event_type === "sqli_episode_started") return { title: "Episode started", note: "Scenario and contract versions are pinned for this episode." };
  return { title: event.event_type.replaceAll("_", " "), note: "Persisted lab event." };
}

function EventDetails({ event }: { event: RangeEvent }) {
  const payload = objectValue(event.payload);
  const action = objectValue(payload.action ?? payload.selected_action);
  const response = objectValue(payload.response);
  const metrics = objectValue(payload.metrics);
  const components = objectValue(payload.components);
  const rows = Array.isArray(response.rows) ? response.rows as Record<string, unknown>[] : [];
  const facts: [string, string][] = [];
  if (action.operation) facts.push(["Blue action", String(action.operation)]);
  if (action.field) facts.push(["Input field", String(action.field)]);
  if (action.base_term) facts.push(["Base term", String(action.base_term)]);
  if (action.connector) facts.push(["Boolean connector", String(action.connector)]);
  if (action.predicate) facts.push(["Predicate", String(action.predicate)]);
  if (action.close_quote !== undefined) facts.push(["Quote boundary", action.close_quote ? "closed by policy" : "unchanged"]);
  if (action.comment !== undefined) facts.push(["Comment boundary", action.comment ? "selected" : "none"]);
  if (payload.http_status !== undefined) facts.push(["HTTP status", String(payload.http_status)]);
  if (payload.unauthorized_rows_returned !== undefined) facts.push(["Unauthorized rows", String(payload.unauthorized_rows_returned)]);
  if (payload.legitimate_query_served !== undefined) facts.push(["Legitimate query", payload.legitimate_query_served ? "served" : "blocked"]);
  if (payload.accepted !== undefined) facts.push(["PPO eligible", payload.accepted ? "accepted" : "excluded"]);
  if (payload.policy_version !== undefined) facts.push(["Policy version", String(payload.policy_version)]);
  if (payload.reward !== undefined) facts.push(["Reward", String(payload.reward)]);

  return <div className="space-y-4">
    {facts.length > 0 && <dl className="grid gap-2 sm:grid-cols-2">{facts.map(([label, value]) => <div key={label} className="rounded-lg border border-[#d4af37]/10 bg-[#150e09]/70 p-3"><dt className="mono-label">{label}</dt><dd className="mt-1 break-words text-xs text-[#e9d7bf]">{value}</dd></div>)}</dl>}
    {rows.length > 0 && <div className="overflow-x-auto rounded-lg border border-[#d4af37]/12"><table className="table-brutal"><thead><tr><th>Record</th><th>Owner</th><th>Handle</th><th>Department</th></tr></thead><tbody>{rows.slice(0, 8).map((row) => <tr key={String(row.record_ref)}><td>{String(row.record_ref)}</td><td>{String(row.owner_ref)}</td><td>{String(row.handle)}</td><td>{String(row.department)}</td></tr>)}</tbody></table></div>}
    {(Object.keys(components).length > 0 || Object.keys(metrics).length > 0 || event.event_type === "sqli_policy_probe") && <pre className="max-h-72 overflow-auto rounded-xl border border-[#d4af37]/10 bg-[#100a07] p-4 font-mono text-[10px] leading-5 text-[#c1aa91]">{JSON.stringify({ components, metrics, payload }, null, 2)}</pre>}
    {event.event_type === "sqli_target_response" && <pre className="max-h-72 overflow-auto rounded-xl border border-[#d4af37]/10 bg-[#100a07] p-4 font-mono text-[10px] leading-5 text-[#c1aa91]">{JSON.stringify({ request: payload.request, response: payload.response, telemetry: payload.telemetry, target_event_id: payload.target_event_id }, null, 2)}</pre>}
    {facts.length === 0 && rows.length === 0 && Object.keys(components).length === 0 && Object.keys(metrics).length === 0 && event.event_type !== "sqli_target_response" && <pre className="max-h-80 overflow-auto rounded-xl border border-[#d4af37]/10 bg-[#100a07] p-4 font-mono text-[10px] leading-5 text-[#c1aa91]">{JSON.stringify(payload, null, 2)}</pre>}
  </div>;
}

function MetricChart({
  data,
  lines,
  title,
  description,
  percent = false,
}: {
  data: Record<string, number | string | null>[];
  lines: { dataKey: string; label: string; color: string }[];
  title: string;
  description: string;
  percent?: boolean;
}) {
  return <section className="panel min-w-0 p-4 sm:p-5">
    <div className="mb-4"><h3 className="text-sm font-semibold text-[#f0e2cf]">{title}</h3><p className="mt-1 text-[10px] leading-5 text-[#8f7965]">{description}</p></div>
    <div className="h-64 w-full">
      {data.length ? <ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ top: 5, right: 12, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 4" stroke="rgba(212,175,55,.12)" />
        <XAxis dataKey="episode" tick={{ fill: COLORS.muted, fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(212,175,55,.18)" }} />
        <YAxis tick={{ fill: COLORS.muted, fontSize: 9 }} tickLine={false} axisLine={false} tickFormatter={(value: number) => percent ? `${Math.round(value * 100)}%` : Number(value).toFixed(2)} />
        <Tooltip contentStyle={{ background: "#1d130d", border: "1px solid rgba(212,175,55,.25)", borderRadius: 10, color: "#eadcc8", fontSize: 11 }} formatter={(value) => [typeof value === "number" ? (percent ? `${(value * 100).toFixed(0)}%` : value.toFixed(3)) : String(value), undefined]} />
        <Legend wrapperStyle={{ color: "#a99079", fontSize: 10 }} />
        {lines.map((line) => <Line key={line.dataKey} type="monotone" dataKey={line.dataKey} name={line.label} stroke={line.color} strokeWidth={2} dot={{ r: 2, fill: line.color }} activeDot={{ r: 4 }} connectNulls />)}
      </LineChart></ResponsiveContainer> : <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-[#d4af37]/15 text-xs text-[#8f7965]">No persisted episode points yet</div>}
    </div>
  </section>;
}

export default function SqliReplayPage() {
  const params = useParams<{ runId: string }>();
  const runId = params.runId;
  const [run, setRun] = useState<TrainingRun | null>(null);
  const [events, setEvents] = useState<RangeEvent[]>([]);
  const [eventsEpisodeId, setEventsEpisodeId] = useState("");
  const [summaryEvents, setSummaryEvents] = useState<RangeEvent[]>([]);
  const [episodes, setEpisodes] = useState<{ episode_id: string; episode_index: number; seed: number | null; status: string; config: Record<string, unknown> }[]>([]);
  const [selectedEpisodeId, setSelectedEpisodeId] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const summaryCursor = useRef(0);
  const summaryRunId = useRef("");

  const refresh = useCallback(async () => {
    if (!runId) return;
    if (summaryRunId.current !== runId) {
      summaryRunId.current = runId;
      summaryCursor.current = 0;
      setSummaryEvents([]);
    }
    try {
      const [runRecord, episodeList, summary] = await Promise.all([
        rangeApi.trainingRun(runId), rangeApi.episodes(runId), rangeApi.sqliSummary(runId, summaryCursor.current),
      ]);
      setRun(runRecord);
      setEpisodes(episodeList.items);
      if (summary.items.length) {
        summaryCursor.current = summary.next_after_seq;
        setSummaryEvents((current) => [...current, ...summary.items]);
      }
      setSelectedEpisodeId((current) => current && episodeList.items.some((item) => item.episode_id === current)
        ? current : episodeList.items.at(-1)?.episode_id ?? "");
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load the persisted SQLi event chain");
    } finally { setLoading(false); }
  }, [runId]);

  useEffect(() => { void Promise.resolve().then(refresh); const timer = window.setInterval(() => void refresh(), 4000); return () => window.clearInterval(timer); }, [refresh]);
  useEffect(() => {
    if (!runId || !selectedEpisodeId) return;
    let active = true;
    const load = async (replace: boolean) => {
      try {
        const timeline = await rangeApi.episodeTimeline(runId, selectedEpisodeId);
        if (!active) return;
        setEvents((current) => replace ? timeline.items : [...current, ...timeline.items.filter((item) => !current.some((saved) => saved.event_id === item.event_id))].sort((a, b) => a.seq - b.seq));
        if (replace) { setEventsEpisodeId(selectedEpisodeId); setCursor(0); setPlaying(false); }
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : "Could not load episode timeline");
      }
    };
    void load(true);
    const timer = window.setInterval(() => void load(false), 2200);
    return () => { active = false; window.clearInterval(timer); };
  }, [runId, selectedEpisodeId]);

  const guideEvents = useMemo(() => eventsEpisodeId === selectedEpisodeId ? events.filter((event) => event.event_type.startsWith("sqli_")) : [], [events, eventsEpisodeId, selectedEpisodeId]);
  const currentEvent = guideEvents[cursor] ?? null;
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setCursor((value) => {
      if (value >= guideEvents.length - 1) { setPlaying(false); return Math.max(0, guideEvents.length - 1); }
      return value + 1;
    }), 1800);
    return () => window.clearInterval(timer);
  }, [playing, guideEvents.length]);

  const summaries = useMemo(() => summaryEvents.filter((event) => event.event_type === "sqli_episode_summary").map((event) => objectValue(event.payload)).filter((item) => item.kind === "candidate"), [summaryEvents]);
  const returnsData = summaries.map((summary, index) => ({ episode: index + 1, Red: numberValue(summary.red_return), Blue: numberValue(summary.blue_return) }));
  const securityData = summaries.map((summary, index) => ({
    episode: index + 1,
    "Unauthorized read": numberValue(summary.unauthorized_read_rate),
    "Verified prevention": numberValue(summary.verified_prevention_rate),
    "Legitimate availability": numberValue(summary.legitimate_query_availability),
    "False positives": numberValue(summary.false_positive_rate),
  }));
  const updates = summaryEvents.filter((event) => event.event_type === "sqli_ppo_update").map((event, index) => {
    const payload = objectValue(event.payload);
    const metrics = objectValue(payload.metrics);
    return {
      updated: payload.updated === true || (payload.updated !== false && (numberValue(metrics.sample_count) ?? 0) > 0),
      role: String(payload.role ?? "PPO"),
      episode: Number(payload.update_index ?? index + 1),
      policyLoss: numberValue(metrics.policy_loss),
      valueLoss: numberValue(metrics.value_loss),
      advantage: numberValue(metrics.advantage_mean),
      entropy: numberValue(metrics.entropy),
      kl: numberValue(metrics.approx_kl),
      clipFraction: numberValue(metrics.clip_fraction),
    };
  });
  const updateEpisodes = [...new Set(updates.filter((item) => item.updated).map((item) => item.episode))].sort((a, b) => a - b);
  const learningData = updateEpisodes.map((episode) => {
    const red = updates.find((item) => item.episode === episode && item.role === "red" && item.updated);
    const blue = updates.find((item) => item.episode === episode && item.role === "blue" && item.updated);
    return {
      episode,
      "Red policy loss": red?.policyLoss ?? null,
      "Blue policy loss": blue?.policyLoss ?? null,
      "Red value loss": red?.valueLoss ?? null,
      "Blue value loss": blue?.valueLoss ?? null,
      "Red advantage": red?.advantage ?? null,
      "Blue advantage": blue?.advantage ?? null,
      "Red entropy": red?.entropy ?? null,
      "Blue entropy": blue?.entropy ?? null,
    };
  });
  const stabilityData = updateEpisodes.map((episode) => {
    const red = updates.find((item) => item.episode === episode && item.role === "red" && item.updated);
    const blue = updates.find((item) => item.episode === episode && item.role === "blue" && item.updated);
    return {
      episode,
      "Red KL": red?.kl ?? null,
      "Blue KL": blue?.kl ?? null,
      "Red clipping": red?.clipFraction ?? null,
      "Blue clipping": blue?.clipFraction ?? null,
    };
  });
  const selectedLabel = currentEvent ? eventLabel(currentEvent) : null;
  const scenario = String(run?.config.scenario_id ?? "");

  return <div className="space-y-7">
    <Link href="/sqli" className="inline-flex items-center gap-2 text-xs text-[#a99079] transition hover:text-[#f1d995]"><ArrowLeft size={14} />SQLi walkthrough</Link>
    <header className="flex flex-wrap items-end justify-between gap-4 border-b border-[#d4af37]/15 pb-6">
      <div className="max-w-3xl">
        <div className="eyebrow">Live HTTP evidence · persisted replay</div>
        <h1 className="page-heading break-all">SQLi episode room</h1>
        <p className="page-lede">Run <span className="font-mono text-[#e6ce92]">{runId}</span> · {scenario || "scenario loading"}. Red and Blue policies, target requests, independent checks, rewards, transitions, and optimizer changes are replayed from the stored event chain.</p>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-2 text-[10px] text-[#9d846c]">Episode
          <select className="select-brutal min-w-44 py-1.5" value={selectedEpisodeId} onChange={(event) => setSelectedEpisodeId(event.target.value)} disabled={!episodes.length}>
            {episodes.length === 0 && <option value="">Waiting for first episode</option>}
            {episodes.map((episode) => <option key={episode.episode_id} value={episode.episode_id}>#{episode.episode_index + 1} · {episode.status} · seed {episode.seed ?? "—"}</option>)}
          </select>
        </label>
        <button onClick={() => void refresh()} className="btn-brutal btn-brutal-ghost"><RefreshCw size={14} />Refresh records</button>
      </div>
    </header>
    {error && <div role="alert" className="rounded-lg border border-[#f07832]/30 bg-[#f07832]/10 p-3 text-xs text-[#ffd0af]">{error}</div>}
    <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {[["RUN STATUS", run?.status ?? "loading"], ["SCENARIO", String(run?.config.scenario_id ?? "—")], ["TARGET", String(run?.config.target_version ?? "—")], ["EPISODES / SUMMARY POINTS", `${episodes.length} / ${summaryEvents.filter((event) => event.event_type === "sqli_episode_summary").length}`]].map(([label, value]) => <div key={label} className="panel p-4"><div className="mono-label">{label}</div><div className="mt-2 break-all text-sm font-semibold text-[#eadcc8]">{value}</div></div>)}
    </section>

    <section className="panel p-4 sm:p-5" aria-labelledby="guided-replay-heading">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div><div className="eyebrow">Instructor mode</div><h2 id="guided-replay-heading" className="mt-1 text-lg font-semibold text-[#f0e2cf]">Guided event replay</h2><p className="mt-1 text-xs text-[#8f7965]">Step through the committed decision → request → defense → verification → reward → update stream.</p></div>
        <div className="flex items-center gap-2">
          <button className="btn-brutal btn-brutal-ghost" onClick={() => setPlaying(false)} aria-label="Pause replay"><Pause size={14} />Pause</button>
          <button className="btn-brutal btn-brutal-yellow" onClick={() => setPlaying(true)} aria-label="Play replay"><Play size={14} />Play</button>
        </div>
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
        <div className="min-w-0 rounded-xl border border-[#d4af37]/12 bg-[#120c08]/75 p-4 sm:p-5">
          {currentEvent && selectedLabel ? <>
            <div className="flex flex-wrap items-center justify-between gap-2"><span className="badge-brutal badge-yellow">Step {cursor + 1} / {guideEvents.length}</span><span className="font-mono text-[9px] text-[#8f7965]">seq {currentEvent.seq} · {new Date(currentEvent.occurred_at).toLocaleTimeString()}</span></div>
            <h3 className="mt-4 text-base font-semibold text-[#f3e6d5]">{selectedLabel.title}</h3>
            <p className="mt-1 max-w-3xl text-xs leading-6 text-[#a99079]">{selectedLabel.note}</p>
            <div className="mt-4"><EventDetails event={currentEvent} /></div>
          </> : <div className="flex min-h-52 items-center justify-center text-center text-xs text-[#8f7965]">{loading ? "Loading persisted replay…" : "No SQLi events are stored for this run yet."}</div>}
        </div>
        <div className="flex flex-col justify-between gap-4">
          <div className="rounded-xl border border-[#d4af37]/10 bg-[#120c08]/65 p-4">
            <div className="mono-label">Replay cursor</div>
            <div className="mt-2 font-mono text-2xl text-[#f1d995]">{String(cursor + (guideEvents.length ? 1 : 0)).padStart(3, "0")}<span className="text-sm text-[#8f7965]"> / {guideEvents.length}</span></div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/5"><div className="h-full rounded-full bg-gradient-to-r from-[#d4af37] to-[#f07832] transition-all" style={{ width: `${guideEvents.length ? ((cursor + 1) / guideEvents.length) * 100 : 0}%` }} /></div>
            <div className="mt-4 flex gap-2">
              <button className="btn-brutal btn-brutal-ghost flex-1" onClick={() => { setPlaying(false); setCursor((value) => Math.max(0, value - 1)); }} disabled={cursor <= 0}><ChevronLeft size={14} />Back</button>
              <button className="btn-brutal btn-brutal-yellow flex-1" onClick={() => { setPlaying(false); setCursor((value) => Math.min(guideEvents.length - 1, value + 1)); }} disabled={cursor >= guideEvents.length - 1}><ChevronRight size={14} />Advance</button>
            </div>
          </div>
          <div className="rounded-xl border border-[#d4af37]/10 bg-[#120c08]/65 p-4">
            <div className="mono-label">Current phase</div>
            <div className="mt-2 text-sm font-semibold text-[#eadcc8]">{selectedLabel?.title ?? "Waiting for episode"}</div>
            <p className="mt-2 text-[10px] leading-5 text-[#8f7965]">The event list is loaded from the hash-linked database stream. Refreshing restores the same sequence.</p>
          </div>
        </div>
      </div>
      <ol className="mt-4 flex gap-2 overflow-x-auto pb-1" aria-label="Replay event sequence">
        {guideEvents.map((event, index) => <li key={event.event_id} className="shrink-0"><button onClick={() => { setPlaying(false); setCursor(index); }} className={`rounded-lg border px-3 py-2 text-left transition ${index === cursor ? "border-[#d4af37]/40 bg-[#d4af37]/10" : "border-[#d4af37]/10 bg-[#160f0a]/55 hover:border-[#d4af37]/25"}`}><span className="block font-mono text-[8px] text-[#8f7965]">#{event.seq}</span><span className="mt-1 block max-w-32 truncate text-[10px] text-[#decbb7]">{eventLabel(event).title}</span></button></li>)}
      </ol>
    </section>

    <section aria-labelledby="performance-heading" className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3"><div><div className="eyebrow">Measured security and learning</div><h2 id="performance-heading" className="mt-1 text-xl font-semibold text-[#f0e2cf]">Performance outcomes</h2></div><span className="badge-brutal badge-green">Independent verifier metrics</span></div>
      <div className="grid gap-4 xl:grid-cols-2">
        <MetricChart data={returnsData} title="Per-episode returns" description="Red and Blue return totals from the episode reward components. Returns describe the training objective, not security by themselves." lines={[{ dataKey: "Red", label: "Red return", color: COLORS.red }, { dataKey: "Blue", label: "Blue return", color: COLORS.blue }]} />
        <MetricChart data={securityData} title="Verified security and availability" description="Unauthorized reads and prevention are measured on target responses. Legitimate availability and false positives remain separate." percent lines={[{ dataKey: "Unauthorized read", label: "Unauthorized read rate", color: COLORS.red }, { dataKey: "Verified prevention", label: "Verified prevention rate", color: COLORS.gold }, { dataKey: "Legitimate availability", label: "Legitimate query availability", color: COLORS.green }, { dataKey: "False positives", label: "False positive rate", color: COLORS.blue }]} />
      </div>
    </section>

    <section aria-labelledby="learning-heading" className="space-y-3">
      <div className="flex items-end justify-between gap-3"><div><div className="eyebrow">Separate optimizer view</div><h2 id="learning-heading" className="mt-1 text-xl font-semibold text-[#f0e2cf]">PPO learning signals</h2></div><Activity size={17} className="text-[#f07832]" /></div>
      <div className="grid gap-4 xl:grid-cols-2">
        <MetricChart data={learningData} title="Policy, value, advantage, entropy" description="Red and Blue are separate optimizers. Advantage estimates the value of an action above the critic&apos;s expected return." lines={[{ dataKey: "Red policy loss", label: "Red policy loss", color: COLORS.red }, { dataKey: "Blue policy loss", label: "Blue policy loss", color: COLORS.blue }, { dataKey: "Red value loss", label: "Red value loss", color: COLORS.gold }, { dataKey: "Blue value loss", label: "Blue value loss", color: COLORS.green }, { dataKey: "Red advantage", label: "Red advantage", color: "#dc9970" }, { dataKey: "Blue advantage", label: "Blue advantage", color: "#a2c8e8" }]} />
        <MetricChart data={stabilityData} title="KL movement and clipping" description="Approximate KL and clipping show how far the PPO update moved from its sampled policy. Entropy is shown beside losses in the companion chart." lines={[{ dataKey: "Red KL", label: "Red approximate KL", color: COLORS.red }, { dataKey: "Blue KL", label: "Blue approximate KL", color: COLORS.blue }, { dataKey: "Red clipping", label: "Red clip fraction", color: COLORS.gold }, { dataKey: "Blue clipping", label: "Blue clip fraction", color: COLORS.green }]} />
      </div>
      <div className="rounded-xl border border-[#d4af37]/15 bg-[#130d09]/70 p-4 text-xs leading-6 text-[#b69c82]"><strong className="text-[#eadcc8]">Reading this correctly:</strong> state is the policy-visible summary; action is the selected component set or defense; reward is the immediate score; return is the episode sum; advantage compares the sampled action with the critic&apos;s expectation. Loss, entropy, KL, or a changed probability is not evidence of improved security. Use the verifier rates above and held-out runs.</div>
    </section>

    <section className="panel p-4 sm:p-5">
      <div className="flex items-center justify-between gap-3"><div><div className="eyebrow">Action-probability probes</div><h2 className="mt-1 text-base font-semibold text-[#f0e2cf]">Same observation, before and after update</h2></div><span className="badge-brutal">Policy versioned</span></div>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        {guideEvents.filter((event) => event.event_type === "sqli_policy_probe").map((event) => {
          const payload = objectValue(event.payload);
          return <div key={event.event_id} className="rounded-xl border border-[#d4af37]/10 bg-[#130d09]/60 p-4"><div className="flex items-center justify-between"><span className={`badge-brutal ${payload.role === "red" ? "badge-red" : "badge-blue"}`}>{String(payload.role)} PPO</span><span className="font-mono text-[9px] text-[#8f7965]">v{String(payload.policy_version_before)} → v{String(payload.policy_version_after)}</span></div><div className="mt-3 grid grid-cols-2 gap-3"><div><div className="mono-label">Before</div><pre className="mt-1 overflow-auto rounded-lg bg-[#100a07] p-2 font-mono text-[9px] leading-4 text-[#a99079]">{JSON.stringify(payload.before_action_probabilities)}</pre></div><div><div className="mono-label">After</div><pre className="mt-1 overflow-auto rounded-lg bg-[#100a07] p-2 font-mono text-[9px] leading-4 text-[#f1d995]">{JSON.stringify(payload.after_action_probabilities)}</pre></div></div><p className="mt-2 text-[9px] text-[#8f7965]">{String(payload.same_reference_observation) === "true" ? "Both vectors use the identical stored reference observation." : "Reference observation status unknown."}</p></div>;
        })}
      </div>
      {!guideEvents.some((event) => event.event_type === "sqli_policy_probe") && <p className="mt-4 rounded-lg border border-dashed border-[#d4af37]/15 p-5 text-center text-xs text-[#8f7965]">Policy probes appear after the first episode-boundary PPO update.</p>}
    </section>

    <div className="flex items-center gap-2 text-[10px] text-[#8f7965]"><span className="status-dot status-dot-green" />Live server evidence is sourced from the isolated HTTP target. Graph simulation runs are stored elsewhere and do not appear on this replay.</div>
  </div>;
}
