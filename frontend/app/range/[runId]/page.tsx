"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowLeft, CircleStop, RefreshCw } from "lucide-react";
import { rangeApi, type RangeEvent, type TrainingRun } from "../../../lib/range-api";

function evidenceLabel(event: RangeEvent): { label: string; tone: string } {
  const payload = event.payload;
  if (event.event_type === "agent_decision_recorded") return { label: "Decision recorded", tone: "text-[#d4af37]" };
  if (event.event_type === "action_submitted") return { label: "Attempt submitted", tone: "text-[#e3c879]" };
  if (event.event_type === "target_evidence_ingested") return { label: "Request reached target", tone: "text-[#79b8ff]" };
  if (event.event_type === "executor_result") {
    const finding = payload.finding as Record<string, unknown> | undefined;
    if (finding?.status === "confirmed") return { label: "Finding confirmed", tone: "text-[#ff9b8e]" };
    return { label: "Response recorded", tone: "text-[#e6d6c4]" };
  }
  if (event.event_type === "detection_suspected") return { label: "Finding suspected", tone: "text-[#e3c879]" };
  if (event.event_type === "target_control_acknowledged") return { label: "Control applied", tone: "text-[#8fc49a]" };
  if (event.event_type === "target_control_verified") return { label: "Control verified", tone: "text-[#a8d0ad]" };
  if (event.event_type === "reward_recorded") return { label: "Reward recorded", tone: "text-[#e6d6c4]" };
  return { label: event.event_type.replaceAll("_", " "), tone: "text-[#c1aa91]" };
}

export default function RangeReplayPage({ params }: { params: Promise<{ runId: string }> }) {
  const [runId, setRunId] = useState("");
  const [run, setRun] = useState<TrainingRun | null>(null);
  const [events, setEvents] = useState<RangeEvent[]>([]);
  const [selected, setSelected] = useState<RangeEvent | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const cursor = events.length ? events[events.length - 1].seq : 0;

  useEffect(() => { void params.then(({ runId: id }) => setRunId(id)); }, [params]);

  useEffect(() => {
    if (!runId) return;
    let active = true;
    let source: EventSource | null = null;
    async function connect() {
      try {
        const [runRecord, replay] = await Promise.all([
          rangeApi.trainingRun(runId), rangeApi.events(runId, 0),
        ]);
        if (!active) return;
        setRun(runRecord);
        setEvents(replay.events);
        setLoading(false);
        const afterSeq = replay.events.at(-1)?.seq ?? 0;
        source = new EventSource(rangeApi.streamUrl(runId, afterSeq));
        source.onmessage = (message) => {
          try {
            const event = JSON.parse(message.data) as RangeEvent;
            setEvents((previous) => previous.some((item) => item.seq === event.seq)
              ? previous : [...previous, event].sort((a, b) => a.seq - b.seq));
            setError("");
          } catch { setError("The live stream returned an invalid event; replay remains available."); }
        };
        source.onerror = () => setError("Live stream disconnected; the browser will reconnect and replay from its last committed sequence.");
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Replay service unavailable");
        setLoading(false);
      }
    }
    void connect();
    const timer = window.setInterval(() => {
      void rangeApi.trainingRun(runId).then(setRun).catch(() => undefined);
    }, 4000);
    return () => { active = false; source?.close(); window.clearInterval(timer); };
  }, [runId]);

  async function refresh() {
    if (!runId) return;
    try {
      const [runRecord, replay] = await Promise.all([
        rangeApi.trainingRun(runId), rangeApi.events(runId, cursor),
      ]);
      setRun(runRecord);
      if (replay.events.length) setEvents((previous) => [...previous, ...replay.events.filter((event) => !previous.some((item) => item.seq === event.seq))].sort((a, b) => a.seq - b.seq));
      setError("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Replay service unavailable"); }
  }

  async function stop() {
    if (!runId) return;
    try { setRun(await rangeApi.stopTrainingRun(runId)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not request stop"); }
  }

  return <div className="space-y-6 pb-12">
    <Link href="/range" className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#8f7965] transition hover:text-[#f1d995]"><ArrowLeft size={14} />All range runs</Link>
    <header className="flex flex-wrap items-end justify-between gap-4 border-b border-[#d4af37]/15 pb-6">
      <div><p className="font-mono text-[10px] uppercase tracking-[.2em] text-[#f1d995]">Violet Ledger · persisted replay</p><h1 className="mt-2 break-all text-2xl font-semibold text-[#f5e6d0]">{runId || "Loading run…"}</h1><p className="mt-2 text-sm text-[#8f7965]">The graph simulator has a separate run store and does not appear on this timeline.</p></div>
      <div className="flex items-center gap-2"><button onClick={() => void refresh()} className="btn-brutal btn-brutal-ghost"><RefreshCw size={13} />Refresh · seq {cursor}</button>{run && ["queued", "running"].includes(run.status) && <button onClick={() => void stop()} className="btn-brutal btn-brutal-red"><CircleStop size={13} />Request stop</button>}</div>
    </header>
    {error && <div role="alert" className="rounded-xl border border-[#ff7868]/20 bg-[#ff7868]/10 px-4 py-3 text-sm text-[#ffd2ca]">{error}</div>}
    {run && <section className="grid gap-3 sm:grid-cols-4">
      {[["STATUS", run.status], ["POLICY ROLE", run.role.toUpperCase()], ["MODE", run.mode], ["EVIDENCE LEVEL", events.some((event) => event.event_type === "target_evidence_ingested") ? "Target evidence present" : "No target request recorded"]].map(([label, value]) => <div key={label} className="panel p-4"><div className="font-mono text-[10px] tracking-[.15em] text-[#8f7965]">{label}</div><div className="mt-2 text-sm text-[#e6d6c4]">{value}</div></div>)}
    </section>}
    <div className="grid min-h-[480px] gap-4 lg:grid-cols-[.9fr_1.1fr]">
      <section className="panel p-4">
        <div className="mb-4 flex items-center justify-between"><div><h2 className="text-sm font-semibold text-[#eadcc8]">Event chain</h2><p className="mt-1 text-[11px] text-[#8f7965]">Committed events · hash-linked sequence</p></div><span className="font-mono text-xs text-[#8f7965]">{events.length}</span></div>
        <div className="max-h-[680px] space-y-1 overflow-auto pr-1">
          {loading && events.length === 0 && <p className="p-6 text-sm text-[#8f7965]">Loading persisted records…</p>}
          {!loading && events.length === 0 && <p className="rounded-lg border border-dashed border-[#d4af37]/15 p-6 text-sm text-[#8f7965]">No events committed yet. Training may still be queued or unavailable.</p>}
          {events.map((event) => {
            const evidence = evidenceLabel(event);
            return <button key={event.event_id} onClick={() => setSelected(event)} className={`block w-full rounded-xl border p-3 text-left transition ${selected?.event_id === event.event_id ? "border-[#d4af37]/40 bg-[#d4af37]/10" : "border-transparent hover:border-[#d4af37]/15 hover:bg-white/[.025]"}`}>
              <div className="flex items-center justify-between gap-2"><span className="font-mono text-[10px] text-[#8f7965]">#{String(event.seq).padStart(4, "0")}</span><span className={`text-xs font-medium ${evidence.tone}`}>{evidence.label}</span></div>
              <div className="mt-1 truncate font-mono text-[11px] text-[#c1aa91]">{event.event_type}</div>
              <div className="mt-1 text-[10px] text-[#8f7965]">{new Date(event.occurred_at).toLocaleTimeString()} {event.episode_id ? `· episode ${event.episode_id.slice(0, 8)}` : ""}</div>
            </button>;
          })}
        </div>
      </section>
      <section className="panel p-4">
        <div className="mb-4"><h2 className="text-sm font-semibold text-[#eadcc8]">Evidence inspector</h2><p className="mt-1 text-[11px] text-[#8f7965]">Sanitized event payload from the canonical replay API</p></div>
        {selected ? <div className="space-y-4">
          <div className="flex flex-wrap gap-2"><span className="rounded-full border border-[#d4af37]/15 px-2.5 py-1 font-mono text-[10px] text-[#c1aa91]">seq {selected.seq}</span><span className={`rounded-full border border-[#d4af37]/15 px-2.5 py-1 font-mono text-[10px] ${evidenceLabel(selected).tone}`}>{evidenceLabel(selected).label}</span></div>
          <pre className="max-h-[590px] overflow-auto rounded-lg border border-[#d4af37]/10 bg-[#130d09] p-4 text-[11px] leading-5 text-[#decbb7]">{JSON.stringify({ event_id: selected.event_id, event_type: selected.event_type, occurred_at: selected.occurred_at, episode_id: selected.episode_id, previous_hash: selected.previous_hash, event_hash: selected.event_hash, payload: selected.payload }, null, 2)}</pre>
        </div> : <div className="flex min-h-[320px] items-center justify-center rounded-xl border border-dashed border-[#d4af37]/15 p-8 text-center text-sm text-[#8f7965]">Select an event to inspect its persisted evidence and provenance.</div>}
      </section>
    </div>
    <p className="text-[11px] leading-5 text-[#8f7965]">This view reports evidence levels independently. It never labels an action as a confirmed finding without a verifier result. Policy improvement requires held-out multi-seed evaluation and is not inferred from training reward.</p>
  </div>;
}
