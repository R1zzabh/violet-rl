"use client";

import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type RangeEvent, type RangeRun } from "@/lib/api";

const label = (value: unknown) =>
  typeof value === "string" || typeof value === "number" || typeof value === "boolean"
    ? String(value)
    : value == null ? "—" : JSON.stringify(value);

export default function RangeTrainingPage() {
  const [runs, setRuns] = useState<RangeRun[]>([]);
  const [selected, setSelected] = useState("");
  const [events, setEvents] = useState<RangeEvent[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    setBusy(true);
    try {
      const result = await api.rangeRuns();
      setRuns(result.runs);
      const id = selected && result.runs.some((run) => run.run_id === selected)
        ? selected : result.runs[0]?.run_id ?? "";
      setSelected(id);
      setEvents(id ? (await api.rangeEvents(id)).events : []);
      setError("");
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }, [selected]);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  const chooseRun = async (id: string) => {
    setSelected(id);
    try {
      setEvents(id ? (await api.rangeEvents(id)).events : []);
      setError("");
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : String(reason));
    }
  };

  const updates = events.filter((event) => event.event_type === "learning_policy_update");
  const transitions = events.filter((event) => event.event_type === "learning_transition");
  return (
    <main className="brutal-container max-w-[1280px] mx-auto w-full px-6 py-8">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <div>
          <p className="mono text-xs text-violet-400">HTTP TRAINING RANGE</p>
          <h1 className="text-3xl font-bold mt-2">Real-range learning</h1>
          <p className="text-muted-foreground mt-2">Persisted transitions, verifier outcomes, and optimizer updates from the typed executor path.</p>
        </div>
        <button className="btn-brutal" onClick={() => void refresh()} disabled={busy}>{busy ? "Refreshing…" : "Refresh events"}</button>
      </div>
      {error && <p className="border border-red-500 p-3 mb-4 text-red-300">Range API unavailable: {error}</p>}
      <section className="grid md:grid-cols-3 gap-4 mb-6">
        <div className="card-brutal p-4"><p className="mono text-xs text-muted-foreground">RUNS</p><strong className="text-2xl">{runs.length}</strong></div>
        <div className="card-brutal p-4"><p className="mono text-xs text-muted-foreground">PERSISTED TRANSITIONS</p><strong className="text-2xl">{transitions.length}</strong></div>
        <div className="card-brutal p-4"><p className="mono text-xs text-muted-foreground">RECORDED PPO UPDATES</p><strong className="text-2xl">{updates.length}</strong></div>
      </section>
      <section className="card-brutal p-4 mb-6">
        <label className="mono text-xs block mb-2" htmlFor="range-run">RUN</label>
        <select id="range-run" className="bg-black border border-white/20 p-2 w-full" value={selected} onChange={(event) => void chooseRun(event.target.value)}>
          {runs.length === 0 && <option value="">No persisted real-range runs</option>}
          {runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {run.status} · {run.event_count} events</option>)}
        </select>
      </section>
      <section className="card-brutal p-4">
        <h2 className="font-bold text-xl mb-3">Event replay</h2>
        <p className="text-sm text-muted-foreground mb-4">This view reads the hash-linked event history. It does not submit or replay historical actions to the target.</p>
        {events.length === 0 ? <p className="mono text-sm text-muted-foreground">No events for this run.</p> :
          <ol className="space-y-3 max-h-[65vh] overflow-auto">
            {events.map((event) => <li key={`${event.run_id}-${event.seq}`} className="border border-white/10 p-3">
              <div className="flex flex-wrap justify-between gap-2 mono text-xs"><strong>#{event.seq} {event.event_type}</strong><time>{new Date(event.occurred_at).toLocaleString()}</time></div>
              <div className="mt-2 grid sm:grid-cols-2 gap-x-4 gap-y-1 text-sm">
                <span>episode: {event.episode_id ?? "—"}</span>
                {Object.entries(event.payload).map(([key, value]) => <span key={key}>{key}: {label(value)}</span>)}
              </div>
            </li>)}
          </ol>}
      </section>
    </main>
  );
}
