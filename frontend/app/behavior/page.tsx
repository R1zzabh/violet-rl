"use client";
// app/behavior/page.tsx — Learned Behaviors, driven exclusively by the
// behavior-discovery report generated from real evaluation trajectories
// (runs/<run_id>/evaluation/behavior_report.json, served by /behavior/{run_id}).
// Only observable policy tendencies are reported — never claimed cognition.

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, RefreshCw, Search } from "lucide-react";
import {
  ApiError,
  api,
  type BehaviorReport,
  type RunSummary,
} from "@/lib/api";

function Heatmap({ data, title }: { data: Record<string, Record<string, number>>; title: string }) {
  const states = Object.keys(data);
  const actions = [...new Set(states.flatMap((s) => Object.keys(data[s])))].sort();
  const max = Math.max(1, ...states.flatMap((s) => Object.values(data[s])));

  if (states.length === 0) {
    return (
      <div className="mono text-xs text-muted-foreground p-4 border border-[#222]">
        NOT AVAILABLE — no conditioned trajectory data in this report.
      </div>
    );
  }

  const fill = (v: number) =>
    v === 0
      ? "#161616"
      : `rgb(${Math.round(255 * (v / max))}, ${Math.round(60 * (1 - v / max))}, ${Math.round(180 * (1 - v / max))})`;

  return (
    <div className="overflow-x-auto">
      <div className="mono text-xs font-bold mb-2">{title}</div>
      <table className="table-brutal">
        <thead>
          <tr>
            <th>STATE</th>
            {actions.map((a) => (
              <th key={a} className="text-center">
                {a}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {states.map((s) => (
            <tr key={s}>
              <td className="mono text-white font-bold whitespace-nowrap">{s}</td>
              {actions.map((a) => {
                const v = data[s][a] ?? 0;
                return (
                  <td key={a} className="text-center mono">
                    <span
                      className="inline-block w-9 h-6 border border-black leading-6"
                      style={{ backgroundColor: fill(v), color: v / max > 0.5 ? "#000" : "#9a9a9a" }}
                    >
                      {v}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mono text-[10px] text-muted-foreground mt-2">
        cell = observed event count · rows = measured condition · cols = action selected · max = {max}
      </div>
    </div>
  );
}

function BehaviorCards({
  behaviors,
  accent,
}: {
  behaviors: BehaviorReport["red_behaviors"];
  accent: string;
}) {
  if (behaviors.length === 0) {
    return (
      <div className="card-brutal p-5 mono text-sm text-muted-foreground">
        NOT AVAILABLE — no behaviors met the evidence threshold in this report.
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {behaviors.map((b) => (
        <div key={b.id} className="card-brutal p-5 border-l-[6px]" style={{ borderLeftColor: accent }}>
          <div className="flex items-start justify-between mb-3">
            <span className="badge-brutal mono" style={{ borderColor: accent, color: accent }}>
              {b.id}
            </span>
            <span
              className="badge-brutal mono"
              style={{
                borderColor: b.confidence === "HIGH" ? "#2ecc71" : "#f5c518",
                color: b.confidence === "HIGH" ? "#2ecc71" : "#f5c518",
              }}
            >
              confidence {b.confidence}
            </span>
          </div>
          <div className="space-y-2 text-sm">
            <div>
              <div className="section-label mb-1">Condition</div>
              <div className="text-white font-semibold">{b.condition}</div>
            </div>
            <div>
              <div className="section-label mb-1">Observed behavior</div>
              <div className="text-white font-semibold">{b.observed_behavior}</div>
            </div>
            <div className="flex gap-6 pt-2 border-t border-[#222]">
              <div>
                <div className="section-label mb-1">Frequency</div>
                <span className="mono text-violet-400 font-bold">{b.frequency}</span>
              </div>
              <div>
                <div className="section-label mb-1">Baseline</div>
                <span className="mono text-muted-foreground">{b.baseline}</span>
              </div>
            </div>
            <div className="mono text-[11px] text-muted-foreground">{b.improvement}</div>
            {b.side_effects && b.side_effects.length > 0 && (
              <div className="mono text-[11px] text-yellow-400">side effects: {b.side_effects.join("; ")}</div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function BehaviorPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [report, setReport] = useState<BehaviorReport | null>(null);
  const [error, setError] = useState<string>("");

  const load = useCallback(async () => {
    try {
      const resp = await api.runs();
      setRuns(resp.runs);
      if (resp.runs.length > 0) setRunId((cur) => cur ?? resp.runs[0].run_id);
      setError("");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void Promise.resolve().then(load);
  }, [load]);

  // Fetched per selected run; staleness is derived at render time instead of
  // clearing synchronously inside the effect.
  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    api
      .behavior(runId)
      .then((r) => {
        if (!cancelled) setReport(r);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Search size={14} className="text-violet-400" />
            <span className="text-xs mono uppercase tracking-[0.2em] text-muted-foreground font-bold">
              Behavior Discovery Engine
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight">Learned Behaviors</h1>
        </div>
        <div className="flex items-center gap-2">
          {runs.length > 0 && (
            <select
              className="border-2 border-[#333] bg-[#111] px-2 py-1.5 mono text-xs"
              value={runId ?? ""}
              onChange={(e) => setRunId(e.target.value)}
            >
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id}
                </option>
              ))}
            </select>
          )}
          <button className="btn-brutal btn-brutal-ghost btn-brutal-sm" onClick={() => void load()}>
            <RefreshCw size={13} /> REFRESH
          </button>
        </div>
      </div>

      <div className="card-brutal bg-[#111] border-yellow-500 border-2">
        <div className="flex items-start gap-3">
          <AlertTriangle size={16} className="text-yellow-400 shrink-0 mt-0.5" />
          <div className="text-sm">
            <span className="mono uppercase tracking-[0.12em] text-yellow-400 font-bold text-xs">Methodology</span>
            <p className="text-muted-foreground mt-1 leading-relaxed">
              Behaviors are discovered by analyzing stored evaluation trajectories
              (runs/&lt;run_id&gt;/evaluation/). Each card reports the measured selection frequency
              of the policy under a condition, the baseline frequency, and a confidence rating.
              We do not claim the network &quot;understands&quot; anything — only observable
              action tendencies are reported.
            </p>
          </div>
        </div>
      </div>

      {error && (
        <div className="card-brutal border-red-500 border-2 p-4">
          <div className="mono text-xs font-bold text-red-400 uppercase mb-1">Not available</div>
          <p className="text-sm text-muted-foreground">{error}</p>
          <p className="text-xs mono text-muted-foreground mt-2">
            Behavior artifacts are generated by{" "}
            <code className="text-violet-400">backend/scripts/generate_eval_artifacts.py</code> after training.
          </p>
        </div>
      )}

      {report && report.run_id === runId && (
        <>
          <section>
            <div className="flex items-center gap-2 mb-4">
              <span className="status-dot status-dot-red" />
              <h2 className="text-lg font-bold tracking-tight">🔴 RED — observed policy tendencies</h2>
            </div>
            <BehaviorCards behaviors={report.red_behaviors} accent="#ff2a2a" />
          </section>

          <section>
            <div className="flex items-center gap-2 mb-4">
              <span className="status-dot status-dot-blue" />
              <h2 className="text-lg font-bold tracking-tight">🔵 BLUE — observed policy tendencies</h2>
            </div>
            <BehaviorCards behaviors={report.blue_behaviors} accent="#2a7aff" />
          </section>

          <section className="card-brutal-lg space-y-6">
            <div className="section-label mb-0">State × action heatmaps (real event counts)</div>
            <Heatmap data={report.blue_heatmap} title="BLUE — state × action" />
            <Heatmap data={report.red_heatmap} title="RED — state × action" />
          </section>

          <p className="mono text-[11px] text-muted-foreground">
            Source: <span className="text-violet-400">backend/runs/{report.run_id}/evaluation/behavior_report.json</span>
          </p>
        </>
      )}
    </div>
  );
}
