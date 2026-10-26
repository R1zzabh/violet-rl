"use client";
// app/training/page.tsx — Training Runs, driven exclusively by real backend
// artifacts (runs/<run_id>/metrics/training.jsonl served by /runs and
// /runs/{id}/metrics). If no runs exist, this page says NOT AVAILABLE.

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { GitBranch, RefreshCw } from "lucide-react";
import {
  ApiError,
  api,
  type RunMetricsResponse,
  type RunSummary,
} from "@/lib/api";

function MetricChart({
  data,
  accent,
  label,
}: {
  data: { x: number; y: number }[];
  accent: string;
  label: string;
}) {
  // Minimal dependency-free SVG line chart: fast, no client chart lib needed
  // for a single series, and re-renders only when data changes.
  const W = 900;
  const H = 220;
  const PAD = 30;
  const xs = data.map((d) => d.x);
  const ys = data.map((d) => d.y);
  const xMin = Math.min(...xs, 0);
  const xMax = Math.max(...xs, 1);
  const yMin = Math.min(...ys, 0);
  const yMax = Math.max(...ys, 1);
  const sx = (x: number) => PAD + ((x - xMin) / (xMax - xMin || 1)) * (W - 2 * PAD);
  const sy = (y: number) => H - PAD - ((y - yMin) / (yMax - yMin || 1)) * (H - 2 * PAD);
  const path = data.map((d, i) => `${i === 0 ? "M" : "L"}${sx(d.x).toFixed(1)},${sy(d.y).toFixed(1)}`).join(" ");
  const zeroY = sy(0);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: H }} role="img" aria-label={label}>
        <line x1={PAD} y1={zeroY} x2={W - PAD} y2={zeroY} stroke="rgba(212,175,55,.2)" strokeDasharray="4 4" />
        <path d={path} fill="none" stroke={accent} strokeWidth={2} />
        <text x={PAD} y={12} fill="#8f7965" fontSize={10} fontFamily="'Space Mono', monospace">
          {label}
        </text>
        <text x={PAD} y={H - 8} fill="#8f7965" fontSize={9} fontFamily="'Space Mono', monospace">
          {xMin}
        </text>
        <text x={W - PAD} y={H - 8} fill="#8f7965" fontSize={9} fontFamily="'Space Mono', monospace" textAnchor="end">
          {xMax}
        </text>
        <text x={PAD} y={yMin === yMax ? H / 2 : H - PAD + 4} fill="#8f7965" fontSize={9} fontFamily="'Space Mono', monospace">
          {yMin.toFixed(1)}
        </text>
        <text x={PAD} y={PAD - 6} fill="#8f7965" fontSize={9} fontFamily="'Space Mono', monospace">
          {yMax.toFixed(1)}
        </text>
      </svg>
    </div>
  );
}

export default function TrainingPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<RunMetricsResponse | null>(null);
  const [error, setError] = useState<string>("");

  const loadRuns = useCallback(async () => {
    try {
      const resp = await api.runs();
      setRuns(resp.runs);
      if (resp.runs.length > 0) {
        setSelected((cur) => cur ?? resp.runs[0].run_id);
      }
      setError("");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setRuns([]);
    }
  }, []);

  // Initial load deferred to a microtask so no setState runs synchronously
  // inside the effect body (react-hooks/set-state-in-effect).
  useEffect(() => {
    void Promise.resolve().then(loadRuns);
  }, [loadRuns]);

  // Metrics are fetched per selected run; staleness is derived at render time
  // (a stale payload simply doesn't match `selected`) instead of being cleared
  // synchronously in the effect.
  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    api
      .runMetrics(selected)
      .then((m) => {
        if (!cancelled) setMetrics(m);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const rewardSeries = useMemo(
    () =>
      (metrics?.metrics ?? []).flatMap((m) =>
        typeof m.global_step === "number" && Number.isFinite(m.global_step) &&
          typeof m.mean_episode_reward === "number" && Number.isFinite(m.mean_episode_reward)
          ? [{ x: m.global_step, y: m.mean_episode_reward }]
          : [],
      ),
    [metrics],
  );
  const successSeries = useMemo(
    () =>
      (metrics?.metrics ?? []).flatMap((m) =>
        typeof m.global_step === "number" && Number.isFinite(m.global_step) &&
          typeof m.attack_success_rate === "number" && Number.isFinite(m.attack_success_rate)
          ? [{ x: m.global_step, y: m.attack_success_rate }]
          : [],
      ),
    [metrics],
  );

  // Only show metrics that belong to the currently selected run.
  const current = metrics !== null && metrics.run_id === selected ? metrics : null;
  const last = current?.metrics[current.metrics.length - 1];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <GitBranch size={14} className="text-[#d4af37]" />
            <span className="text-xs mono uppercase tracking-[0.2em] text-muted-foreground font-bold">
              Training Runs
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight">Training Progress</h1>
        </div>
        <button className="btn-brutal btn-brutal-ghost btn-brutal-sm" onClick={() => void loadRuns()}>
          <RefreshCw size={13} /> REFRESH
        </button>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[#d4af37]/20 bg-[#24170f]/70 px-4 py-3 text-xs text-[#decbb7]">
        <span>These artifacts describe the graph simulator only. Live target training and evidence are kept in the SQLi lab and Violet Ledger archives.</span>
        <Link href="/sqli" className="font-mono uppercase tracking-wider text-[#f1d995] hover:text-white">Open SQLi learning lab →</Link>
      </div>

      {error && (
        <div className="card-brutal border-red-500 border-2 p-4">
          <div className="mono text-xs font-bold text-red-400 uppercase mb-1">Backend unavailable</div>
          <p className="text-sm text-muted-foreground">{error}</p>
        </div>
      )}

      {runs !== null && runs.length === 0 && !error && (
        <div className="card-brutal p-6 mono text-sm text-muted-foreground">
          NOT AVAILABLE — no graph training runs found in <span className="text-[#d4af37]">backend/runs/</span>.
          Train one with <code className="text-[#d4af37]">violet train</code>, then refresh.
        </div>
      )}

      {runs !== null && runs.length > 0 && (
        <>
          {/* Run selector */}
          <div className="card-brutal">
            <div className="section-label mb-3">EXPERIMENT REGISTRY — runs/ artifacts</div>
            <div className="overflow-x-auto">
              <table className="table-brutal">
                <thead>
                  <tr>
                    <th>Run ID</th>
                    <th>Algorithm</th>
                    <th>Updates</th>
                    <th>Global step</th>
                    <th>Mean ep. reward</th>
                    <th>Attack success</th>
                    <th>Checkpoint</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr
                      key={r.run_id}
                      onClick={() => setSelected(r.run_id)}
                      style={{ cursor: "pointer", outline: selected === r.run_id ? "2px solid #d4af37" : undefined }}
                    >
                      <td className="mono font-bold text-[#d4af37]">{r.run_id}</td>
                      <td>
                        <span className="badge-brutal mono" style={{ color: "#79b8ff", borderColor: "#79b8ff" }}>
                          {r.algorithm}
                        </span>
                      </td>
                      <td className="mono">{r.updates}</td>
                      <td className="mono">{r.global_step ?? "—"}</td>
                      <td className="mono">{typeof r.mean_episode_reward === "number" ? r.mean_episode_reward.toFixed(2) : "—"}</td>
                      <td className="mono">{typeof r.attack_success_rate === "number" ? `${(r.attack_success_rate * 100).toFixed(0)}%` : "—"}</td>
                      <td>
                        <span
                          className="badge-brutal mono"
                          style={{ color: r.has_checkpoint ? "#8fc49a" : "#d4af37", borderColor: r.has_checkpoint ? "#8fc49a" : "#d4af37" }}
                        >
                          {r.has_checkpoint ? "best" : "none"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {current && current.metrics.length === 0 && (
            <div className="card-brutal p-6 mono text-sm text-muted-foreground">NOT AVAILABLE — run has no metrics yet.</div>
          )}

          {current && current.metrics.length > 0 && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "Updates", value: String(current.total_updates), color: "#f5e6d0" },
                  { label: "Global step", value: String(last?.global_step ?? "—"), color: "#f5e6d0" },
                  { label: "Mean ep. reward", value: typeof last?.mean_episode_reward === "number" ? last.mean_episode_reward.toFixed(2) : "—", color: "#d4af37" },
                  { label: "Attack success", value: typeof last?.attack_success_rate === "number" ? `${(last.attack_success_rate * 100).toFixed(0)}%` : "—", color: "#f07832" },
                ].map((s) => (
                  <div key={s.label} className="card-brutal p-4">
                    <div className="section-label mb-1">{s.label}</div>
                    <div className="mono-value" style={{ color: s.color }}>
                      {s.value}
                    </div>
                  </div>
                ))}
              </div>

              <div className="card-brutal-lg">
                <div className="section-label mb-3">
                  Mean episode reward per update — run {current.run_id} ({current.downsampled_to} of {current.total_updates} updates shown)
                </div>
                <MetricChart data={rewardSeries} accent="#d4af37" label="mean_episode_reward" />
              </div>

              <div className="card-brutal-lg">
                <div className="section-label mb-3">Attack success rate per update</div>
                <MetricChart data={successSeries} accent="#f07832" label="attack_success_rate" />
              </div>

              <p className="mono text-[11px] text-muted-foreground">
                Source: <span className="text-[#d4af37]">backend/runs/{current.run_id}/metrics/training.jsonl</span> — real PPO
                updates logged by the training loop. No fabricated curves.
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}
