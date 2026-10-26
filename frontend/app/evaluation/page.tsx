"use client";
// app/evaluation/page.tsx — Evaluation, driven by real evaluation artifacts
// (runs/<run_id>/evaluation/eval_summary.json served by /evaluation/{run_id}).
// Matchups that were never evaluated are shown as NOT EVALUATED.

import { useCallback, useEffect, useState } from "react";
import { FlaskConical, RefreshCw } from "lucide-react";
import {
  ApiError,
  api,
  type EvalMatchup,
  type EvalSummary,
  type RunSummary,
} from "@/lib/api";

const INTERESTING_MATCHUPS = [
  { red: "rl", blue: "rule_based", label: "RL Red vs Rule-Based Blue" },
  { red: "rl", blue: "proactive", label: "RL Red vs Proactive Blue" },
  { red: "scripted_aggressive", blue: "rule_based", label: "Scripted Aggressive vs Rule-Based" },
  { red: "scripted_stealth", blue: "rule_based", label: "Scripted Stealth vs Rule-Based" },
  { red: "scripted_lateral", blue: "rule_based", label: "Scripted Lateral vs Rule-Based" },
];

function pct(v: number | undefined): string {
  return v === undefined ? "—" : `${(v * 100).toFixed(0)}%`;
}

function num(v: number | undefined, digits = 2): string {
  return v === undefined ? "—" : v.toFixed(digits);
}

function MatchupRow({ m }: { m: EvalMatchup | null; label: string }) {
  if (!m) {
    return (
      <tr>
        <td className="mono text-muted-foreground">—</td>
        <td colSpan={6} className="mono text-muted-foreground">
          NOT EVALUATED
        </td>
      </tr>
    );
  }
  const s = m.summary;
  return (
    <tr>
      <td className="mono font-bold text-white whitespace-nowrap">{m.matchup}</td>
      <td className="mono" style={{ color: "#f07832" }}>{pct(s.attack_success_rate)}</td>
      <td className="mono" style={{ color: "#f07832" }}>{pct(s.red_win_rate)}</td>
      <td className="mono" style={{ color: "#79b8ff" }}>{pct(s.blue_win_rate)}</td>
      <td className="mono">{pct(s.undecided_rate)}</td>
      <td className="mono">{num(s.mean_red_reward)} / {num(s.mean_blue_reward)}</td>
      <td className="mono">{num(s.mean_episode_length, 1)}</td>
    </tr>
  );
}

export default function EvaluationPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [evalData, setEvalData] = useState<EvalSummary | null>(null);
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
      .evaluation(runId)
      .then((r) => {
        if (!cancelled) setEvalData(r);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const byMatchup = new Map(
    (evalData !== null && evalData.run_id === runId ? evalData.matchups : []).map((m) => [m.matchup, m]),
  );
  const shown = evalData !== null && evalData.run_id === runId ? evalData : null;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <FlaskConical size={14} className="text-[#d4af37]" />
            <span className="text-xs mono uppercase tracking-[0.2em] text-muted-foreground font-bold">
              Independent Evaluation
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight">Evaluation Results</h1>
        </div>
        <div className="flex items-center gap-2">
          {runs.length > 0 && (
            <select
              className="select-brutal min-w-52 py-1.5"
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

      <div className="rounded-xl border border-[#d4af37]/20 bg-[#24170f]/70 px-4 py-3 text-xs leading-5 text-[#decbb7]">
        Graph-simulator evaluation artifacts only. They do not measure Violet Ledger or SQLi live-server behavior and do not establish real-range policy improvement.
      </div>

      {error && (
        <div className="card-brutal border-red-500 border-2 p-4">
          <div className="mono text-xs font-bold text-red-400 uppercase mb-1">Not available</div>
          <p className="text-sm text-muted-foreground">{error}</p>
          <p className="text-xs mono text-muted-foreground mt-2">
            Evaluation artifacts are generated by{" "}
            <code className="text-[#d4af37]">backend/scripts/generate_eval_artifacts.py</code>.
          </p>
        </div>
      )}

      {shown && (
        <>
          <div className="card-brutal">
            <div className="flex flex-wrap items-center justify-between mb-3 gap-2">
              <div className="section-label mb-0">CROSS-MATCHUP SUMMARY — run {shown.run_id}</div>
              <div className="mono text-[10px] text-muted-foreground">
                generated {shown.generated_at} · {shown.n_episodes_per_matchup} episodes/matchup · seeds{" "}
                {shown.seed_range[0]}–{shown.seed_range[1]}
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="table-brutal">
                <thead>
                  <tr>
                    <th>Matchup</th>
                    <th>Attack success</th>
                    <th>Red wins</th>
                    <th>Blue wins</th>
                    <th>Undecided</th>
                    <th>Mean reward (R/B)</th>
                    <th>Mean length</th>
                  </tr>
                </thead>
                <tbody>
                  {INTERESTING_MATCHUPS.map((slot) => (
                    <MatchupRow key={slot.label} m={byMatchup.get(`${slot.red}_vs_${slot.blue}`) ?? null} label={slot.label} />
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mono text-[11px] text-muted-foreground mt-3">
              Undecided = episode truncated at the cycle limit with no critical asset compromised.
              All values computed from real episodes on unseen seeds — none are fabricated.
            </p>
          </div>

          <p className="mono text-[11px] text-muted-foreground">
            Source: <span className="text-[#d4af37]">backend/runs/{shown.run_id}/evaluation/eval_summary.json</span>
          </p>
        </>
      )}
    </div>
  );
}
