"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowUpRight,
  BookOpenCheck,
  Bug,
  CircleStop,
  Database,
  FlaskConical,
  RefreshCw,
  ShieldCheck,
  Swords,
} from "lucide-react";
import { rangeApi, type SqliTargetStatus, type TrainingRun } from "../../lib/range-api";

function scenarioRuns(items: TrainingRun[]) {
  return items.filter((run) => run.config.scenario_id === "sqli_lab_v1");
}

export default function SqliLabPage() {
  const router = useRouter();
  const [target, setTarget] = useState<SqliTargetStatus | null>(null);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [mode, setMode] = useState<"train" | "evaluate">("train");
  const [runKind, setRunKind] = useState<"teaching" | "research">("teaching");
  const [episodes, setEpisodes] = useState(1);
  const [maxSteps, setMaxSteps] = useState(6);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1_000_000));
  const [checkpointRunId, setCheckpointRunId] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [status, response] = await Promise.all([rangeApi.sqliTargetStatus(), rangeApi.trainingRuns()]);
      setTarget(status);
      setRuns(scenarioRuns(response.items));
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "SQLi lab service is unavailable");
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { void Promise.resolve().then(refresh); const timer = window.setInterval(() => void refresh(), 5000); return () => window.clearInterval(timer); }, [refresh]);

  const completedCheckpoints = useMemo(() => runs.filter((run) =>
    run.status === "completed" && run.mode === "train" && run.role === "joint" && Boolean(run.result?.checkpoint_sha256),
  ), [runs]);
  const activeRuns = runs.filter((run) => run.status === "queued" || run.status === "running");
  const minimumEpisodes = runKind === "research" ? (mode === "train" ? 10000 : 100) : 1;

  async function launch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (target?.status !== "available") return;
    setBusy(true); setError("");
    try {
      const run = await rangeApi.createTrainingRun({
        role: "joint",
        mode,
        run_kind: runKind,
        scenario_id: "sqli_lab_v1",
        episodes,
        max_steps: maxSteps,
        seed,
        opponent: "none",
        ...(mode === "evaluate" ? { checkpoint_run_id: checkpointRunId } : {}),
      });
      router.push(`/sqli/${run.training_run_id}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not start the SQLi lab run");
    } finally { setBusy(false); }
  }

  async function stop(runId: string) {
    try { await rangeApi.stopTrainingRun(runId); await refresh(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not request a stop"); }
  }

  return <div className="space-y-8">
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        <div className="eyebrow">Scenario sqli_lab_v1 · {target?.target_version ?? "target status checking"}</div>
        <h1 className="page-heading">SQL injection learning lab</h1>
        <p className="page-lede">Red and Blue use separate PPO policies against an isolated HTTP service backed by a read-only database of synthetic records. The target has no public port; all requests pass through the fixed SQLi executor.</p>
      </div>
      <button onClick={() => void refresh()} className="btn-brutal btn-brutal-ghost" aria-label="Refresh lab status"><RefreshCw size={14} />Refresh status</button>
    </header>
    {error && <div role="alert" className="rounded-lg border border-[#f07832]/30 bg-[#f07832]/10 px-4 py-3 text-xs text-[#ffd0af]">{error}</div>}

    <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <div className="panel p-4"><div className="mono-label">HTTP target</div><div className={`mt-2 text-sm font-semibold ${target?.status === "available" ? "text-[#9ac69c]" : "text-[#f1b38e]"}`}>{target?.status ?? (loading ? "checking" : "unavailable")}</div><div className="mt-1 text-[10px] text-[#8f7965]">Fixed sqli-target:8002 · private network</div></div>
      <div className="panel p-4"><div className="mono-label">Database role</div><div className="mt-2 text-sm font-semibold text-[#eadcc8]">SELECT only</div><div className="mt-1 text-[10px] text-[#8f7965]">Separate database · synthetic fixture set</div></div>
      <div className="panel p-4"><div className="mono-label">Active runs</div><div className="mt-2 font-mono text-2xl text-[#f1d995]">{activeRuns.length}</div><div className="mt-1 text-[10px] text-[#8f7965]">PPO training or evaluation jobs</div></div>
      <div className="panel p-4"><div className="mono-label">Saved episodes</div><div className="mt-2 font-mono text-2xl text-[#f1d995]">{runs.reduce((sum, run) => sum + Number(run.result?.completed_episodes ?? 0), 0)}</div><div className="mt-1 text-[10px] text-[#8f7965]">Persisted decisions and target evidence</div></div>
    </section>

    <section className="grid gap-4 xl:grid-cols-[1.2fr_.8fr]">
      <div className="space-y-4">
        <div className="grid gap-3 md:grid-cols-2">
          <article className="panel border-t-2 border-t-[#f07832]/70 p-5">
            <div className="flex items-center justify-between"><span className="eyebrow" style={{ color: "var(--orange)" }}>Red · request policy</span><Swords size={17} className="text-[#f07832]" /></div>
            <h2 className="mt-3 text-base font-semibold text-[#f0e2cf]">Compose from bounded parts</h2>
            <p className="mt-2 text-xs leading-6 text-[#a99079]">The PPO selects a field, base term, quote boundary, boolean connector, predicate, and comment boundary. The executor composes those components into a length-limited lookup parameter for the fixed endpoint.</p>
            <div className="mt-4 flex flex-wrap gap-2">{["field", "base term", "quote", "connector", "predicate", "comment"].map((item) => <span key={item} className="chip-mono">{item}</span>)}</div>
          </article>
          <article className="panel border-t-2 border-t-[#91bce2]/70 p-5">
            <div className="flex items-center justify-between"><span className="eyebrow" style={{ color: "var(--blue)" }}>Blue · defense policy</span><ShieldCheck size={17} className="text-[#91bce2]" /></div>
            <h2 className="mt-3 text-base font-semibold text-[#f0e2cf]">Act on sanitized telemetry</h2>
            <p className="mt-2 text-xs leading-6 text-[#a99079]">A separate PPO policy chooses query parameterization, a guest-scoped rate limit, temporary guest lookup isolation, or monitoring. Independent probes check whether the defense blocks unauthorized reads and serves a legitimate query.</p>
            <div className="mt-4 flex flex-wrap gap-2">{["parameterize", "rate limit", "isolate", "monitor"].map((item) => <span key={item} className="chip-mono">{item}</span>)}</div>
          </article>
        </div>
        <div className="rounded-xl border border-[#d4af37]/15 bg-[#130d09]/70 p-5">
          <div className="flex items-center gap-2"><Database size={15} className="text-[#d4af37]" /><span className="section-label">Learning contract</span></div>
          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            <div><h3 className="text-xs font-semibold text-[#eadcc8]">State</h3><p className="mt-1 text-[10px] leading-5 text-[#8f7965]">Normalized response history for Red; request counts, status classes, injection signals, and active controls for Blue.</p></div>
            <div><h3 className="text-xs font-semibold text-[#eadcc8]">Reward</h3><p className="mt-1 text-[10px] leading-5 text-[#8f7965]">Red is rewarded for a verified unauthorized read. Blue is rewarded for verified prevention and legitimate availability; false positives are penalized.</p></div>
            <div><h3 className="text-xs font-semibold text-[#eadcc8]">Update</h3><p className="mt-1 text-[10px] leading-5 text-[#8f7965]">Policies remain fixed through each episode. Accepted transitions update each role independently with clipped PPO and GAE.</p></div>
          </div>
        </div>
      </div>

      <section className="panel p-5 sm:p-6">
        <div className="eyebrow">Run control</div>
        <h2 className="mt-1 text-lg font-semibold text-[#f0e2cf]">Train or evaluate the PPO pair</h2>
        <p className="mt-2 text-xs leading-5 text-[#a99079]">Classroom training contains only the two learned policies. The named no-defense comparator is run only during evaluation.</p>
        <form onSubmit={launch} className="mt-5 space-y-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block text-[10px] text-[#9d846c]">Mode
              <select className="select-brutal mt-1.5" value={mode} onChange={(event) => { const next = event.target.value as "train" | "evaluate"; setMode(next); setEpisodes(runKind === "research" ? (next === "train" ? 10000 : 100) : 1); }}>
                <option value="train">Train Red + Blue</option><option value="evaluate">Evaluate frozen checkpoint</option>
              </select>
            </label>
            <label className="block text-[10px] text-[#9d846c]">Run scale
              <select className="select-brutal mt-1.5" value={runKind} onChange={(event) => { const next = event.target.value as "teaching" | "research"; setRunKind(next); setEpisodes(next === "research" ? (mode === "train" ? 10000 : 100) : 1); }}>
                <option value="teaching">Teaching walkthrough</option><option value="research">Research campaign</option>
              </select>
            </label>
            <label className="block text-[10px] text-[#9d846c]">Episodes <span className="float-right text-[#e7d5c0]">min {minimumEpisodes.toLocaleString()}</span>
              <input className="input-brutal mt-1.5" type="number" min={minimumEpisodes} max={10000} value={episodes} onChange={(event) => setEpisodes(Math.max(minimumEpisodes, Math.min(10000, Number(event.target.value) || minimumEpisodes)))} />
            </label>
            <label className="block text-[10px] text-[#9d846c]">Decisions per episode
              <select className="select-brutal mt-1.5" value={maxSteps} onChange={(event) => setMaxSteps(Number(event.target.value))}><option value={4}>4 decisions</option><option value={6}>6 decisions</option><option value={8}>8 decisions</option><option value={12}>12 decisions</option></select>
            </label>
            <label className="block text-[10px] text-[#9d846c] sm:col-span-2">Seed
              <div className="mt-1.5 flex gap-2"><input className="input-brutal font-mono" type="number" min={0} max={2147483647} value={seed} onChange={(event) => setSeed(Math.max(0, Number(event.target.value) || 0))} /><button type="button" className="btn-brutal btn-brutal-ghost shrink-0" onClick={() => setSeed(Math.floor(Math.random() * 1_000_000))}>New seed</button></div>
            </label>
          </div>
          {mode === "evaluate" && <label className="block text-[10px] text-[#9d846c]">Completed SQLi joint checkpoint
            <select className="select-brutal mt-1.5" value={checkpointRunId} onChange={(event) => setCheckpointRunId(event.target.value)} required>
              <option value="">Choose a compatible checkpoint</option>
              {completedCheckpoints.map((run) => <option key={run.training_run_id} value={run.training_run_id}>{run.training_run_id.slice(0, 8)} · {String(run.result?.completed_episodes ?? "")} episodes</option>)}
            </select>
          </label>}
          <button disabled={busy || target?.status !== "available" || (mode === "evaluate" && !checkpointRunId)} className="btn-brutal btn-brutal-yellow w-full"><BookOpenCheck size={14} />{busy ? "Submitting…" : mode === "train" ? "Start PPO training" : "Start held-out evaluation"}</button>
          <p className="text-[9px] leading-5 text-[#8f7965]">Research gates require 10,000 completed training episodes per seed, at least 100 paired held-out episodes, five independent training seeds, a positive paired interval, and full legitimate-query availability. This worktree records one training seed, so it cannot claim a supported improvement by itself.</p>
        </form>
        {target?.status !== "available" && <div className="mt-3 rounded-lg border border-[#f07832]/20 bg-[#f07832]/[.07] p-3 text-[10px] text-[#e6b497]">The private executor has not reported a healthy SQLi target. Run training becomes available when its HTTP health check succeeds.</div>}
      </section>
    </section>

    <section className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3"><div><div className="eyebrow">Persistent run records</div><h2 className="mt-1 text-xl font-semibold text-[#f0e2cf]">Recent SQLi runs</h2></div><span className="badge-brutal">Refresh-safe replay</span></div>
      {runs.length ? <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">{runs.slice(0, 12).map((run) => {
        const active = run.status === "queued" || run.status === "running";
        return <article key={run.training_run_id} className="panel flex min-w-0 flex-col gap-3 p-4">
          <div className="flex items-center justify-between gap-2"><span className={`badge-brutal ${run.mode === "train" ? "badge-yellow" : "badge-blue"}`}>{run.mode === "train" ? "PPO training" : "Evaluation"}</span><span className="font-mono text-[9px] text-[#8f7965]">{new Date(run.created_at).toLocaleString()}</span></div>
          <div className="font-mono text-xs text-[#eadcc8]">{run.training_run_id}</div>
          <div className="text-[10px] text-[#a99079]">{run.status} · {String(run.config.run_kind ?? "teaching")} · seed {String(run.config.seed ?? "—")} · {String(run.result?.completed_episodes ?? 0)} completed</div>
          <div className="mt-auto flex items-center justify-between border-t border-[#d4af37]/10 pt-3">
            <Link href={`/sqli/${run.training_run_id}`} className="inline-flex items-center gap-1.5 text-[10px] font-semibold text-[#f1d995] hover:text-white">Open guided replay<ArrowUpRight size={12} /></Link>
            {active && <button onClick={() => void stop(run.training_run_id)} className="btn-brutal btn-brutal-ghost min-h-8 py-1 text-[9px]"><CircleStop size={12} />Stop</button>}
          </div>
        </article>;
      })}</div> : <div className="panel flex min-h-36 flex-col items-center justify-center text-center"><FlaskConical size={20} className="text-[#8f7965]" /><p className="mt-3 text-sm font-medium text-[#decbb7]">No SQLi lab runs yet</p><p className="mt-1 text-xs text-[#8f7965]">Start a teaching episode to create a persisted decision and evidence timeline.</p></div>}
    </section>

    <aside className="rounded-xl border border-[#d4af37]/15 bg-[#130d09]/65 p-4 text-[10px] leading-5 text-[#8f7965]"><div className="mb-1 flex items-center gap-2 text-[#dfc17a]"><Bug size={13} />Live target boundary</div>Each run is pinned to <code className="text-[#eadcc8]">sqli_lab_v1</code>, target version <code className="text-[#eadcc8]">sqli_target_v1</code>, and versioned observation, action, and reward contracts. The SQLi executor has only the private target network; it has no database network or public egress.</aside>
  </div>;
}
