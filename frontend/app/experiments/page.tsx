"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, ArrowDown, ArrowUp, FlaskConical, RefreshCw, Scale } from "lucide-react";
import { rangeApi, type ExposureExperiment, type TrainingRun } from "../../lib/range-api";

const STORAGE_KEY = "violet.exposure-experiment";
const object = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const numeric = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? value : null;
const percent = (value: unknown) => numeric(value) === null ? "—" : `${(numeric(value)! * 100).toFixed(1)}%`;
const shortId = (value: string) => `${value.slice(0, 8)}…${value.slice(-4)}`;

function BranchCard({ title, record, tone }: { title: string; record: Record<string, unknown> | null; tone: "neutral" | "accent" }) {
  if (!record) return <section className={`panel experiment-branch ${tone}`}><span className="panel-kicker">Training branch</span><h2>{title}</h2><p>Waiting for branch record…</p></section>;
  return <section className={`panel experiment-branch ${tone}`}>
    <div className="experiment-branch-title"><div><span className="panel-kicker">Training branch</span><h2>{title}</h2></div><span className={`pill ${record.status === "completed" ? "pill-green" : "pill-amber"}`}>{String(record.status ?? "pending")}</span></div>
    <div className="experiment-branch-metrics">
      <div><span>Configured rate</span><strong>{percent(record.configured_vulnerability_probability)}</strong></div>
      <div><span>Realized vulnerable fixtures</span><strong>{percent(record.realized_vulnerable_configuration_frequency)}</strong><small>{String(record.episodes_with_target_configuration ?? 0)} episodes observed</small></div>
      <div><span>Red verified success</span><strong>{percent(record.verified_red_success_rate)}</strong></div>
      <div><span>Blue prevention</span><strong>{percent(record.verified_blue_prevention_rate)}</strong></div>
      <div><span>Legitimate availability</span><strong>{percent(record.legitimate_service_availability)}</strong></div>
      <div><span>Completed PPO updates</span><strong>{String(record.completed_updates ?? 0)} / {String(record.requested_updates ?? 0)}</strong></div>
    </div>
    <p className="experiment-checkpoint">Final checkpoint <code>{String(record.checkpoint_sha256 ?? "pending")}</code></p>
  </section>;
}

export default function ExperimentsPage() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [sourceId, setSourceId] = useState("");
  const [experimentId, setExperimentId] = useState("");
  const [episodes, setEpisodes] = useState(30);
  const [controlProbability, setControlProbability] = useState(0.35);
  const [increasedProbability, setIncreasedProbability] = useState(0.85);
  const [evaluationMatches, setEvaluationMatches] = useState(20);
  const [report, setReport] = useState<ExposureExperiment | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const eligibleRuns = useMemo(() => runs.filter((run) => run.role === "joint" && run.mode === "train"
    && run.config.scenario_id === "sqli_lab_v1" && ["completed", "stopped"].includes(run.status)
    && typeof run.result?.checkpoint_sha256 === "string"), [runs]);

  const refreshReport = useCallback(async (id: string) => {
    const next = await rangeApi.exposureExperiment(id);
    setReport(next);
    return next;
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const page = await rangeApi.trainingRuns(100);
        if (cancelled) return;
        setRuns(page.items);
        const savedId = window.localStorage.getItem(STORAGE_KEY) ?? "";
        const savedSource = window.localStorage.getItem(`${STORAGE_KEY}.source`) ?? "";
        setExperimentId(savedId);
        setSourceId(savedSource || page.items.find((run) => run.role === "joint" && run.mode === "train"
          && run.config.scenario_id === "sqli_lab_v1" && ["completed", "stopped"].includes(run.status)
          && typeof run.result?.checkpoint_sha256 === "string")?.training_run_id || "");
        if (savedId) {
          try { await refreshReport(savedId); }
          catch { if (!cancelled) window.localStorage.removeItem(STORAGE_KEY); }
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Training runs could not be loaded.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => { cancelled = true; };
  }, [refreshReport]);

  useEffect(() => {
    if (!experimentId || !report || !["training", "evaluation"].includes(report.status)) return;
    const timer = window.setInterval(() => {
      void refreshReport(experimentId).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Experiment refresh failed."));
    }, 3000);
    return () => window.clearInterval(timer);
  }, [experimentId, report, refreshReport]);

  async function start() {
    if (!sourceId) return;
    setBusy(true);
    setError("");
    try {
      const id = crypto.randomUUID();
      window.localStorage.setItem(STORAGE_KEY, id);
      window.localStorage.setItem(`${STORAGE_KEY}.source`, sourceId);
      setExperimentId(id);
      setReport(await rangeApi.startExposureExperiment({
        source_training_run_id: sourceId,
        experiment_id: id,
        episodes_per_branch: episodes,
        seed: 1701,
        control_vulnerability_probability: controlProbability,
        increased_vulnerability_probability: increasedProbability,
        evaluation_matches: evaluationMatches,
      }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Exposure experiment could not be started.");
      window.localStorage.removeItem(STORAGE_KEY);
    } finally {
      setBusy(false);
    }
  }

  async function evaluate() {
    if (!experimentId) return;
    setBusy(true);
    setError("");
    try { setReport(await rangeApi.evaluateExposureExperiment(experimentId, evaluationMatches)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Matched evaluation could not start."); }
    finally { setBusy(false); }
  }

  async function refresh() {
    if (!experimentId) return;
    setBusy(true);
    try { setReport(await refreshReport(experimentId)); setError(""); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Experiment refresh failed."); }
    finally { setBusy(false); }
  }

  const training = report?.training ?? {};
  const heldOut = report?.held_out_evaluation;
  const evaluation = heldOut?.branches ?? {};
  const evaluated = report?.status === "completed";

  return <div className="research-page">
    <header className="page-header"><div><p className="eyebrow">Matched branches / fixed held-out distribution</p><h1 className="page-title">Exposure experiment</h1>
      <p className="page-deck">Branch one checkpoint pair into equal-budget training runs. Change only how often the SQL injection vulnerable server configuration appears, then compare both learned pairs on the same unchanged test fixtures.</p></div>
      <div className="header-actions"><button className="button button-secondary" onClick={() => void refresh()} disabled={busy || !experimentId}><RefreshCw size={14} />Refresh</button></div>
    </header>

    {error && <div className="error-banner"><AlertTriangle size={16} /><span>{error}</span></div>}
    <section className="panel research-launch">
      <div><span className="panel-kicker">Experiment design</span><h2>Increase vulnerable configuration exposure</h2><p>Red and Blue keep full control of their policy actions. No requests or decisions are prescribed. Per-episode configuration seeds are matched between branches.</p></div>
      <div className="research-launch-controls experiment-controls">
        <label>Source policy pair<select value={sourceId} onChange={(event) => setSourceId(event.target.value)} disabled={loading || busy}>
          {!eligibleRuns.length && <option value="">{loading ? "Loading runs…" : "No eligible SQLi checkpoints"}</option>}
          {eligibleRuns.map((run) => <option key={run.training_run_id} value={run.training_run_id}>{new Date(run.created_at).toISOString().slice(0, 16).replace("T", " ")} UTC · {shortId(run.training_run_id)}</option>)}
        </select></label>
        <label>Training episodes per branch<select value={episodes} onChange={(event) => setEpisodes(Number(event.target.value))} disabled={busy}>{[10, 30, 50, 100, 250, 500].map((value) => <option key={value} value={value}>{value} paired updates</option>)}</select></label>
        <label>Control exposure<select value={controlProbability} onChange={(event) => setControlProbability(Number(event.target.value))} disabled={busy}>{[0.2, 0.35, 0.5, 0.65].map((value) => <option key={value} value={value}>{percent(value)} vulnerable fixtures</option>)}</select></label>
        <label>Increased exposure<select value={increasedProbability} onChange={(event) => setIncreasedProbability(Number(event.target.value))} disabled={busy}>{[0.65, 0.75, 0.85, 0.95, 1].map((value) => <option key={value} value={value}>{percent(value)} vulnerable fixtures</option>)}</select></label>
        <label>Held-out matches<select value={evaluationMatches} onChange={(event) => setEvaluationMatches(Number(event.target.value))} disabled={busy}>{[10, 20, 30, 50, 100].map((value) => <option key={value} value={value}>{value} matches per branch</option>)}</select></label>
        <button className="button" onClick={() => void start()} disabled={busy || !sourceId || increasedProbability <= controlProbability}><FlaskConical size={15} />{busy ? "Starting…" : "Start equal-budget branches"}</button>
      </div>
    </section>

    {report && <>
      <div className="research-report-line"><span className={`pill ${evaluated ? "pill-green" : "pill-blue"}`}><Scale size={12} />{report.status.replaceAll("_", " ")}</span><span>Experiment <code>{report.experiment_id}</code></span><span>Equal budget <strong>{report.equal_training_budget ? "confirmed" : "pending"}</strong></span><span>{report.matched_training_seeds.length} matched training seeds</span></div>
      <div className="experiment-branches"><BranchCard title="Control exposure" record={training.control ?? null} tone="neutral" /><BranchCard title="Increased exposure" record={training.increased ?? null} tone="accent" /></div>
      {report.status === "ready_for_evaluation" && <div className="experiment-evaluate panel"><div><span className="panel-kicker">Training branches complete</span><strong>Freeze both latest checkpoints and run matched held-out evaluation.</strong></div><button className="button" onClick={() => void evaluate()} disabled={busy}><Scale size={15} />Evaluate both branches</button></div>}
      {heldOut && <section className="panel research-change-panel"><div className="panel-head"><div><span className="panel-kicker">Unchanged test distribution · 0.5 vulnerability probability</span><h2 className="panel-title">Held-out outcomes</h2></div><span className={`pill ${heldOut.same_seed_schedule ? "pill-green" : "pill-amber"}`}>{heldOut.same_seed_schedule ? "matched seed schedule" : "evaluation pending"}</span></div>
        <div className="experiment-eval-grid">{(["control", "increased"] as const).map((branch) => {
          const row = object(evaluation[branch]);
          return <div className="experiment-eval-card" key={branch}><span>{branch === "control" ? <ArrowDown size={14} /> : <ArrowUp size={14} />}{branch} branch</span><strong>{String(row.status ?? "pending")}</strong><div><span>Red success</span><b>{percent(row.verified_exposure_rate)}</b></div><div><span>Blue prevention</span><b>{percent(row.verified_prevention_rate)}</b></div><div><span>Legitimate availability</span><b>{percent(row.legitimate_service_availability)}</b></div><div><span>Verified disclosures</span><b>{String(row.verified_disclosures ?? "—")}</b></div></div>;
        })}</div>
      </section>}
      <div className="notice research-notice">{report.interpretation} Realized exposure frequency is reported from the seeded server configurations recorded at episode start. These results describe the runs and fixture budget shown here; they do not establish performance across independent research seeds.</div>
    </>}
    {!report && <div className="notice research-notice">This is a SQL injection exposure experiment. Unauthorized record access is not evaluated here, and permission or workflow abuse is not available in the current target.</div>}
  </div>;
}
