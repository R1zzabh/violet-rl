"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, BookOpenCheck, RefreshCw } from "lucide-react";
import { rangeApi, type LearningReport, type TrainingRun } from "../../lib/range-api";

const STORAGE_KEY = "violet.learning-report";
const object = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const numeric = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? value : null;
const percent = (value: unknown) => numeric(value) === null ? "—" : `${(numeric(value)! * 100).toFixed(1)}%`;
const shortId = (value: string) => `${value.slice(0, 8)}…${value.slice(-4)}`;

function Metric({ label, value, foot, asPercent = true }: { label: string; value: unknown; foot?: string; asPercent?: boolean }) {
  const shown = asPercent ? percent(value) : numeric(value) === null ? "—" : String(value);
  return <div className="research-metric"><span>{label}</span><strong>{shown}</strong>{foot && <small>{foot}</small>}</div>;
}

function Summary({ title, record }: { title: string; record: Record<string, unknown> | null }) {
  if (!record) return <section className="panel research-summary"><div className="panel-head"><h2 className="panel-title">{title}</h2><span className="pill pill-amber">waiting</span></div><p className="research-empty">This checkpoint evaluation has not recorded a result yet.</p></section>;
  return <section className="panel research-summary">
    <div className="panel-head"><div><span className="panel-kicker">Frozen checkpoint evaluation</span><h2 className="panel-title">{title}</h2></div><span className={`pill ${record.status === "completed" ? "pill-green" : "pill-amber"}`}>{String(record.status ?? "pending")}</span></div>
    <div className="research-metrics">
      <Metric label="Red verified success" value={record.verified_red_success_rate} />
      <Metric label="Blue verified prevention" value={record.verified_blue_prevention_rate} />
      <Metric label="Legitimate availability" value={record.legitimate_service_availability} />
      <Metric label="Completed matches" value={record.completed_matches} asPercent={false} foot={`${String(record.incomplete_matches ?? 0)} incomplete`} />
    </div>
    <div className="research-record-meta"><span>Policy versions <code>{JSON.stringify(record.policy_versions ?? {})}</code></span><span>Checkpoint <code>{String(record.checkpoint_sha256 ?? "pending")}</code></span></div>
    {Array.isArray(record.red_success_wilson_95) && <p className="research-footnote">Red success Wilson 95% interval: {percent(record.red_success_wilson_95[0])}–{percent(record.red_success_wilson_95[1])}.</p>}
  </section>;
}

export default function CheckpointEvaluation() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [sourceId, setSourceId] = useState("");
  const [evaluationId, setEvaluationId] = useState("");
  const [matches, setMatches] = useState(20);
  const [report, setReport] = useState<LearningReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const eligibleRuns = useMemo(() => runs.filter((run) => run.role === "joint" && run.mode === "train"
    && run.config.scenario_id === "sqli_lab_v1" && ["completed", "stopped"].includes(run.status)
    && typeof run.result?.initial_checkpoint_sha256 === "string"), [runs]);

  const refreshReport = useCallback(async (id: string) => {
    const next = await rangeApi.learningReport(id);
    setReport(next);
    return next;
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [page] = await Promise.all([rangeApi.trainingRuns(100)]);
        if (cancelled) return;
        setRuns(page.items);
        const savedId = window.localStorage.getItem(STORAGE_KEY) ?? "";
        const savedSource = window.localStorage.getItem(`${STORAGE_KEY}.source`) ?? "";
        const candidates = page.items.filter((run) => run.role === "joint" && run.mode === "train"
          && run.config.scenario_id === "sqli_lab_v1" && ["completed", "stopped"].includes(run.status)
          && typeof run.result?.initial_checkpoint_sha256 === "string");
        const source = candidates.find((run) => run.training_run_id === savedSource)?.training_run_id
          ?? candidates[0]?.training_run_id ?? "";
        setSourceId(source);
        const savedReport = page.items.find((run) => run.config.learning_evaluation_id === savedId);
        const recentReport = savedReport ?? page.items.find((run) => run.config.checkpoint_run_id === source
          && typeof run.config.learning_evaluation_id === "string");
        const id = typeof recentReport?.config.learning_evaluation_id === "string"
          ? recentReport.config.learning_evaluation_id : savedId;
        setEvaluationId(id);
        if (id) {
          try {
            const found = await refreshReport(id);
            if (!cancelled) {
              window.localStorage.setItem(STORAGE_KEY, id);
              setSourceId(found.source_training_run_id);
            }
          } catch (reason) {
            if (!cancelled) setError(reason instanceof Error ? reason.message : "Saved evaluation could not be loaded.");
          }
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
    if (!evaluationId || report?.status !== "running") return;
    const timer = window.setInterval(() => {
      void refreshReport(evaluationId).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Report refresh failed."));
    }, 3000);
    return () => window.clearInterval(timer);
  }, [evaluationId, report?.status, refreshReport]);

  async function start() {
    if (!sourceId) return;
    setBusy(true);
    setError("");
    try {
      const id = crypto.randomUUID();
      window.localStorage.setItem(STORAGE_KEY, id);
      window.localStorage.setItem(`${STORAGE_KEY}.source`, sourceId);
      setEvaluationId(id);
      setReport(await rangeApi.startLearningReport({ training_run_id: sourceId, evaluation_id: id, matches }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Matched evaluation could not be started.");
    } finally {
      setBusy(false);
    }
  }

  async function refresh() {
    if (!evaluationId) return;
    setBusy(true);
    try { setReport(await refreshReport(evaluationId)); setError(""); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Report refresh failed."); }
    finally { setBusy(false); }
  }

  const changes = report?.paired_changes ?? {};
  const labels: Record<string, string> = {
    verified_red_success_rate: "Red verified success",
    verified_blue_prevention_rate: "Blue prevention",
    legitimate_service_availability: "Legitimate availability",
  };
  const family = object(report?.family_results);

  return <div className="research-page">
    <header className="page-header">
      <div><p className="eyebrow">Matched held-out fixtures / frozen checkpoints</p><h2 className="panel-title">Measure performance on unseen fixtures</h2>
        <p className="page-deck">Compare the initial and later Red–Blue checkpoints from one training run on the same seeded server fixtures. Changes are reported from verified outcomes, with paired uncertainty.</p></div>
      <div className="header-actions"><button className="button button-secondary" onClick={() => void refresh()} disabled={busy || !evaluationId}><RefreshCw size={14} />Refresh report</button></div>
    </header>

    {error && <div className="error-banner"><AlertTriangle size={16} /><span>{error}</span></div>}
    <section className="panel research-launch">
      <div><span className="panel-kicker">Evaluation setup</span><h2>Choose a completed paired training run</h2><p>Both policy checkpoints are frozen. Evaluation runs the same seeded test schedule at the fixed 0.5 vulnerable-configuration rate and creates no policy updates.</p></div>
      <div className="research-launch-controls">
        <label>Training checkpoint pair<select value={sourceId} onChange={(event) => setSourceId(event.target.value)} disabled={loading || busy}>
          {!eligibleRuns.length && <option value="">{loading ? "Loading runs…" : "No eligible SQLi training runs"}</option>}
          {eligibleRuns.map((run) => <option key={run.training_run_id} value={run.training_run_id}>{new Date(run.created_at).toISOString().slice(0, 16).replace("T", " ")} UTC · {run.status} · {shortId(run.training_run_id)}</option>)}
        </select></label>
        <label>Matched fixtures<select value={matches} onChange={(event) => setMatches(Number(event.target.value))} disabled={busy}>
          {[10, 20, 30, 50, 100].map((value) => <option key={value} value={value}>{value} fixtures per checkpoint</option>)}
        </select></label>
        <button className="button" onClick={() => void start()} disabled={busy || !sourceId || report?.status === "running" && report.source_training_run_id === sourceId}><BookOpenCheck size={15} />{busy ? "Starting…" : report?.status === "running" && report.source_training_run_id === sourceId ? "Evaluation already queued or running" : "Evaluate initial and later"}</button>
      </div>
    </section>

    {report && <>
      {(report.initial?.status === "queued" || report.later?.status === "queued") && <div className="notice">Evaluation is queued for the shared worker. The active training run continues; the comparison starts when the worker is available.</div>}
      <div className="research-report-line"><span className="pill pill-blue"><Activity size={12} />{report.status.replaceAll("_", " ")}</span><span>Evaluation <code>{report.evaluation_id}</code></span><span>Source <code>{report.source_training_run_id}</code></span><span>{report.matched_seed_schedule.length} shared seeds</span></div>
      <div className="research-summary-grid"><Summary title="Initial checkpoint" record={report.initial} /><Summary title="Later checkpoint" record={report.later} /></div>
      <section className="panel research-change-panel"><div className="panel-head"><div><span className="panel-kicker">Paired held-out outcomes</span><h2 className="panel-title">Later minus initial</h2></div><span className="pill">{report.status === "completed" ? "measured" : "in progress"}</span></div>
        <div className="research-change-list">{Object.entries(changes).map(([key, raw]) => {
          const change = object(raw);
          const delta = numeric(change.mean_change);
          const interval = Array.isArray(change.bootstrap_ci_95) ? change.bootstrap_ci_95 : null;
          const Icon = delta === null || delta === 0 ? Activity : delta > 0 ? ArrowUpRight : ArrowDownRight;
          return <div className="research-change-row" key={key}><span>{labels[key] ?? key}</span><strong><Icon size={15} />{delta === null ? "—" : `${delta > 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp`}</strong><small>{interval ? `paired bootstrap 95% CI ${percent(interval[0])} to ${percent(interval[1])}` : "uncertainty unavailable"} · {String(change.matched_fixtures ?? 0)} fixtures</small></div>;
        })}</div>
        <p className="research-footnote">{report.claim_basis}</p>
      </section>
      <section className="panel research-family-panel"><div className="panel-head"><div><span className="panel-kicker">Coverage by vulnerability family</span><h2 className="panel-title">Evaluation scope</h2></div></div><div className="research-family-grid">
        <div><span className="pill pill-green">{String(object(family.sql_injection).status ?? report.status)}</span><strong>SQL injection</strong><small>Evaluated against the local SQLi target.</small></div>
        <div><span className="pill pill-amber">not evaluated</span><strong>Unauthorized record access</strong><small>The separate Violet Ledger target is not included in this matched report.</small></div>
        <div><span className="pill pill-amber">unavailable</span><strong>Permission or workflow abuse</strong><small>This family is not implemented in the current target.</small></div>
      </div></section>
    </>}
    {!report && <div className="notice research-notice">Training loss, optimizer statistics, and action-probability changes explain updates; this report uses held-out verified outcomes to measure policy performance.</div>}
  </div>;
}
