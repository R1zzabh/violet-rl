"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Bug, CircleStop, Clock3, Play, RefreshCw, Save, ShieldCheck } from "lucide-react";
import {
  rangeApi,
  type SqliPilotStatus,
  type SqliTargetStatus,
  type TrainingRun,
  type TrainingUpdateRecord,
} from "../lib/range-api";

const STORAGE_KEY = "violet.sqli.balanced-pilot";
const PRESET_RUN_KEY = "violet.sqli.training-preset-run";
const pct = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;
const time = (value: string | null) => {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "—" : `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`;
};
const interval = (range: [number, number] | null) =>
  range ? `${pct(range[0])}–${pct(range[1])}` : "unavailable";

function stateLabel(status: SqliPilotStatus["status"] | undefined): string {
  switch (status) {
    case "training": return "Training in progress";
    case "ready_for_evaluation": return "Training complete · evaluation starting";
    case "evaluation": return "Held-out evaluation in progress";
    case "completed": return "Pilot complete";
    case "incomplete": return "Pilot incomplete";
    default: return "No pilot started";
  }
}

function TrainingPresets({ targetAvailable }: { targetAvailable: boolean }) {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [run, setRun] = useState<TrainingRun | null>(null);
  const [updates, setUpdates] = useState<TrainingUpdateRecord[]>([]);
  const [sourceId, setSourceId] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const eligibleSources = runs.filter((item) => item.role === "joint" && item.mode === "train"
    && item.config.scenario_id === "sqli_lab_v1" && ["completed", "stopped"].includes(item.status)
    && typeof item.result?.checkpoint_sha256 === "string");
  const active = Boolean(run && ["queued", "running"].includes(run.status));
  const updateIndexes = new Set(updates.filter((item) => item.role === "red").map((item) => item.update_index));
  const completed = updateIndexes.size;
  const requested = Number(run?.config.episodes ?? 0);
  const progress = requested ? Math.min(100, completed * 100 / requested) : 0;

  const refreshRun = useCallback(async (id: string) => {
    const [nextRun, nextUpdates] = await Promise.all([rangeApi.trainingRun(id), rangeApi.trainingUpdates(id)]);
    setRun(nextRun);
    setUpdates(nextUpdates.items);
    return nextRun;
  }, []);

  const refreshSources = useCallback(async () => {
    const page = await rangeApi.trainingRuns(100);
    setRuns(page.items);
    return page.items;
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const page = await rangeApi.trainingRuns(100);
        if (cancelled) return;
        setRuns(page.items);
        const savedId = window.localStorage.getItem(PRESET_RUN_KEY);
        if (savedId) {
          try { await refreshRun(savedId); }
          catch { if (!cancelled) window.localStorage.removeItem(PRESET_RUN_KEY); }
        }
        const lastCheckpoint = page.items.find((item) => item.role === "joint" && item.mode === "train"
          && item.config.scenario_id === "sqli_lab_v1" && ["completed", "stopped"].includes(item.status)
          && typeof item.result?.checkpoint_sha256 === "string");
        if (!cancelled && lastCheckpoint) setSourceId(lastCheckpoint.training_run_id);
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Training run history could not be loaded.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => { cancelled = true; };
  }, [refreshRun]);

  useEffect(() => {
    if (!run?.training_run_id || !["queued", "running"].includes(run.status)) return;
    const timer = window.setInterval(() => {
      void refreshRun(run.training_run_id).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Training status refresh failed."));
    }, 3000);
    return () => window.clearInterval(timer);
  }, [run?.training_run_id, run?.status, refreshRun]);

  async function start(episodes: number, runKind: "teaching" | "research", checkpointRunId?: string) {
    setBusy(true);
    setError("");
    try {
      const currentTarget = await rangeApi.sqliTargetStatus();
      if (currentTarget.status !== "available") throw new Error("The isolated SQLi target is unavailable.");
      const created = await rangeApi.createTrainingRun({
        role: "joint",
        mode: "train",
        run_kind: runKind,
        episodes,
        max_steps: 32,
        seed: 17,
        opponent: "none",
        scenario_id: "sqli_lab_v1",
        ...(checkpointRunId ? { checkpoint_run_id: checkpointRunId } : {}),
      });
      window.localStorage.setItem(PRESET_RUN_KEY, created.training_run_id);
      setRun(created);
      setUpdates([]);
      await refreshSources();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Training run could not be queued.");
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    if (!run) return;
    setBusy(true);
    try { setRun(await rangeApi.stopTrainingRun(run.training_run_id)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Stop request failed."); }
    finally { setBusy(false); }
  }

  async function refresh() {
    setBusy(true);
    try {
      if (run) await refreshRun(run.training_run_id);
      await refreshSources();
      setError("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Training state could not be refreshed."); }
    finally { setBusy(false); setLoading(false); }
  }

  return <section className="panel training-presets">
    <div className="training-presets-head"><div><span className="panel-kicker">CPU training presets</span><h2>Demonstrate or continue learning</h2><p>Each episode runs on the local HTTP target. The joint pair freezes during an episode, then both policies update from eligible saved transitions.</p></div>
      <button className="button button-secondary button-small" onClick={() => void refresh()} disabled={busy}><RefreshCw size={13} />Refresh</button></div>
    <div className="training-presets-actions">
      <button className="button button-secondary" onClick={() => void start(5, "teaching")} disabled={busy || active || !targetAvailable}><Play size={14} />Start 5-update demonstration</button>
      <button className="button" onClick={() => void start(10000, "research")} disabled={busy || active || !targetAvailable}><Clock3 size={14} />Start 10,000-update long run</button>
      {active && <button className="button button-secondary" onClick={() => void stop()} disabled={busy}><CircleStop size={14} />Stop after current episode</button>}
    </div>
    {error && <div className="error-banner" role="alert"><AlertTriangle size={14} /><span>{error}</span></div>}
    {run && <div className="training-presets-current">
      <div className="training-presets-runline"><span className={`pill ${active ? "pill-blue" : run.status === "completed" ? "pill-green" : "pill-amber"}`}>{run.status}</span><code>{run.training_run_id}</code><span>{String(run.config.run_kind ?? "teaching")} · {String(run.config.episodes ?? 0)} updates · 32 rounds maximum</span></div>
      <div className="progress-track" role="progressbar" aria-label="Paired PPO updates" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100}><div className="progress-fill" style={{ width: `${progress}%` }} /></div>
      <div className="training-presets-progress"><strong>{completed.toLocaleString()} / {requested.toLocaleString()}</strong><span>paired optimizer updates recorded</span><span>{updates.at(-1)?.checkpoint_sha256 ? `latest checkpoint ${updates.at(-1)?.checkpoint_sha256?.slice(0, 12)}` : "checkpoint digest pending"}</span></div>
    </div>}
    <div className="training-resume-row">
      <label>Resume from a saved joint checkpoint<select value={sourceId} onChange={(event) => setSourceId(event.target.value)} disabled={loading || busy || active}>
        {!eligibleSources.length && <option value="">{loading ? "Loading checkpoint history…" : "No completed checkpoint yet"}</option>}
        {eligibleSources.map((item) => <option key={item.training_run_id} value={item.training_run_id}>{item.status} · {String(item.config.episodes ?? 0)} updates · {item.training_run_id.slice(0, 8)}</option>)}
      </select></label>
      <button className="button button-secondary button-small" onClick={() => void start(10000, "research", sourceId)} disabled={busy || active || !sourceId || !targetAvailable}><Save size={13} />Resume long run from checkpoint</button>
    </div>
    <p className="training-presets-note">The long preset is a single-seed 10,000-update run. It checkpoints after each completed episode and can continue from the last saved pair after a stop. Multi-seed held-out evaluation is needed for claims about general improvement.</p>
  </section>;
}

export default function TrainingMonitor() {
  const [target, setTarget] = useState<SqliTargetStatus | null>(null);
  const [campaign, setCampaign] = useState<SqliPilotStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const evaluationStarts = useRef(new Set<string>());

  const refreshStatus = useCallback(async (id: string) => {
    const current = await rangeApi.pilotStatus(id);
    setCampaign(current);
    if (current.status === "ready_for_evaluation" && !evaluationStarts.current.has(id)) {
      evaluationStarts.current.add(id);
      try {
        setCampaign(await rangeApi.startPilotEvaluation(id));
      } catch (reason) {
        evaluationStarts.current.delete(id);
        throw reason;
      }
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const targetState = await rangeApi.sqliTargetStatus();
        if (!cancelled) setTarget(targetState);
        const id = window.localStorage.getItem(STORAGE_KEY);
        if (id) {
          const current = await rangeApi.pilotStatus(id);
          if (!cancelled) setCampaign(current);
          if (current.status === "ready_for_evaluation") {
            evaluationStarts.current.add(id);
            const updated = await rangeApi.startPilotEvaluation(id);
            if (!cancelled) setCampaign(updated);
          }
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Could not connect to the pilot API.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!campaign?.campaign_id || campaign.status === "completed" || campaign.status === "incomplete") return;
    let cancelled = false;
    const poll = async () => {
      try {
        await refreshStatus(campaign.campaign_id);
        if (!cancelled) setError("");
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Pilot status refresh failed.");
      }
    };
    const timer = window.setInterval(() => void poll(), 5000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [campaign?.campaign_id, campaign?.status, refreshStatus]);

  useEffect(() => {
    const poll = async () => {
      try { setTarget(await rangeApi.sqliTargetStatus()); }
      catch { /* The next status refresh can recover the target indicator. */ }
    };
    const timer = window.setInterval(() => void poll(), 10000);
    return () => window.clearInterval(timer);
  }, []);

  async function startPilot() {
    setBusy(true);
    setError("");
    try {
      const targetState = await rangeApi.sqliTargetStatus();
      setTarget(targetState);
      if (targetState.status !== "available") throw new Error("The isolated SQLi target is unavailable.");
      const next = await rangeApi.startPilot();
      window.localStorage.setItem(STORAGE_KEY, next.campaign_id);
      setCampaign(next);
      evaluationStarts.current.delete(next.campaign_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The balanced pilot could not start.");
    } finally {
      setBusy(false);
      setLoading(false);
    }
  }

  async function stopTraining() {
    if (!campaign) return;
    setBusy(true);
    setError("");
    try {
      await Promise.all(campaign.training.runs
        .filter((item) => item.status === "queued" || item.status === "running")
        .map((item) => rangeApi.stopTrainingRun(item.training_run_id)));
      await refreshStatus(campaign.campaign_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not request a graceful stop.");
    } finally {
      setBusy(false);
    }
  }

  async function refresh() {
    setBusy(true);
    setError("");
    try {
      setTarget(await rangeApi.sqliTargetStatus());
      if (campaign) await refreshStatus(campaign.campaign_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not refresh pilot status.");
    } finally {
      setBusy(false);
      setLoading(false);
    }
  }

  const training = campaign?.training;
  const evaluation = campaign?.evaluation;
  const activeTraining = Boolean(training?.runs.some((item) => item.status === "queued" || item.status === "running"));
  const trainingProgress = training ? Math.min(100, 100 * training.completed_updates / Math.max(1, training.requested_updates)) : 0;
  const evaluationProgress = evaluation ? Math.min(100, 100 * evaluation.completed_matches / Math.max(1, evaluation.requested_matches)) : 0;
  const verdict = evaluation?.balance_verdict;

  return <div className="pilot-page">
    <header className="page-header">
      <div>
        <div className="eyebrow">Fresh policies · three seeds · held-out matches</div>
        <h1 className="page-title">Balanced adversarial pilot</h1>
        <p className="page-deck">Red and Blue choose from the same round-start evidence. Blue’s acknowledged control reaches the target before Red’s request.</p>
      </div>
      <div className="header-actions">
        {activeTraining && <button className="button button-secondary" onClick={() => void stopTraining()} disabled={busy}>
          <CircleStop size={14} />Stop at episode boundary
        </button>}
        <button className="button button-secondary" onClick={() => void refresh()} disabled={busy || loading}>
          <RefreshCw size={14} />Refresh status
        </button>
        {!activeTraining && <button className="button" onClick={() => void startPilot()} disabled={busy || target?.status !== "available"}>
          <ShieldCheck size={14} />Start three-seed pilot
        </button>}
      </div>
    </header>

    {error && <div className="error-banner" role="alert"><AlertTriangle size={15} /><span>{error}</span></div>}

    <TrainingPresets targetAvailable={target?.status === "available"} />

    <section className="pilot-console panel" aria-label="Pilot protocol and time window">
      <div className="pilot-console-mark"><Bug size={17} /><span>VIOLET / PILOT 01</span></div>
      <div><span>Target</span><strong className={target?.status === "available" ? "is-healthy" : "is-unhealthy"}>{target?.status ?? "checking"}</strong></div>
      <div><span>Interaction</span><strong>{campaign?.interaction_protocol_schema_id ?? "sqli.round_protocol.v2"}</strong></div>
      <div><span>Reward</span><strong>{campaign?.reward_schema_id ?? "sqli.reward.v3"}</strong></div>
      <div><span>Training cutoff</span><strong>{time(campaign?.training_deadline ?? null)}</strong></div>
      <div><span>Evaluation cutoff</span><strong>{time(campaign?.evaluation_deadline ?? null)}</strong></div>
    </section>

    {!campaign ? <section className="panel pilot-intro">
      <div className="panel-kicker">Explicit start · no work runs on page load</div>
      <h2>Three new Red/Blue policy pairs</h2>
      <p>Each seed receives up to 50 complete training episodes, with a 32-round cap. The final hour is reserved for 15 frozen matches per available seed. Evaluation seeds are disjoint from training.</p>
      <div className="pilot-seed-list"><span>Seed 17</span><span>Seed 18</span><span>Seed 19</span></div>
      <button className="button" onClick={() => void startPilot()} disabled={busy || target?.status !== "available"}>
        <ShieldCheck size={14} />{loading ? "Checking pilot status…" : "Start pilot"}
      </button>
    </section> : <>
      <section className="pilot-overview-grid">
        <article className="panel pilot-progress-card">
          <div className="pilot-card-topline"><span className="panel-kicker">Campaign {campaign.campaign_id}</span><span className={`pill ${activeTraining ? "pill-green" : ""}`}>{stateLabel(campaign.status)}</span></div>
          <h2>Training updates</h2>
          <div className="pilot-progress-numbers"><strong>{training?.completed_updates ?? 0}</strong><span>of {training?.requested_updates ?? 150} paired PPO updates</span></div>
          <div className="pilot-progress-track" role="progressbar" aria-label="Training updates" aria-valuenow={Math.round(trainingProgress)} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${trainingProgress}%` }} /></div>
          <p>Fresh initialization on seeds 17, 18, and 19. Red and Blue both update after each complete, verified episode.</p>
        </article>
        <article className="panel pilot-progress-card pilot-evaluation-card">
          <div className="pilot-card-topline"><span className="panel-kicker">Frozen held-out evaluation</span><span className="pill">{evaluation?.attempted_matches ?? 0} attempts</span></div>
          <h2>{evaluation?.completed_matches ?? 0} of {evaluation?.requested_matches ?? 45} matches complete</h2>
          <div className="pilot-progress-track" role="progressbar" aria-label="Held-out matches" aria-valuenow={Math.round(evaluationProgress)} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${evaluationProgress}%` }} /></div>
          <p>{evaluation?.incomplete_matches ?? 45} matches remain incomplete or unstarted. Evaluation stops at the four-hour deadline on a match boundary.</p>
        </article>
      </section>

      <section className="pilot-stat-grid" aria-label="Pooled held-out results">
        <article className="panel pilot-stat pilot-red"><span>Red wins</span><strong>{evaluation?.red_wins ?? 0}</strong><small>{pct(evaluation?.red_win_rate ?? null)} · 95% CI {interval(evaluation?.red_win_rate_wilson_95 ?? null)}</small></article>
        <article className="panel pilot-stat pilot-blue"><span>Blue wins</span><strong>{evaluation?.blue_wins ?? 0}</strong><small>{pct(evaluation?.blue_win_rate ?? null)} · 95% CI {interval(evaluation?.blue_win_rate_wilson_95 ?? null)}</small></article>
        <article className="panel pilot-stat"><span>Verified disclosures</span><strong>{evaluation?.verified_disclosures ?? 0}</strong><small>Target-confirmed unauthorized reads</small></article>
        <article className="panel pilot-stat"><span>Incomplete matches</span><strong>{evaluation?.incomplete_matches ?? 45}</strong><small>{evaluation?.attempted_matches ?? 0} attempts recorded</small></article>
      </section>

      <section className="pilot-results-grid">
        <article className="panel pilot-results-card">
          <div className="panel-head"><div><span className="panel-kicker">Service and control evidence</span><h2 className="panel-title">Held-out outcomes</h2></div><span className={`pill ${evaluation?.balanced ? "pill-green" : "pill-amber"}`}>{verdict?.replaceAll("_", " ") ?? "not evaluated"}</span></div>
          <div className="pilot-metric-list">
            <div><span>Verified disclosure rate</span><strong>{pct(evaluation?.verified_exposure_rate ?? null)}</strong></div>
            <div><span>Verified prevention</span><strong>{pct(evaluation?.verified_prevention_rate ?? null)}</strong></div>
            <div><span>Legitimate lookup availability</span><strong>{pct(evaluation?.legitimate_service_availability ?? null)}</strong></div>
            <div><span>Undecided completed matches</span><strong>{evaluation?.undecided_matches ?? 0}</strong></div>
          </div>
          <p className="pilot-verdict-note">{evaluation?.balance_rule ?? "Balance requires each side to win at least 20% and neither to exceed 80% of completed held-out matches."} A balance claim requires all 45 matches to complete.</p>
          <p className="pilot-verdict-note">Uncertainty: {evaluation?.uncertainty_method ?? "Wilson score 95% intervals will be reported for pooled and per-seed win rates."}</p>
        </article>

        <article className="panel pilot-seeds-card">
          <div className="panel-head"><div><span className="panel-kicker">Training and evaluation by seed</span><h2 className="panel-title">Campaign runs</h2></div><span className="pill">{training?.runs.length ?? 0} training runs</span></div>
          <div className="pilot-seed-table-wrap">
            <table className="pilot-seed-table">
              <thead><tr><th>Seed</th><th>Training</th><th>Checkpoint</th><th>Held out</th><th>Red / Blue wins</th></tr></thead>
              <tbody>{(training?.runs ?? []).map((item) => {
                const result = evaluation?.per_seed.find((row) => row.training_seed === item.seed);
                return <tr key={item.training_run_id}>
                  <th scope="row">{item.seed}</th>
                  <td>{item.completed_episodes} / {item.requested_episodes}<small>{item.status}{item.stop_requested ? " · stop requested" : ""}</small></td>
                  <td>{item.checkpoint_available ? `v${item.policy_versions?.red ?? "—"} / v${item.policy_versions?.blue ?? "—"}` : "unavailable"}<small>{item.checkpoint_sha256 ? item.checkpoint_sha256.slice(0, 12) : "No verified digest"}</small></td>
                  <td>{result ? `${result.completed_matches} / ${result.requested_matches}` : "not started"}<small>{result?.evaluation_status ?? "—"}</small></td>
                  <td>{result ? `${result.red_wins} / ${result.blue_wins}` : "—"}<small>{result ? `disclosures ${result.verified_disclosures}` : "—"}</small></td>
                </tr>;
              })}</tbody>
            </table>
          </div>
          {evaluation?.per_seed.map((item) => <p className="pilot-seed-uncertainty" key={item.training_seed}>
            Seed {item.training_seed}: Red {pct(item.red_win_rate)} (95% CI {interval(item.red_win_rate_wilson_95)}) · Blue {pct(item.blue_win_rate)} (95% CI {interval(item.blue_win_rate_wilson_95)}) · prevention {pct(item.verified_prevention_rate)} · availability {pct(item.legitimate_service_availability)}.
          </p>)}
        </article>
      </section>
    </>}

    <aside className="notice pilot-footnote"><strong>Interpretation.</strong> Training win rates are descriptive. Only the 45 frozen held-out matches determine whether the pooled result meets the stated balance threshold.</aside>
  </div>;
}
