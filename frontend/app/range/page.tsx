"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  BookOpenCheck,
  CircleStop,
  Clock3,
  RefreshCw,
  Shield,
  Swords,
  TerminalSquare,
} from "lucide-react";
import {
  rangeApi,
  type ManualAction,
  type RangeEvent,
  type RangeJobResult,
  type TargetStatus,
  type TrainingRun,
} from "../../lib/range-api";

type View = "live" | "workbench" | "soc" | "learning" | "evaluation";
type ManualSession = { runId: string; episodeId: string; seed: number; step: number };
const VIEW_LABELS: { id: View; label: string }[] = [
  { id: "live", label: "Live range" },
  { id: "workbench", label: "Red workbench" },
  { id: "soc", label: "Blue SOC" },
  { id: "learning", label: "Learning lab" },
  { id: "evaluation", label: "Evaluation" },
];

function objectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function statusClass(value: string) {
  if (value === "completed" || value === "available" || value === "verified") return "badge-brutal badge-green";
  if (value === "failed" || value === "unavailable" || value === "incomplete") return "badge-brutal badge-red";
  if (value === "running" || value === "queued" || value === "suspected") return "badge-brutal badge-yellow";
  return "badge-brutal";
}

function eventTitle(event: RangeEvent): string {
  const titles: Record<string, string> = {
    agent_decision_recorded: "Policy decision",
    agent_transition_recorded: "Transition persisted",
    action_submitted: "Red request submitted",
    manual_action_submitted: "Manual request submitted",
    evaluator_probe_submitted: "Verifier probe submitted",
    target_evidence_ingested: "Target evidence received",
    executor_result: "HTTP response recorded",
    detection_suspected: "Blue alert",
    target_control_submitted: "Control requested",
    target_control_acknowledged: "Target acknowledged control",
    control_verification: "Control independently verified",
    reward_recorded: "Reward attributed",
    training_update_recorded: "PPO update",
    episode_finished: "Episode finished",
  };
  return titles[event.event_type] ?? event.event_type.replaceAll("_", " ");
}

function EventList({ events, empty }: { events: RangeEvent[]; empty: string }) {
  if (!events.length) return <div className="rounded-xl border border-dashed border-[#d4af37]/20 p-8 text-center text-xs text-[#a99079]">{empty}</div>;
  return <ol className="space-y-2">
    {events.map((event) => {
      const payload = objectValue(event.payload);
      const evidence = objectValue(payload.evidence);
      const result = objectValue(payload.result);
      const action = objectValue(payload.action);
      return <li key={event.event_id} className="flex gap-3 rounded-lg border border-[#d4af37]/10 bg-[#130d09]/55 px-3 py-3">
        <div className="flex w-10 shrink-0 flex-col items-center pt-0.5">
          <span className="font-mono text-[9px] text-[#8f7965]">{String(event.seq).padStart(4, "0")}</span>
          <span className="mt-2 h-full min-h-4 w-px bg-[#d4af37]/15" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs font-semibold text-[#eadcc8]">{eventTitle(event)}</span>
            <time className="font-mono text-[9px] text-[#8f7965]">{new Date(event.occurred_at).toLocaleTimeString()}</time>
          </div>
          <div className="mt-1 break-words font-mono text-[9px] leading-5 text-[#a99079]">
            {String(action.operation ?? result.http_status ?? evidence.http_status ?? payload.status ?? "")}
            {evidence.principal_ref ? ` · ${String(evidence.principal_ref)}` : ""}
            {evidence.resource_ref ? ` → ${String(evidence.resource_ref)}` : ""}
            {result.route_catalog_id ? ` · ${String(result.route_catalog_id)}` : ""}
            {event.episode_id ? ` · ep ${event.episode_id.slice(0, 8)}` : ""}
          </div>
        </div>
      </li>;
    })}
  </ol>;
}

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return <div className="panel min-w-0 p-4">
    <div className="mono-label">{label}</div>
    <div className="mt-3 truncate text-xl font-semibold tracking-tight text-[#f5e6d0]">{value}</div>
    {note && <div className="mt-2 text-[10px] text-[#8f7965]">{note}</div>}
  </div>;
}

export default function RangePage() {
  const [view, setView] = useState<View>("live");
  const [target, setTarget] = useState<TargetStatus | null>(null);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [activeRunId, setActiveRunId] = useState("");
  const [events, setEvents] = useState<RangeEvent[]>([]);
  const [manualEvents, setManualEvents] = useState<RangeEvent[]>([]);
  const [manualSession, setManualSession] = useState<ManualSession | null>(null);
  const [aliases, setAliases] = useState<Record<string, string>>({});
  const [manualResult, setManualResult] = useState<RangeJobResult | null>(null);
  const [targetError, setTargetError] = useState("");
  const [runError, setRunError] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [mode, setMode] = useState<"train" | "evaluate">("train");
  const [runKind, setRunKind] = useState<"teaching" | "research">("teaching");
  const [episodes, setEpisodes] = useState(1);
  const [opponent, setOpponent] = useState<"scripted" | "random">("scripted");
  const [checkpointRunId, setCheckpointRunId] = useState("");
  const [operation, setOperation] = useState<"DISCOVER_ROUTE" | "AUTH_TEST" | "CHECK_DOCUMENT_ACCESS" | "CHECK_TARGET_HEALTH">("AUTH_TEST");
  const [credential, setCredential] = useState<"synthetic_alice" | "synthetic_bob" | "synthetic_invalid">("synthetic_alice");
  const [route, setRoute] = useState<"home" | "login" | "document">("login");
  const [documentRef, setDocumentRef] = useState<"doc_alice" | "doc_bob">("doc_bob");
  const [principal, setPrincipal] = useState<"alice" | "bob">("alice");

  const refresh = useCallback(async () => {
    try {
      const [targetStatus, runList] = await Promise.all([rangeApi.targetStatus(), rangeApi.trainingRuns()]);
      setTarget(targetStatus);
      setRuns(runList.items);
      setTargetError("");
      setRunError("");
      setActiveRunId((current) => current || runList.items[0]?.training_run_id || "");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Range service is unavailable";
      setTargetError(message);
      setRunError(message);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
  }, [refresh]);

  const refreshEvents = useCallback(async (runId: string) => {
    if (!runId) { setEvents([]); return; }
    try { setEvents((await rangeApi.events(runId)).events); }
    catch (error) { setRunError(error instanceof Error ? error.message : "Could not load the persisted event chain"); }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void refreshEvents(activeRunId), 0);
    return () => window.clearTimeout(initial);
  }, [activeRunId, refreshEvents]);

  const refreshManualEvents = useCallback(async (session: ManualSession) => {
    try { setManualEvents((await rangeApi.events(session.runId)).events); }
    catch (error) { setRunError(error instanceof Error ? error.message : "Could not load manual request history"); }
  }, []);

  const selectedRun = useMemo(() => runs.find((run) => run.training_run_id === activeRunId) ?? null, [runs, activeRunId]);
  const completedJointRuns = runs.filter((run) => run.status === "completed" && run.role === "joint" && run.mode === "train" && run.result?.checkpoint_sha256 && (runKind !== "research" || (run.config.run_kind === "research" && Number(run.result.completed_training_episodes ?? 0) >= 10000)));
  const latestUpdateEvents = events.filter((event) => event.event_type === "training_update_recorded").slice(-12).reverse();
  const decisions = events.filter((event) => event.event_type === "agent_decision_recorded").slice(-8).reverse();
  const evidenceEvents = events.filter((event) => ["target_evidence_ingested", "detection_suspected", "target_control_acknowledged", "control_verification"].includes(event.event_type)).slice(-30).reverse();
  const requestCount = events.filter((event) => ["action_submitted", "manual_action_submitted", "evaluator_probe_submitted"].includes(event.event_type)).length;
  const latestTargetEvidence = events.filter((event) => event.event_type === "target_evidence_ingested").at(-1);

  async function waitForControl(commandId: string) {
    for (let attempt = 0; attempt < 60; attempt += 1) {
      const control = await rangeApi.control(commandId);
      if (control.status === "applied") return;
      if (["failed", "rejected"].includes(String(control.status))) throw new Error(`Target reset ${String(control.status)}`);
      await new Promise((resolve) => window.setTimeout(resolve, 150));
    }
    throw new Error("Target reset acknowledgement timed out");
  }

  async function startManualSession() {
    setBusy(true); setNotice(""); setRunError("");
    try {
      const session: ManualSession = {
        runId: crypto.randomUUID(), episodeId: crypto.randomUUID(), seed: Math.floor(Math.random() * 1_000_000), step: 0,
      };
      const reset = await rangeApi.resetTarget({ seed: session.seed, run_id: session.runId, episode_id: session.episodeId });
      await waitForControl(reset.command_id);
      setManualSession(session); setAliases({}); setManualEvents([]); setManualResult(null);
      setNotice(`Fresh synthetic target session ready · seed ${session.seed}`);
    } catch (error) { setRunError(error instanceof Error ? error.message : "Could not reset the target"); }
    finally { setBusy(false); }
  }

  async function pollJob(jobId: string): Promise<RangeJobResult> {
    for (let attempt = 0; attempt < 60; attempt += 1) {
      const value = await rangeApi.job(jobId);
      if ("result" in value) return value;
      await new Promise((resolve) => window.setTimeout(resolve, 150));
    }
    throw new Error("Executor response timed out; the request may have an unknown outcome");
  }

  async function sendManualAction(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!manualSession) { setRunError("Start a fresh manual session before sending a request."); return; }
    let action: ManualAction;
    if (operation === "DISCOVER_ROUTE") action = { operation, route_candidate_id: route };
    else if (operation === "AUTH_TEST") action = { operation, credential_ref: credential };
    else if (operation === "CHECK_TARGET_HEALTH") action = { operation };
    else {
      const sessionAlias = aliases[principal];
      if (!sessionAlias) { setRunError(`Authenticate synthetic ${principal} before requesting a document.`); return; }
      action = { operation, document_ref: documentRef, session_alias: sessionAlias };
    }
    setBusy(true); setRunError(""); setNotice("");
    try {
      const submitted = await rangeApi.submitManualAction({
        action, run_id: manualSession.runId, episode_id: manualSession.episodeId, step_id: manualSession.step,
      });
      const completed = await pollJob(submitted.job_id);
      const result = completed.result;
      if (result.principal_ref && result.session_alias) {
        setAliases((current) => ({ ...current, [result.principal_ref as string]: result.session_alias as string }));
      }
      const nextSession = { ...manualSession, step: manualSession.step + 1 };
      setManualSession(nextSession); setManualResult(completed);
      setNotice(`Manual request completed · ${result.http_status ?? result.status} · target event ${result.target_event_id ?? "not recorded"}`);
      await refreshManualEvents(nextSession);
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "Manual request failed");
    } finally { setBusy(false); }
  }

  async function launchTraining(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setRunError(""); setNotice("");
    try {
      const run = await rangeApi.createTrainingRun({
        role: "joint", mode, run_kind: runKind, episodes, max_steps: 12, seed: Math.floor(Math.random() * 1_000_000), opponent,
        ...(mode === "evaluate" && checkpointRunId ? { checkpoint_run_id: checkpointRunId } : {}),
      });
      setActiveRunId(run.training_run_id); setView("learning");
      setNotice(`Joint ${mode} job created · ${run.training_run_id}`);
      await refresh();
    } catch (error) { setRunError(error instanceof Error ? error.message : "Could not create training run"); }
    finally { setBusy(false); }
  }

  async function stopRun() {
    if (!activeRunId) return;
    try { await rangeApi.stopTrainingRun(activeRunId); await refresh(); }
    catch (error) { setRunError(error instanceof Error ? error.message : "Could not request a stop"); }
  }

  return <div className="space-y-7">
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <div className="eyebrow">Instrumented range · single target</div>
        <h1 className="page-heading">Violet Ledger</h1>
        <p className="page-lede">A synchronized Red, target, and Blue console for the synthetic Alice/Bob data-access objective. Requests and policy decisions are linked to persisted target evidence.</p>
      </div>
      <button className="btn-brutal btn-brutal-ghost" onClick={() => void refresh()} aria-label="Refresh range data"><RefreshCw size={14} />Refresh</button>
    </header>

    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <Metric label="Target status" value={target?.status ?? (targetError ? "unavailable" : "checking")} note={target ? `${target.target_id} · ${target.target_version}` : "Live API status"} />
      <Metric label="Selected run" value={selectedRun?.status ?? "no run"} note={selectedRun ? `${selectedRun.role} policy · ${selectedRun.mode}` : "Choose or create a persisted run"} />
      <Metric label="Target requests" value={String(requestCount)} note={selectedRun ? "Current run event chain" : "No run selected"} />
      <Metric label="Latest target event" value={latestTargetEvidence ? String(objectValue(latestTargetEvidence.payload).target_event_id ?? "recorded").slice(0, 12) : "none"} note={latestTargetEvidence ? new Date(latestTargetEvidence.occurred_at).toLocaleTimeString() : "No target evidence in selected run"} />
    </div>

    {(runError || targetError) && <div role="alert" className="rounded-lg border border-[#ff7868]/30 bg-[#ff7868]/[.07] px-4 py-3 text-xs text-[#ffc0b7]">{runError || targetError}</div>}
    {notice && <div role="status" className="rounded-lg border border-[#8fc49a]/25 bg-[#8fc49a]/[.06] px-4 py-3 text-xs text-[#c8e0ca]">{notice}</div>}

    <nav role="tablist" aria-label="Violet Ledger sections" className="flex gap-1 overflow-x-auto border-b border-[#d4af37]/15">
      {VIEW_LABELS.map(({ id, label }) => <button key={id} type="button" role="tab" aria-selected={view === id} onClick={() => setView(id)} className={`shrink-0 border-b-2 px-3 py-3 text-[11px] font-medium transition ${view === id ? "border-[#d4af37] text-[#f1d995]" : "border-transparent text-[#9d846c] hover:text-[#e9dac8]"}`}>{label}</button>)}
    </nav>

    {view === "live" && <div className="space-y-5">
      <section className="grid gap-3 lg:grid-cols-3">
        <div className="panel min-h-36 border-t-2 border-t-[#ff7868]/65">
          <div className="flex items-center justify-between"><span className="eyebrow" style={{ color: "var(--red)" }}>01 · Red policy</span><Swords size={16} className="text-[#ff7868]" /></div>
          <h2 className="mt-4 text-base font-semibold">Bounded requests</h2>
          <p className="mt-1 text-xs leading-5 text-[#a99079]">Decisions are persisted before dispatch. Manual workbench requests carry a separate origin and never enter PPO rollouts.</p>
          <div className="mt-3 font-mono text-[9px] text-[#8f7965]">{events.filter((event) => ["action_submitted", "manual_action_submitted"].includes(event.event_type)).length} actions in selected run</div>
        </div>
        <div className="panel min-h-36 border-t-2 border-t-[#d4af37]/65">
          <div className="flex items-center justify-between"><span className="eyebrow">02 · Violet Ledger</span><Activity size={16} className="text-[#d4af37]" /></div>
          <h2 className="mt-4 text-base font-semibold">{target?.status === "available" ? "Service responding" : "Service status unknown"}</h2>
          <p className="mt-1 text-xs leading-5 text-[#a99079]">The executor sends finite typed actions to the isolated target. Target-side telemetry provides correlated request evidence.</p>
          <div className="mt-3 font-mono text-[9px] text-[#8f7965]">{target?.status === "available" ? `${target.target_id} · health age ${target.last_health_check_age_seconds?.toFixed(1) ?? "—"}s` : "No current health evidence"}</div>
        </div>
        <div className="panel min-h-36 border-t-2 border-t-[#79b8ff]/65">
          <div className="flex items-center justify-between"><span className="eyebrow" style={{ color: "var(--blue)" }}>03 · Blue policy / SOC</span><Shield size={16} className="text-[#79b8ff]" /></div>
          <h2 className="mt-4 text-base font-semibold">Telemetry, controls, verification</h2>
          <p className="mt-1 text-xs leading-5 text-[#a99079]">Ownership hardening is the only learnable control until its effect and legitimate-user impact are verified independently.</p>
          <div className="mt-3 font-mono text-[9px] text-[#8f7965]">{evidenceEvents.filter((event) => event.event_type === "detection_suspected").length} alerts · {evidenceEvents.filter((event) => event.event_type === "control_verification").length} verifications</div>
        </div>
      </section>
      <section className="grid gap-4 xl:grid-cols-[1.1fr_.9fr]">
        <div className="panel">
          <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
            <div><span className="section-label">Persisted event chain</span><h2 className="mt-1 text-base font-semibold">Latest activity</h2></div>
            <label className="flex items-center gap-2 text-[10px] text-[#9d846c]">Run
              <select aria-label="Select range run" className="select-brutal min-w-44 py-1.5" value={activeRunId} onChange={(event) => setActiveRunId(event.target.value)}>
                <option value="">No run selected</option>
                {runs.map((run) => <option key={run.training_run_id} value={run.training_run_id}>{run.role.toUpperCase()} · {run.status} · {run.training_run_id.slice(0, 8)}</option>)}
              </select>
            </label>
          </div>
          <EventList events={events.slice(-18).reverse()} empty="No persisted events yet. Launch a joint run or create a manual workbench session." />
        </div>
        <div className="panel">
          <div className="section-label">Run controls</div>
          <h2 className="mt-1 text-base font-semibold">Joint Red / Blue policies</h2>
          <p className="mt-2 text-xs leading-5 text-[#a99079]">Both policies stay fixed during an episode and update independently after its rollout is complete. Incomplete episodes are excluded.</p>
          <form onSubmit={launchTraining} className="mt-5 space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="block text-[10px] text-[#9d846c]">Mode
                <select className="select-brutal mt-1.5" value={mode} onChange={(event) => {
                  const nextMode = event.target.value as "train" | "evaluate";
                  setMode(nextMode);
                  if (runKind === "research") setEpisodes(nextMode === "train" ? 10000 : 100);
                }}>
                  <option value="train">Train both policies</option><option value="evaluate">Evaluate frozen pair</option>
                </select>
              </label>
              <label className="block text-[10px] text-[#9d846c]">Run type
                <select className="select-brutal mt-1.5" value={runKind} onChange={(event) => {
                  const nextKind = event.target.value as "teaching" | "research";
                  setRunKind(nextKind);
                  if (nextKind === "research") setEpisodes(mode === "train" ? 10000 : 100);
                }}>
                  <option value="teaching">Teaching run</option><option value="research">Research run</option>
                </select>
              </label>
              <label className="block text-[10px] text-[#9d846c]">{mode === "train" ? "Training episodes per seed" : "Paired held-out episodes"} <span className="float-right text-[#e7d5c0]">{episodes.toLocaleString()}</span>
                <input className="input-brutal mt-1.5" type="number" min={runKind === "research" ? (mode === "train" ? 10000 : 100) : 1} max={10000} value={episodes} onChange={(event) => setEpisodes(Math.max(runKind === "research" ? (mode === "train" ? 10000 : 100) : 1, Math.min(10000, Number(event.target.value) || 1)))} />
              </label>
            </div>
            <label className="block text-[10px] text-[#9d846c]">Baseline matchup
              <select className="select-brutal mt-1.5" value={opponent} onChange={(event) => setOpponent(event.target.value as "scripted" | "random")}><option value="scripted">Scripted pair</option><option value="random">Random legal actions</option></select>
            </label>
            {mode === "evaluate" && <label className="block text-[10px] text-[#9d846c]">Completed joint checkpoint
              <select className="select-brutal mt-1.5" value={checkpointRunId} onChange={(event) => setCheckpointRunId(event.target.value)} required>
                <option value="">Choose a checkpoint</option>
                {completedJointRuns.map((run) => <option key={run.training_run_id} value={run.training_run_id}>{run.training_run_id.slice(0, 8)} · {String(run.config.episodes ?? "")} episodes</option>)}
              </select>
            </label>}
            <button disabled={busy || (mode === "evaluate" && !checkpointRunId)} className="btn-brutal btn-brutal-yellow w-full"><BookOpenCheck size={14} />{busy ? "Submitting…" : mode === "train" ? "Start joint training" : "Start held-out evaluation"}</button>
            <p className="text-[9px] leading-5 text-[#8f7965]">Research training requires 10,000 completed episodes for this seed. Research evaluation uses at least 100 paired held-out episodes and reports no improvement claim without five independent training seeds and the availability criterion.</p>
          </form>
          {selectedRun && <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-[#d4af37]/10 pt-4">
            <Link className="inline-flex items-center gap-1.5 text-[10px] text-[#f1d995] hover:text-white" href={`/range/${selectedRun.training_run_id}`}>Open replay and details<ArrowUpRight size={12} /></Link>
            {(["queued", "running"] as string[]).includes(selectedRun.status) && <button onClick={() => void stopRun()} className="btn-brutal btn-brutal-ghost min-h-8 py-1.5 text-[9px]"><CircleStop size={12} />Request stop</button>}
          </div>}
        </div>
      </section>
    </div>}

    {view === "workbench" && <div className="grid gap-4 xl:grid-cols-[.85fr_1.15fr]">
      <section className="panel">
        <div className="flex items-center gap-2"><TerminalSquare size={17} className="text-[#f07832]" /><div className="section-label">Red workbench · typed HTTP only</div></div>
        <h2 className="mt-2 text-base font-semibold">Request composer</h2>
        <p className="mt-2 text-xs leading-5 text-[#a99079]">Actions go through the finite executor catalog to Violet Ledger. Each is tagged manual and excluded from policy training.</p>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-[#d4af37]/10 bg-[#130d09]/65 px-3 py-3">
          <div><div className="mono-label">Manual target session</div><div className="mt-1 font-mono text-[10px] text-[#c1aa91]">{manualSession ? `${manualSession.runId.slice(0, 8)} · seed ${manualSession.seed}` : "No active session"}</div></div>
          <button className="btn-brutal btn-brutal-ghost min-h-8 py-1.5 text-[9px]" onClick={() => void startManualSession()} disabled={busy}><RefreshCw size={12} />Reset target</button>
        </div>
        {manualSession && <div className="mt-3 flex flex-wrap gap-2">{(["alice", "bob"] as const).map((name) => <span key={name} className={`badge-brutal ${aliases[name] ? "badge-green" : ""}`}>{name} {aliases[name] ? "authenticated" : "no session"}</span>)}</div>}
        <form className="mt-5 space-y-4" onSubmit={sendManualAction}>
          <label className="block text-[10px] text-[#9d846c]">Supported operation
            <select className="select-brutal mt-1.5" value={operation} onChange={(event) => setOperation(event.target.value as typeof operation)}>
              <option value="DISCOVER_ROUTE">Discover a catalog route</option><option value="AUTH_TEST">Authenticate synthetic account</option><option value="CHECK_DOCUMENT_ACCESS">Request a document</option><option value="CHECK_TARGET_HEALTH">Check target health</option>
            </select>
          </label>
          {operation === "DISCOVER_ROUTE" && <label className="block text-[10px] text-[#9d846c]">Route candidate<select className="select-brutal mt-1.5" value={route} onChange={(event) => setRoute(event.target.value as typeof route)}><option value="home">Home</option><option value="login">Login</option><option value="document">Document API</option></select></label>}
          {operation === "AUTH_TEST" && <label className="block text-[10px] text-[#9d846c]">Synthetic credential reference<select className="select-brutal mt-1.5" value={credential} onChange={(event) => setCredential(event.target.value as typeof credential)}><option value="synthetic_alice">Alice</option><option value="synthetic_bob">Bob</option><option value="synthetic_invalid">Invalid fixture credential</option></select></label>}
          {operation === "CHECK_DOCUMENT_ACCESS" && <div className="grid gap-3 sm:grid-cols-2">
            <label className="block text-[10px] text-[#9d846c]">Session principal<select className="select-brutal mt-1.5" value={principal} onChange={(event) => setPrincipal(event.target.value as typeof principal)}><option value="alice">Alice</option><option value="bob">Bob</option></select></label>
            <label className="block text-[10px] text-[#9d846c]">Document reference<select className="select-brutal mt-1.5" value={documentRef} onChange={(event) => setDocumentRef(event.target.value as typeof documentRef)}><option value="doc_alice">Alice document</option><option value="doc_bob">Bob document</option></select></label>
          </div>}
          <button disabled={busy || !manualSession} className="btn-brutal btn-brutal-red w-full"><Swords size={14} />{busy ? "Waiting for target response…" : "Send bounded request"}</button>
        </form>
        {manualResult && <div className="mt-5 rounded-lg border border-[#d4af37]/15 bg-[#130d09] p-3">
          <div className="flex flex-wrap items-center justify-between gap-2"><span className="section-label">Actual HTTP response</span><span className={statusClass(manualResult.result.http_status && manualResult.result.http_status < 400 ? "completed" : "failed")}>{manualResult.result.http_status ?? manualResult.result.status}</span></div>
          <div className="mt-3 grid grid-cols-2 gap-2 font-mono text-[9px] text-[#a99079]"><span>{manualResult.result.method ?? "—"} {manualResult.result.route_catalog_id ?? "—"}</span><span>request {manualResult.result.request_id.slice(0, 10)}</span><span>target event {manualResult.result.target_event_id?.slice(0, 10) ?? "none"}</span><span>principal {manualResult.result.principal_ref ?? "anonymous"}</span></div>
          <pre className="mt-3 max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-md border border-[#d4af37]/10 bg-black/20 p-3 font-mono text-[9px] leading-5 text-[#e3d2bd]">{manualResult.result.response_excerpt || manualResult.result.error_code || "No response body was recorded."}</pre>
        </div>}
      </section>
      <section className="panel">
        <div className="mb-4 flex items-center justify-between gap-3"><div><div className="section-label">Manual action origin</div><h2 className="mt-1 text-base font-semibold">Request history</h2></div><span className="badge-brutal">Never in PPO</span></div>
        <EventList events={manualEvents.slice().reverse()} empty="Start a manual target session, then send a request to see its committed evidence here." />
      </section>
    </div>}

    {view === "soc" && <section className="panel">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3"><div><div className="eyebrow" style={{ color: "var(--blue)" }}>Blue security operations</div><h2 className="mt-1 text-lg font-semibold">Telemetry and control evidence</h2><p className="mt-1 text-xs text-[#a99079]">Only sanitized target telemetry and evidence-linked detections are shown to the Blue policy.</p></div><span className="badge-brutal badge-blue">{evidenceEvents.length} records</span></div>
      <div className="mb-4 grid gap-3 sm:grid-cols-3">
        <Metric label="Candidate alerts" value={String(evidenceEvents.filter((event) => event.event_type === "detection_suspected").length)} />
        <Metric label="Acknowledged controls" value={String(evidenceEvents.filter((event) => event.event_type === "target_control_acknowledged").length)} />
        <Metric label="Independent checks" value={String(evidenceEvents.filter((event) => event.event_type === "control_verification").length)} />
      </div>
      <EventList events={evidenceEvents} empty="No Blue telemetry is available for this run. Select a persisted run or start a manual target session." />
      <p className="mt-4 text-[10px] leading-5 text-[#8f7965]">Ownership hardening earns credit only after Alice’s cross-account read is denied and both legitimate owner reads succeed. Rate limits and actor blocks remain unavailable to policy sampling pending independent verification.</p>
    </section>}

    {view === "learning" && <div className="grid gap-4 xl:grid-cols-[1.05fr_.95fr]">
      <section className="panel">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><div className="eyebrow">Episode-boundary optimization</div><h2 className="mt-1 text-base font-semibold">Red and Blue update records</h2></div><span className={statusClass(selectedRun?.status ?? "unknown")}>{selectedRun?.status ?? "no run"}</span></div>
        <div className="overflow-x-auto"><table className="table-brutal"><thead><tr><th>Policy</th><th>Episode</th><th>Samples</th><th>Loss</th><th>Entropy</th><th>Policy version</th></tr></thead><tbody>
          {latestUpdateEvents.map((event) => {
            const payload = objectValue(event.payload); const metrics = objectValue(payload.metrics);
            return <tr key={event.event_id}><td className={payload.role === "red" ? "text-[#ff9a8d]" : "text-[#9bcbff]"}>{String(payload.role ?? "—").toUpperCase()}</td><td>{String(payload.episode_index ?? payload.update_index ?? "—")}</td><td>{String(metrics.sample_count ?? "—")}</td><td>{typeof metrics.loss === "number" ? metrics.loss.toFixed(4) : "—"}</td><td>{typeof metrics.entropy === "number" ? metrics.entropy.toFixed(3) : "—"}</td><td>{String(payload.policy_version ?? "—")}</td></tr>;
          })}
        </tbody></table></div>
        {!latestUpdateEvents.length && <div className="mt-3 rounded-lg border border-dashed border-[#d4af37]/15 p-6 text-center text-xs text-[#8f7965]">No optimizer update records in the selected run. Updates appear only after a completed episode rollout.</div>}
        <Link href={selectedRun ? `/range/${selectedRun.training_run_id}` : "/range"} className="mt-4 inline-flex items-center gap-2 text-[10px] text-[#f1d995] hover:text-white">Inspect decision replay<ArrowUpRight size={12} /></Link>
      </section>
      <section className="panel">
        <div className="section-label">Saved policy samples</div><h2 className="mt-1 text-base font-semibold">Action distributions</h2>
        <p className="mt-2 text-xs leading-5 text-[#a99079]">The probabilities below are the persisted distribution at the selected decision. A changed distribution alone does not demonstrate better security behavior.</p>
        <div className="mt-4 space-y-2">
          {decisions.map((event) => {
            const payload = objectValue(event.payload); const decision = objectValue(payload.decision); const sample = objectValue(decision.policy_sample);
            const distribution = Array.isArray(sample.action_distribution) ? sample.action_distribution as number[] : [];
            return <div key={event.event_id} className="rounded-lg border border-[#d4af37]/10 bg-[#130d09]/55 p-3">
              <div className="flex items-center justify-between gap-2"><span className={decision.role === "red" ? "badge-brutal badge-red" : "badge-brutal badge-blue"}>{String(decision.role ?? "policy")}</span><span className="font-mono text-[9px] text-[#8f7965]">{String(decision.policy_id ?? "baseline")}</span></div>
              <div className="mt-2 font-mono text-[9px] text-[#c1aa91]">{JSON.stringify(decision.selected_action ?? {})}</div>
              {distribution.length > 0 && <div className="mt-3 grid gap-1.5">{distribution.map((probability, index) => <div key={index} className="grid grid-cols-[25px_1fr_42px] items-center gap-2 font-mono text-[8px] text-[#9d846c]"><span>a{index}</span><div className="h-1.5 overflow-hidden rounded-full bg-white/[.06]"><div className="h-full rounded-full bg-[#d4af37]" style={{ width: `${Math.max(0, Math.min(100, probability * 100))}%` }} /></div><span className="text-right">{(probability * 100).toFixed(1)}%</span></div>)}</div>}
              {!distribution.length && <div className="mt-2 text-[9px] text-[#8f7965]">No sampled distribution: scripted/manual decision.</div>}
            </div>;
          })}
          {!decisions.length && <div className="rounded-lg border border-dashed border-[#d4af37]/15 p-6 text-center text-xs text-[#8f7965]">No saved policy decisions for the selected run.</div>}
        </div>
      </section>
    </div>}

    {view === "evaluation" && <section className="panel">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3"><div><div className="eyebrow">Frozen-policy comparison</div><h2 className="mt-1 text-lg font-semibold">Held-out evaluation record</h2></div>{selectedRun && <Link href={`/range/${selectedRun.training_run_id}`} className="btn-brutal btn-brutal-ghost min-h-8 py-1.5 text-[9px]">Open run<ArrowUpRight size={12} /></Link>}</div>
      {selectedRun?.result?.evaluation ? (() => {
        const summary = objectValue(selectedRun.result.evaluation);
        const interval = Array.isArray(summary.paired_bootstrap_ci_95) ? summary.paired_bootstrap_ci_95 as number[] : null;
        return <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Metric label="Paired episodes" value={`${String(summary.paired_episodes ?? 0)} / ${String(summary.requested_pairs ?? 0)}`} />
            <Metric label="Mean paired difference" value={typeof summary.mean_paired_difference === "number" ? summary.mean_paired_difference.toFixed(4) : "—"} />
            <Metric label="95% bootstrap interval" value={interval?.length === 2 ? `${interval[0].toFixed(3)} to ${interval[1].toFixed(3)}` : "Unavailable"} />
            <Metric label="Claim status" value={String(summary.claim ?? "not reported")} />
          </div>
          <div className="rounded-lg border border-[#d4af37]/10 bg-[#130d09]/55 p-4 text-xs leading-6 text-[#c1aa91]">
            <div>Baseline: <span className="font-mono text-[#eadcc8]">{String(summary.baseline ?? "—")}</span></div>
            <div>Independent training seeds represented: <span className="font-mono text-[#eadcc8]">{String(summary.independent_training_seeds ?? summary.training_seed_count ?? "—")}</span></div>
            <div>Reproducibility: raw paired episodes and summary digests are stored with the run artifacts.</div>
            <div className="mt-2 text-[#8f7965]">An improvement claim requires at least five independent training seeds, 100 paired held-out episodes, and the preregistered availability criteria.</div>
          </div>
        </div>;
      })() : <div className="rounded-lg border border-dashed border-[#d4af37]/15 p-9 text-center">
        <Clock3 size={21} className="mx-auto text-[#8f7965]" />
        <p className="mt-3 text-sm text-[#c1aa91]">No evaluation report is attached to the selected run.</p>
        <p className="mt-1 text-[10px] text-[#8f7965]">Choose a completed joint checkpoint and launch a separate frozen evaluation from Live range.</p>
      </div>}
    </section>}
  </div>;
}
