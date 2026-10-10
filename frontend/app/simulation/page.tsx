"use client";
// app/simulation/page.tsx — VIOLET LIVE BATTLE.
// Everything rendered here comes from the backend REST + WebSocket stream.
// When the backend is down or a feature is unavailable, the UI says so
// explicitly instead of showing placeholder values.

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Activity, Pause, Play, RotateCcw, StepForward, Square } from "lucide-react";
import NetworkGraph from "@/components/network-graph";
import { useSimulation } from "@/lib/use-simulation";

const EVENT_COLORS: Record<string, string> = {
  red: "#ff2a2a",
  blue: "#2a7aff",
  system: "#f5c518",
};

function EventRow({ ev }: { ev: import("@/lib/use-simulation").SimEvent }) {
  const color = EVENT_COLORS[ev.agent] ?? "#9a9a9a";
  return (
    <div className="flex items-start gap-2 border-b border-[#1c1c1c] py-1.5 last:border-0">
      <span className="mono text-[10px] text-muted-foreground w-10 shrink-0 text-right">
        {ev.step ?? "—"}
      </span>
      <span
        className="mono text-[10px] font-bold shrink-0 uppercase"
        style={{ color, minWidth: 34 }}
      >
        {ev.agent}
      </span>
      <span className="mono text-[11px] text-white shrink-0">→ {ev.action}</span>
      {ev.target && (
        <span className="mono text-[11px] text-violet-400 shrink-0">{ev.target}</span>
      )}
      <span className="mono text-[10px] text-muted-foreground truncate">{ev.result}</span>
    </div>
  );
}

function phaseBadge(phase: string, wsStatus: string | null) {
  const map: Record<string, { label: string; color: string }> = {
    idle: { label: "IDLE", color: "#6a6a6a" },
    "loading-config": { label: "LOADING", color: "#f5c518" },
    "config-error": { label: "CONFIG ERROR", color: "#ff2a2a" },
    ready: { label: "READY", color: "#2ecc71" },
    "creating-session": { label: "CREATING", color: "#f5c518" },
    connecting: { label: "CONNECTING", color: "#f5c518" },
    live: { label: "LIVE", color: "#ff2a2a" },
    paused: { label: "PAUSED", color: "#f5c518" },
    ended: { label: "ENDED", color: "#2a7aff" },
    "backend-down": { label: "BACKEND DOWN", color: "#ff2a2a" },
  };
  const badge = map[phase] ?? { label: phase.toUpperCase(), color: "#6a6a6a" };
  return (
    <span className="flex items-center gap-3">
      {wsStatus && (
        <span className="mono text-[10px] text-muted-foreground uppercase">ws: {wsStatus}</span>
      )}
      <span
        className="badge-brutal mono"
        style={{ borderColor: badge.color, color: badge.color }}
      >
        {badge.label}
      </span>
    </span>
  );
}

function SimulationInner() {
  const sim = useSimulation();
  const { state } = sim;
  const searchParams = useSearchParams();
  const eventLogRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);
  // Mirror of autoScrollRef for rendering — refs must not be read during render.
  const [autoScroll, setAutoScroll] = useState(true);

  // Optional deep-link config: /simulation?red=scripted_lateral&blue=reactive&seed=7
  useEffect(() => {
    const red = searchParams.get("red");
    const blue = searchParams.get("blue");
    const seed = searchParams.get("seed");
    if (red) sim.setConfig("redPolicy", red);
    if (blue) sim.setConfig("bluePolicy", blue);
    if (seed) sim.setConfig("seed", Number(seed) || 42);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  // Keep the event log pinned to the newest events while streaming.
  useEffect(() => {
    if (autoScrollRef.current && eventLogRef.current) {
      eventLogRef.current.scrollTop = eventLogRef.current.scrollHeight;
    }
  }, [state.events]);

  const hasSession = state.simId !== null && state.simId !== "pending";
  const controlsDisabled = !hasSession || state.phase === "backend-down";

  const nodeStats = useMemo(() => {
    const nodes = Object.values(state.network?.nodes ?? {});
    const compromised = nodes.filter(
      (n) => n.compromise_status !== "none" && n.compromise_status !== "discovered",
    ).length;
    const isolated = nodes.filter((n) => n.isolated === "isolated").length;
    const alerts = nodes.filter((n) => n.alert_level !== "none").length;
    return { total: nodes.length, compromised, isolated, alerts };
  }, [state.network]);

  return (
    <div className="space-y-4">
      {/* ---------- HEADER ---------- */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Activity size={14} className="text-violet-400" />
            <span className="text-xs mono uppercase tracking-[0.2em] text-muted-foreground font-bold">
              VIOLET — LIVE BATTLE
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight">
            {state.scenarios.find((s) => s.id === state.scenarioId)?.name ?? "Live Battle"}
            <span className="mono text-sm text-muted-foreground ml-3">
              episode ≤ {state.maxCycles} · seed {state.seed}
            </span>
          </h1>
        </div>
        <div className="flex items-center gap-2">
          {phaseBadge(state.phase, state.wsStatus)}
          <span className="badge-brutal mono">
            STEP {state.step}/{state.maxCycles}
          </span>
        </div>
      </div>

      {state.phase === "backend-down" && (
        <div className="card-brutal border-red-500 border-2 p-4">
          <div className="mono text-xs font-bold text-red-400 uppercase mb-1">
            Backend unavailable
          </div>
          <p className="text-sm text-muted-foreground">
            {state.configError || "Could not reach the VIOLET API."} Start it with{" "}
            <code className="mono text-violet-400">uvicorn violet.api.app:app --port 8000</code>{" "}
            (or <code className="mono text-violet-400">docker compose up</code>) and retry.
          </p>
        </div>
      )}

      {/* ---------- CONFIGURATION ---------- */}
      <div className="card-brutal">
        <div className="section-label mb-3">CONFIGURATION</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <label className="block">
            <span className="mono text-[10px] uppercase text-muted-foreground tracking-wider">Scenario</span>
            <select
              className="w-full mt-1 border-2 border-[#333] bg-[#111] px-2 py-1.5 mono text-xs"
              value={state.scenarioId}
              onChange={(e) => sim.setConfig("scenarioId", e.target.value)}
              disabled={state.phase === "live"}
            >
              {state.scenarios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.node_count} nodes)
                </option>
              ))}
              {state.scenarios.length === 0 && <option value="v1_enterprise">v1_enterprise</option>}
            </select>
          </label>
          <label className="block">
            <span className="mono text-[10px] uppercase text-muted-foreground tracking-wider">Red agent</span>
            <select
              className="w-full mt-1 border-2 border-[#333] bg-[#111] px-2 py-1.5 mono text-xs"
              value={state.redPolicy}
              onChange={(e) => sim.setConfig("redPolicy", e.target.value)}
              disabled={state.phase === "live"}
            >
              {state.redAgents.map((a) => (
                <option key={a.id} value={a.id} disabled={a.rl}>
                  {a.label}
                  {a.rl ? " — unavailable (no trained red checkpoint)" : ""}
                </option>
              ))}
              {state.redAgents.length === 0 && <option value="scripted_aggressive">Scripted Aggressive</option>}
            </select>
          </label>
          <label className="block">
            <span className="mono text-[10px] uppercase text-muted-foreground tracking-wider">Blue agent</span>
            <select
              className="w-full mt-1 border-2 border-[#333] bg-[#111] px-2 py-1.5 mono text-xs"
              value={state.bluePolicy}
              onChange={(e) => sim.setConfig("bluePolicy", e.target.value)}
              disabled={state.phase === "live"}
            >
              {state.blueAgents.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.label}
                </option>
              ))}
              {state.blueAgents.length === 0 && <option value="rule_based">Rule-Based</option>}
            </select>
          </label>
          <label className="block">
            <span className="mono text-[10px] uppercase text-muted-foreground tracking-wider">Seed</span>
            <input
              type="number"
              className="w-full mt-1 border-2 border-[#333] bg-[#111] px-2 py-1.5 mono text-xs"
              value={state.seed}
              onChange={(e) => sim.setConfig("seed", Number(e.target.value) || 0)}
              disabled={state.phase === "live"}
            />
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-2 mt-4 pt-3 border-t border-[#222]">
          {!hasSession ? (
            <button
              className="btn-brutal btn-brutal-red btn-brutal-sm"
              onClick={() => void sim.createSession()}
              disabled={state.phase === "backend-down" || state.phase === "loading-config"}
            >
              <Play size={13} /> START
            </button>
          ) : (
            <>
              {state.phase !== "live" && state.phase !== "ended" && (
                <button className="btn-brutal btn-brutal-red btn-brutal-sm" onClick={sim.resume}>
                  <Play size={13} /> RESUME
                </button>
              )}
              {state.phase === "live" && (
                <button className="btn-brutal btn-brutal-yellow btn-brutal-sm" onClick={sim.stop}>
                  <Pause size={13} /> PAUSE
                </button>
              )}
              <button
                className="btn-brutal btn-brutal-ghost btn-brutal-sm"
                onClick={sim.stepOnce}
                disabled={controlsDisabled || state.phase === "ended"}
              >
                <StepForward size={13} /> STEP
              </button>
              <button className="btn-brutal btn-brutal-ghost btn-brutal-sm" onClick={sim.reset} disabled={controlsDisabled}>
                <RotateCcw size={13} /> RESET
              </button>
              <button
                className="btn-brutal btn-brutal-ghost btn-brutal-sm"
                onClick={() => void sim.restart(state.seed)}
                disabled={state.phase === "backend-down"}
                title="Discard this session and start a fresh episode"
              >
                <Square size={13} /> RESTART
              </button>
            </>
          )}
          {state.phase === "ended" && (
            <span className="mono text-xs font-bold" style={{ color: state.winner === "red" ? "#ff2a2a" : state.winner === "blue" ? "#2a7aff" : "#f5c518" }}>
              {state.outcome === "timeout"
                ? "EPISODE TIMEOUT — undecided"
                : state.winner === "red"
                  ? "RED WINS — critical asset compromised"
                  : state.winner === "blue"
                    ? "BLUE WINS — attack contained"
                    : "EPISODE ENDED"}
            </span>
          )}
          {state.configError && state.phase !== "backend-down" && (
            <span className="mono text-xs text-red-400">⚠ {state.configError}</span>
          )}
        </div>
      </div>

      {/* ---------- AGENT PANELS ---------- */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card-brutal border-l-[6px]" style={{ borderLeftColor: "#ff2a2a" }}>
          <div className="section-label mb-3">🔴 RED — ATTACKER ({state.redPolicy})</div>
          <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm mb-3">
            <span className="mono text-muted-foreground text-[10px] uppercase tracking-wider">Reward</span>
            <span className="mono font-bold text-white">{state.redReward.toFixed(2)}</span>
            <span className="mono text-muted-foreground text-[10px] uppercase tracking-wider">Last action</span>
            <span className="mono font-bold text-white">{state.lastRedAction || "—"}</span>
            <span className="mono text-muted-foreground text-[10px] uppercase tracking-wider">Compromised nodes</span>
            <span className="mono font-bold" style={{ color: nodeStats.compromised ? "#ff2a2a" : "#fff" }}>
              {nodeStats.compromised} / {nodeStats.total || "—"}
            </span>
          </div>
        </div>
        <div className="card-brutal border-l-[6px]" style={{ borderLeftColor: "#2a7aff" }}>
          <div className="section-label mb-3">🔵 BLUE — DEFENDER ({state.bluePolicy})</div>
          <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm mb-3">
            <span className="mono text-muted-foreground text-[10px] uppercase tracking-wider">Reward</span>
            <span className="mono font-bold text-white">{state.blueReward.toFixed(2)}</span>
            <span className="mono text-muted-foreground text-[10px] uppercase tracking-wider">Last action</span>
            <span className="mono font-bold text-white">{state.lastBlueAction || "—"}</span>
            <span className="mono text-muted-foreground text-[10px] uppercase tracking-wider">Alerts / isolated</span>
            <span className="mono font-bold text-white">
              {nodeStats.alerts} / {nodeStats.isolated}
            </span>
          </div>
        </div>
      </div>

      {/* ---------- LIVE NETWORK ---------- */}
      <div className="card-brutal-lg">
        <div className="flex items-center justify-between mb-3">
          <div className="section-label mb-0">LIVE NETWORK — actual backend state</div>
          <div className="mono text-[10px] text-muted-foreground">
            t = {state.network?.step ?? state.step}
          </div>
        </div>
        <NetworkGraph network={state.network} />
      </div>

      {/* ---------- EVENT STREAM ---------- */}
      <div className="card-brutal">
        <div className="flex items-center justify-between mb-2">
          <div className="section-label mb-0">LIVE EVENT STREAM — backend events only</div>
          <label className="mono text-[10px] text-muted-foreground flex items-center gap-1">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => {
                autoScrollRef.current = e.target.checked;
                setAutoScroll(e.target.checked);
              }}
            />
            auto-scroll
          </label>
        </div>
        <div ref={eventLogRef} className="max-h-[240px] overflow-y-auto pr-1">
          {state.events.length === 0 ? (
            <div className="mono text-xs text-muted-foreground py-3">
              {hasSession ? "No events yet — press STEP or RESUME." : "No episode running."}
            </div>
          ) : (
            state.events.map((ev) => <EventRow key={ev.seq} ev={ev} />)
          )}
        </div>
      </div>
    </div>
  );
}

export default function SimulationPage() {
  return (
    <Suspense fallback={<div className="mono text-xs text-muted-foreground">Loading…</div>}>
      <SimulationInner />
    </Suspense>
  );
}
