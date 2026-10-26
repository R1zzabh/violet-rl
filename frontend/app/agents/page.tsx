"use client";
// app/agents/page.tsx — Agent registry. Shows the agents that ACTUALLY exist
// in violet.agents (served by /agents). RL variants are marked unavailable
// until a trained checkpoint is loaded — never faked.

import { useCallback, useEffect, useState } from "react";
import { Bot, RefreshCw } from "lucide-react";
import { ApiError, api, type AgentInfo } from "@/lib/api";

function AgentColumn({ title, agents, accent, note }: { title: string; agents: AgentInfo[]; accent: string; note: string }) {
  return (
    <div className="card-brutal border-l-[6px]" style={{ borderLeftColor: accent }}>
      <div className="section-label mb-1">{title}</div>
      <p className="mono text-[10px] text-muted-foreground mb-3">{note}</p>
      <div className="space-y-2">
        {agents.map((a) => (
          <div key={a.id} className="rounded-lg border border-[rgba(212,175,55,.12)] bg-[rgba(19,13,9,.62)] p-3">
            <div className="flex items-center justify-between gap-2 mb-1">
              <span className="mono text-sm font-bold text-white">{a.label}</span>
              <span
                className="badge-brutal mono"
                style={{ color: a.rl ? "#d4af37" : "#8fc49a", borderColor: a.rl ? "#d4af37" : "#8fc49a" }}
              >
                {a.rl ? "UNAVAILABLE" : "AVAILABLE"}
              </span>
            </div>
            <div className="mono text-[10px] text-muted-foreground">id: {a.id}</div>
            <div className="text-xs text-muted-foreground mt-1">{a.description}</div>
            {a.rl && (
              <div className="mono text-[10px] text-[#e3c879] mt-1">
                Requires a trained checkpoint (backend/runs/&lt;run&gt;/checkpoints/best). Not trained in this phase.
              </div>
            )}
          </div>
        ))}
        {agents.length === 0 && (
          <div className="mono text-xs text-muted-foreground">Backend did not return any agents.</div>
        )}
      </div>
    </div>
  );
}

export default function AgentsPage() {
  const [red, setRed] = useState<AgentInfo[]>([]);
  const [blue, setBlue] = useState<AgentInfo[]>([]);
  const [error, setError] = useState<string>("");

  const load = useCallback(async () => {
    try {
      const resp = await api.agents();
      setRed(resp.red);
      setBlue(resp.blue);
      setError("");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void Promise.resolve().then(load);
  }, [load]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Bot size={14} className="text-[#d4af37]" />
            <span className="text-xs mono uppercase tracking-[0.2em] text-muted-foreground font-bold">
              Agent Registry
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight">Red &amp; Blue Agents</h1>
        </div>
        <button className="btn-brutal btn-brutal-ghost btn-brutal-sm" onClick={() => void load()}>
          <RefreshCw size={13} /> REFRESH
        </button>
      </div>

      {error && (
        <div className="card-brutal border-red-500 border-2 p-4">
          <div className="mono text-xs font-bold text-red-400 uppercase mb-1">Backend unavailable</div>
          <p className="text-sm text-muted-foreground">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <AgentColumn title="🔴 RED — ATTACKERS" agents={red} accent="#f07832" note="Implemented in backend/violet/agents/red.py" />
        <AgentColumn title="🔵 BLUE — DEFENDERS" agents={blue} accent="#91bce2" note="Implemented in backend/violet/agents/blue.py" />
      </div>

      <p className="mono text-[11px] text-muted-foreground">
        Source: <span className="text-[#d4af37]">GET /agents</span> — registry mirrors the agent classes that exist in the
        backend. Nothing here is speculative.
      </p>
    </div>
  );
}
