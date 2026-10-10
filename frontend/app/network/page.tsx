"use client";
// app/network/page.tsx — Network Explorer. Shows the real V1 Enterprise
// topology: node inventory, vulnerabilities, services, and connectivity from
// /scenarios/v1_enterprise/topology. Static scenario data — no simulation
// state, no invented hosts.

import { useCallback, useEffect, useState } from "react";
import { Network as NetworkIcon, RefreshCw } from "lucide-react";
import NetworkGraph from "@/components/network-graph";
import { ApiError, api, type ScenarioDetail } from "@/lib/api";

export default function NetworkPage() {
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [error, setError] = useState<string>("");

  const load = useCallback(() => {
    api
      .scenarioTopology("v1_enterprise")
      .then((d) => {
        setDetail(d);
        setError("");
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  useEffect(() => {
    const t = setTimeout(load, 0);
    return () => clearTimeout(t);
  }, [load]);

  const nodes = detail ? Object.values(detail.network.nodes) : [];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <NetworkIcon size={14} className="text-violet-400" />
            <span className="text-xs mono uppercase tracking-[0.2em] text-muted-foreground font-bold">
              Network Explorer
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight">V1 Enterprise Topology</h1>
        </div>
        <button className="btn-brutal btn-brutal-ghost btn-brutal-sm" onClick={load}>
          <RefreshCw size={13} /> REFRESH
        </button>
      </div>

      {error && (
        <div className="card-brutal border-red-500 border-2 p-4">
          <div className="mono text-xs font-bold text-red-400 uppercase mb-1">Backend unavailable</div>
          <p className="text-sm text-muted-foreground">{error}</p>
        </div>
      )}

      {detail && (
        <>
          <div className="card-brutal">
            <div className="section-label mb-2">SCENARIO</div>
            <p className="text-sm text-muted-foreground">{String(detail.scenario.description)}</p>
            <div className="flex flex-wrap gap-2 mt-3 mono text-[10px]">
              <span className="badge-brutal mono">{nodes.length} nodes</span>
              <span className="badge-brutal mono">
                critical: {(detail.scenario.critical_assets as string[] | undefined)?.join(", ") ?? "—"}
              </span>
              <span className="badge-brutal mono">
                honeypots: {(detail.scenario.honeypots as string[] | undefined)?.join(", ") ?? "—"}
              </span>
            </div>
          </div>

          <div className="card-brutal-lg">
            <div className="section-label mb-3">GROUND-TRUTH TOPOLOGY (fresh reset state)</div>
            <NetworkGraph network={detail.network} />
          </div>

          <div className="card-brutal">
            <div className="section-label mb-3">NODE INVENTORY — vulnerabilities &amp; services</div>
            <div className="overflow-x-auto">
              <table className="table-brutal">
                <thead>
                  <tr>
                    <th>Node</th>
                    <th>Type</th>
                    <th>Value</th>
                    <th>Critical</th>
                    <th>Honeypot</th>
                    <th>Vulnerabilities</th>
                    <th>Active services</th>
                    <th>Connects to</th>
                  </tr>
                </thead>
                <tbody>
                  {nodes.map((n) => (
                    <tr key={n.id}>
                      <td className="mono font-bold text-white whitespace-nowrap">{n.label}</td>
                      <td className="mono">{n.node_type}</td>
                      <td className="mono">{n.value}</td>
                      <td>{n.is_critical_asset ? "★" : ""}</td>
                      <td>{n.is_honeypot ? "◆" : ""}</td>
                      <td className="mono text-[10px]">{n.vulnerabilities.length ? n.vulnerabilities.join(", ") : "—"}</td>
                      <td className="mono text-[10px]">{n.services.length ? n.services.join(", ") : "—"}</td>
                      <td className="mono text-[10px]">{n.connections.join(", ") || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
