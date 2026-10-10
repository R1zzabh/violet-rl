"use client";
// components/network-graph.tsx — live network visualization driven by real
// backend state. Layout is computed deterministically from the scenario's
// connection graph (BFS depth layering); node statuses map the backend enums
// to the VIOLET status vocabulary.

import { useMemo } from "react";
import type { NetworkSnapshot, NodeSnapshot } from "@/lib/api";

export type StatusKey =
  | "SAFE"
  | "DISCOVERED"
  | "SUSPICIOUS"
  | "COMPROMISED"
  | "ISOLATED"
  | "PATCHED"
  | "DECOY"
  | "RECOVERING";

export const STATUS_COLORS: Record<StatusKey, string> = {
  SAFE: "#2ecc71",
  DISCOVERED: "#f5c518",
  SUSPICIOUS: "#ff9f1a",
  COMPROMISED: "#ff2a2a",
  ISOLATED: "#888888",
  PATCHED: "#2a7aff",
  DECOY: "#9b59ff",
  RECOVERING: "#00b8a9",
};

const STATUS_ORDER: StatusKey[] = [
  "SAFE",
  "DISCOVERED",
  "SUSPICIOUS",
  "COMPROMISED",
  "ISOLATED",
  "PATCHED",
  "DECOY",
  "RECOVERING",
];

export function statusForNode(n: NodeSnapshot): StatusKey {
  if (n.isolated === "isolated") return "ISOLATED";
  if (n.is_honeypot) return "DECOY";
  if (n.service_status === "patched") return "PATCHED";
  if (n.service_status === "down") return "RECOVERING";
  if (n.compromise_status === "compromised" || n.compromise_status === "privileges_escalated" || n.compromise_status === "exfiltrated") return "COMPROMISED";
  if (n.alert_level === "high" || n.alert_level === "medium") return "SUSPICIOUS";
  if (n.compromise_status === "discovered" || n.alert_level === "low") return "DISCOVERED";
  return "SAFE";
}

type LayoutNode = {
  id: string;
  label: string;
  x: number;
  y: number;
  node: NodeSnapshot;
};

function layout(network: NetworkSnapshot): { nodes: LayoutNode[]; edges: { from: string; to: string }[]; width: number; height: number } {
  const ids = Object.keys(network.nodes);
  if (ids.length === 0) return { nodes: [], edges: [], width: 100, height: 60 };

  // BFS depth layering from gateways (or the first id as a fallback root).
  const depth = new Map<string, number>();
  const roots = ids.filter((id) => network.nodes[id].node_type === "gateway");
  const queue = [...(roots.length ? roots : [ids[0]])];
  queue.forEach((id) => depth.set(id, 0));
  let qi = 0;
  while (qi < queue.length) {
    const cur = queue[qi++];
    const d = depth.get(cur) ?? 0;
    for (const nb of network.nodes[cur].connections) {
      if (!depth.has(nb)) {
        depth.set(nb, d + 1);
        queue.push(nb);
      }
    }
  }
  // Orphans (not reached via outgoing connections) get max depth + 1.
  const maxD = Math.max(0, ...[...depth.values()]);
  for (const id of ids) if (!depth.has(id)) depth.set(id, maxD + 1);

  // Group by depth, order within layer by id for deterministic layout.
  const layers = new Map<number, string[]>();
  for (const id of ids) {
    const d = depth.get(id) ?? 0;
    if (!layers.has(d)) layers.set(d, []);
    layers.get(d)!.push(id);
  }
  for (const arr of layers.values()) arr.sort();

  const LAYER_W = 220;
  const NODE_H = 92;
  const width = layers.size * LAYER_W + 60;
  const maxLayerSize = Math.max(...[...layers.values()].map((a) => a.length));
  const height = maxLayerSize * NODE_H + 80;

  const nodes: LayoutNode[] = [];
  for (const [d, arr] of [...layers.entries()].sort((a, b) => a[0] - b[0])) {
    const layerSize = arr.length;
    const yCenter = height / 2;
    arr.forEach((id, i) => {
      const spread = (layerSize - 1) / 2;
      nodes.push({
        id,
        label: network.nodes[id].label,
        x: 60 + d * LAYER_W,
        y: yCenter + (i - spread) * NODE_H,
        node: network.nodes[id],
      });
    });
  }

  const edges: { from: string; to: string }[] = [];
  for (const id of ids) {
    for (const to of network.nodes[id].connections) {
      if (network.nodes[to]) edges.push({ from: id, to });
    }
  }
  return { nodes, edges, width, height };
}

export default function NetworkGraph({ network }: { network: NetworkSnapshot | null }) {
  const geo = useMemo(() => (network ? layout(network) : null), [network]);

  if (!geo || !network) {
    return (
      <div className="border-2 border-[#222] p-6 mono text-xs text-muted-foreground">
        NO NETWORK DATA — START A SIMULATION TO LOAD TOPOLOGY
      </div>
    );
  }

  const NW = 118;
  const NH = 44;

  return (
    <div>
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${geo.width} ${geo.height}`}
          width={geo.width}
          height={geo.height}
          style={{ minWidth: "100%", background: "transparent" }}
          role="img"
          aria-label="Live network topology"
        >
          {/* edges */}
          {geo.edges.map((e, i) => {
            const a = geo.nodes.find((n) => n.id === e.from);
            const b = geo.nodes.find((n) => n.id === e.to);
            if (!a || !b) return null;
            const x1 = a.x + NW / 2 + 4;
            const y1 = a.y;
            const x2 = b.x - NW / 2 - 4;
            const y2 = b.y;
            const alertHigh = network.active_alerts[e.to] && network.active_alerts[e.to] !== "none";
            return (
              <g key={`e-${i}`}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={alertHigh ? "#f5c518" : "#2a2a2a"} strokeWidth={alertHigh ? 3 : 2} />
                <circle cx={(x1 + x2) / 2} cy={(y1 + y2) / 2} r={2} fill="#3a3a3a" />
              </g>
              );
          })}

          {/* nodes */}
          {geo.nodes.map((n) => {
            const status = statusForNode(n.node);
            const color = STATUS_COLORS[status];
            const isComp = status === "COMPROMISED";
            return (
              <g key={n.id} transform={`translate(${n.x - NW / 2}, ${n.y - NH / 2})`}>
                <rect
                  x={4}
                  y={4}
                  width={NW}
                  height={NH}
                  fill="#000"
                  opacity={0.9}
                />
                <rect
                  x={0}
                  y={0}
                  width={NW}
                  height={NH}
                  fill="#141414"
                  stroke={isComp ? color : "#3a3a3a"}
                  strokeWidth={isComp ? 3 : 2}
                />
                <rect x={0} y={NH - 6} width={NW} height={6} fill={color} />
                <text x={NW / 2} y={18} textAnchor="middle" fill="#f5f5f5" fontSize={11} fontWeight={700} fontFamily="'Space Mono', monospace">
                  {n.label}
                </text>
                <text x={NW / 2} y={32} textAnchor="middle" fill={color} fontSize={8} fontFamily="'Space Mono', monospace">
                  {status}
                </text>
                {n.node.is_critical_asset && (
                  <text x={8} y={16} fontSize={9} fill="#f5c518">
                    ★
                  </text>
                )}
                {n.node.is_honeypot && (
                  <text x={NW - 14} y={16} fontSize={9} fill="#9b59ff">
                    ◆
                  </text>
                )}
                {isComp && (
                  <text x={NW - 16} y={30} fontSize={9} fill={color}>
                    ▲
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* status legend — exactly the required vocabulary */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 mono text-[10px]">
        {STATUS_ORDER.map((s) => (
          <span key={s} className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5" style={{ backgroundColor: STATUS_COLORS[s] }} />
            <span className="text-muted-foreground">{s}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
