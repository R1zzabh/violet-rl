"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

type HeatMapRow = { state: string; values: number[] };
type HeatMapData = HeatMapRow[];

export default function HeatmapChart({ data }: { data: HeatMapData }) {
  const maxVal = Math.max(...data.flatMap((r) => r.values));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout="vertical" barSize={28}>
        <XAxis type="number" domain={[0, maxVal + 2]} tick={{ fill: "#9a9a9a", fontFamily: "var(--font-mono)", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "#222" }} />
        <YAxis dataKey="state" type="category" width={90} tick={{ fill: "#f5f5f5", fontFamily: "var(--font-mono)", fontSize: 10 }} tickLine={false} axisLine={false} />
        <Tooltip contentStyle={{ fontFamily: "var(--font-mono)", fontSize: 10, backgroundColor: "#111", border: "2px solid #000", borderRadius: 0, boxShadow: "0 4px 0 #000" }} />
        <Bar dataKey="values" radius={0} barSize={28}>
          {data.map((row, ri) =>
            row.values.map((v, ci) => (
              <Cell
                key={`${ri}-${ci}`}
                fill={v === 0 ? "#1a1a1a" : `rgb(${Math.round(255 * (v / maxVal))}, ${Math.round(42 * (1 - v / maxVal))}, ${Math.round(170 * (1 - v / maxVal))})`}
              />
            ))
          )}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
