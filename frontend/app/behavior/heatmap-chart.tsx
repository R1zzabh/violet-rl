"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

type HeatMapRow = { state: string; values: number[] };
type HeatMapData = HeatMapRow[];

export default function HeatmapChart({ data }: { data: HeatMapData }) {
  const maxVal = Math.max(...data.flatMap((r) => r.values));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout="vertical" barSize={28}>
        <XAxis type="number" domain={[0, maxVal + 2]} tick={{ fill: "#8f7965", fontFamily: "var(--font-mono)", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "rgba(212,175,55,.2)" }} />
        <YAxis dataKey="state" type="category" width={90} tick={{ fill: "#eadcc8", fontFamily: "var(--font-mono)", fontSize: 10 }} tickLine={false} axisLine={false} />
        <Tooltip contentStyle={{ fontFamily: "var(--font-mono)", fontSize: 10, backgroundColor: "#24170f", border: "1px solid rgba(212,175,55,.25)", borderRadius: 9, color: "#f5e6d0", boxShadow: "0 10px 24px rgba(7,4,2,.4)" }} />
        <Bar dataKey="values" radius={0} barSize={28}>
          {data.map((row, ri) =>
            row.values.map((v, ci) => (
              <Cell
                key={`${ri}-${ci}`}
                fill={v === 0 ? "#21150e" : `rgb(${Math.round(212 * (v / maxVal) + 54 * (1 - v / maxVal))}, ${Math.round(175 * (v / maxVal) + 36 * (1 - v / maxVal))}, ${Math.round(55 * (v / maxVal) + 20 * (1 - v / maxVal))})`}
              />
            ))
          )}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
