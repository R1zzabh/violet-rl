"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

type DataPoint = { action: string; pct: number };

export default function SimulationCharts({ data, accent }: { data: DataPoint[]; accent: string }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout="vertical" barSize={14}>
        <XAxis type="number" domain={[0, 50]} tick={{ fill: "#9a9a9a", fontFamily: "var(--font-mono)", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "#222" }} />
        <YAxis dataKey="action" type="category" width={52} tick={{ fill: "#f5f5f5", fontFamily: "var(--font-mono)", fontSize: 9 }} tickLine={false} axisLine={false} />
        <Tooltip contentStyle={{ fontFamily: "var(--font-mono)", fontSize: 10, backgroundColor: "#111", border: "2px solid #000", borderRadius: 0, boxShadow: "0 4px 0 #000" }} />
        <Bar dataKey="pct" radius={0}>
          {data.map((_, i) => (
            <Cell key={i} fill={i === 0 ? accent : i === 1 ? accent + "cc" : i === 2 ? "#f5c518" : i === 3 ? "#c99b00" : "#4a4a4a"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
