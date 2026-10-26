"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

type DataPoint = { action: string; pct: number };

export default function SimulationCharts({ data, accent }: { data: DataPoint[]; accent: string }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout="vertical" barSize={14}>
        <XAxis type="number" domain={[0, 50]} tick={{ fill: "#8f7965", fontFamily: "var(--font-mono)", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(212,175,55,.2)" }} />
        <YAxis dataKey="action" type="category" width={52} tick={{ fill: "#f5e6d0", fontFamily: "var(--font-mono)", fontSize: 9 }} tickLine={false} axisLine={false} />
        <Tooltip contentStyle={{ fontFamily: "var(--font-mono)", fontSize: 10, backgroundColor: "#24170f", color: "#f5e6d0", border: "1px solid rgba(212,175,55,.25)", borderRadius: 9, boxShadow: "0 10px 24px rgba(7,4,2,.4)" }} />
        <Bar dataKey="pct" radius={0}>
          {data.map((_, i) => (
            <Cell key={i} fill={i === 0 ? accent : i === 1 ? accent + "cc" : i === 2 ? "#d4af37" : i === 3 ? "#f07832" : "#604733"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
