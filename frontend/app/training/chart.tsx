"use client";

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

type MetricPoint = { episode: number; reward: number };

export default function TrainingChart({ data }: { data: MetricPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 8 }}>
        <XAxis
          dataKey="episode"
          tick={{ fill: "#8f7965", fontFamily: "var(--font-mono)", fontSize: 9 }}
          tickLine={false}
          axisLine={{ stroke: "rgba(212,175,55,.2)" }}
          domain={[0, 2000]}
        />
        <YAxis
          tick={{ fill: "#8f7965", fontFamily: "var(--font-mono)", fontSize: 9 }}
          tickLine={false}
          axisLine={{ stroke: "rgba(212,175,55,.2)" }}
          domain={[-60, 80]}
        />
        <Tooltip
          contentStyle={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            backgroundColor: "#24170f",
            color: "#f5e6d0",
            border: "1px solid rgba(212,175,55,.25)",
            borderRadius: 9,
            boxShadow: "0 10px 24px rgba(7,4,2,.4)",
          }}
        />
        <ReferenceLine y={0} stroke="rgba(212,175,55,.2)" strokeWidth={1} />
        <Line
          type="monotone"
          dataKey="reward"
          stroke="#d4af37"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#d4af37", stroke: "#1a110d", strokeWidth: 2 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
