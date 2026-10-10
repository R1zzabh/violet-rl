"use client";

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

type MetricPoint = { episode: number; reward: number };

export default function TrainingChart({ data }: { data: MetricPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 8 }}>
        <XAxis
          dataKey="episode"
          tick={{ fill: "#9a9a9a", fontFamily: "var(--font-mono)", fontSize: 9 }}
          tickLine={false}
          axisLine={{ stroke: "#222" }}
          domain={[0, 2000]}
        />
        <YAxis
          tick={{ fill: "#9a9a9a", fontFamily: "var(--font-mono)", fontSize: 9 }}
          tickLine={false}
          axisLine={{ stroke: "#222" }}
          domain={[-60, 80]}
        />
        <Tooltip
          contentStyle={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            backgroundColor: "#111",
            border: "2px solid #000",
            borderRadius: 0,
            boxShadow: "0 4px 0 #000",
          }}
        />
        <ReferenceLine y={0} stroke="#222" strokeWidth={1} />
        <Line
          type="monotone"
          dataKey="reward"
          stroke="#f5c518"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#f5c518", stroke: "#000", strokeWidth: 2 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
