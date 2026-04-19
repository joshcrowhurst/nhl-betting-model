import React from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  Tooltip, CartesianGrid, Legend,
} from "recharts";
import { DailyBreakdown } from "../types";

interface Props {
  data: DailyBreakdown[];
}

function toChartData(data: DailyBreakdown[]) {
  return data.map((d) => ({
    date: d.date.slice(5),   // "MM-DD"
    accuracy: d.total > 0 ? Math.round((d.correct / d.total) * 1000) / 10 : null,
    vb_accuracy: d.vb_total > 0 ? Math.round((d.vb_correct / d.vb_total) * 1000) / 10 : null,
  }));
}

export default function TrendChart({ data }: Props) {
  if (!data || data.length === 0) {
    return (
      <div style={{ color: "#475569", fontSize: 14, textAlign: "center", padding: "40px 0" }}>
        No trend data yet — resolve some predictions first.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={toChartData(data)} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#64748b" }} />
        <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "#64748b" }} unit="%" />
        <Tooltip
          contentStyle={{ background: "#1e293b", border: "none", borderRadius: 8, fontSize: 13 }}
          formatter={(v: number) => `${v.toFixed(1)}%`}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          type="monotone"
          dataKey="accuracy"
          name="Model accuracy"
          stroke="#38bdf8"
          strokeWidth={2}
          dot={false}
          connectNulls
        />
        <Line
          type="monotone"
          dataKey="vb_accuracy"
          name="Value bet accuracy"
          stroke="#22c55e"
          strokeWidth={2}
          dot={false}
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
