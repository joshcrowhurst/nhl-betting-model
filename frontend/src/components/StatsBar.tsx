import React from "react";
import { Performance } from "../types";

interface Props {
  perf: Performance;
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div style={{
      background: "#1e293b",
      borderRadius: 10,
      padding: "14px 18px",
      flex: "1 1 140px",
      minWidth: 120,
    }}>
      <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 24, fontWeight: 800, color: "#f1f5f9" }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function StatsBar({ perf }: Props) {
  const accuracy = perf.accuracy !== null ? `${(perf.accuracy * 100).toFixed(1)}%` : "—";
  const vbAcc = perf.value_bet_accuracy !== null ? `${(perf.value_bet_accuracy * 100).toFixed(1)}%` : "—";
  const roi = perf.value_bet_roi !== null
    ? `${perf.value_bet_roi >= 0 ? "+" : ""}${(perf.value_bet_roi * 100).toFixed(1)}%`
    : "—";
  const roiColor = perf.value_bet_roi !== null && perf.value_bet_roi >= 0 ? "#22c55e" : "#ef4444";

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 24 }}>
      <Stat
        label="Accuracy"
        value={accuracy}
        sub={`${perf.correct_predictions}/${perf.resolved_predictions} resolved`}
      />
      <Stat
        label="Value Bet Acc"
        value={vbAcc}
        sub={`${perf.value_bets_correct}/${perf.total_value_bets} bets`}
      />
      <div style={{
        background: "#1e293b",
        borderRadius: 10,
        padding: "14px 18px",
        flex: "1 1 140px",
        minWidth: 120,
      }}>
        <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 4 }}>
          ROI
        </div>
        <div style={{ fontSize: 24, fontWeight: 800, color: roiColor }}>{roi}</div>
        <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 2 }}>flat-stake on value bets</div>
      </div>
      <Stat
        label="Total EV"
        value={perf.total_ev !== null ? perf.total_ev.toFixed(2) : "—"}
        sub="cumulative expected value"
      />
      <Stat
        label="Predictions"
        value={String(perf.total_predictions)}
        sub={`${perf.resolved_predictions} resolved · ${perf.total_predictions - perf.resolved_predictions} pending`}
      />
    </div>
  );
}
