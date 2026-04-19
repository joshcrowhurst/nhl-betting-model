import React from "react";
import { Prediction } from "../types";

function fmtOdds(o: number | null): string {
  if (o === null) return "—";
  return o > 0 ? `+${Math.round(o)}` : String(Math.round(o));
}

function fmtPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

function fmtEv(ev: number | null): string {
  if (ev === null) return "";
  return ev > 0 ? `+${ev.toFixed(3)}` : ev.toFixed(3);
}

interface Props {
  p: Prediction;
}

export default function GameCard({ p }: Props) {
  const isPlayoff = p.game_type === 3;
  const homeEdge = p.home_win_prob - (p.market_home_prob ?? p.home_win_prob);
  const awayEdge = -homeEdge;

  const resolved = p.actual_home_win !== null;
  const homeWon = p.actual_home_win === 1;
  const awayWon = p.actual_home_win === 0;

  return (
    <div style={{
      background: "#1e293b",
      borderRadius: 12,
      padding: "16px 20px",
      borderLeft: p.is_value_bet ? "4px solid #22c55e" : "4px solid transparent",
      position: "relative",
    }}>
      {isPlayoff && (
        <span style={{
          position: "absolute", top: 10, right: 12,
          fontSize: 10, fontWeight: 700, color: "#f59e0b",
          background: "#451a03", padding: "2px 6px", borderRadius: 4,
        }}>PLAYOFFS</span>
      )}

      {/* Matchup header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: 0.5 }}>
          <TeamName team={p.away_team} won={awayWon} resolved={resolved} />
          <span style={{ color: "#475569", margin: "0 8px", fontWeight: 400, fontSize: 14 }}>@</span>
          <TeamName team={p.home_team} won={homeWon} resolved={resolved} />
        </div>
        {resolved && (
          <span style={{
            fontSize: 11, fontWeight: 600,
            color: p.correct ? "#22c55e" : "#ef4444",
            background: p.correct ? "#052e16" : "#2d0a0a",
            padding: "3px 8px", borderRadius: 6,
          }}>
            {p.correct ? "✓ CORRECT" : "✗ WRONG"}
          </span>
        )}
      </div>

      {/* Win probs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <ProbBar label={p.away_team} prob={p.away_win_prob} isPick={p.predicted_winner === p.away_team} />
        <ProbBar label={p.home_team} prob={p.home_win_prob} isPick={p.predicted_winner === p.home_team} />
      </div>

      {/* Odds row */}
      {(p.home_odds !== null || p.away_odds !== null) && (
        <div style={{ display: "flex", gap: 16, fontSize: 13, color: "#94a3b8", marginBottom: p.is_value_bet ? 10 : 0 }}>
          <span>{p.away_team} {fmtOdds(p.away_odds)}</span>
          <span>{p.home_team} {fmtOdds(p.home_odds)}</span>
          {p.market_home_prob !== null && (
            <span style={{ marginLeft: "auto" }}>
              Market: {fmtPct(p.market_home_prob)} / {fmtPct(1 - p.market_home_prob)}
            </span>
          )}
        </div>
      )}

      {/* Value bet highlight */}
      {p.is_value_bet && p.value_team && (
        <div style={{
          marginTop: 10,
          background: "#052e16",
          borderRadius: 8,
          padding: "8px 12px",
          display: "flex",
          alignItems: "center",
          gap: 10,
          fontSize: 13,
        }}>
          <span style={{ fontSize: 16 }}>★</span>
          <span style={{ color: "#22c55e", fontWeight: 700 }}>{p.value_team}</span>
          <span style={{ color: "#86efac" }}>{fmtOdds(p.value_odds)}</span>
          <span style={{ color: "#4ade80" }}>EV {fmtEv(p.value_ev)}</span>
          {resolved && (
            <span style={{
              marginLeft: "auto",
              fontWeight: 600,
              color: p.value_bet_correct ? "#22c55e" : "#ef4444",
            }}>
              {p.value_bet_correct ? "WIN ✓" : "LOSS ✗"}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function TeamName({ team, won, resolved }: { team: string; won: boolean; resolved: boolean }) {
  return (
    <span style={{
      color: resolved ? (won ? "#f1f5f9" : "#475569") : "#f1f5f9",
      fontWeight: won ? 800 : 600,
    }}>
      {team}
    </span>
  );
}

function ProbBar({ label, prob, isPick }: { label: string; prob: number; isPick: boolean }) {
  return (
    <div style={{ flex: 1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 12 }}>
        <span style={{ color: isPick ? "#f1f5f9" : "#64748b", fontWeight: isPick ? 700 : 400 }}>{label}</span>
        <span style={{ color: isPick ? "#38bdf8" : "#64748b", fontWeight: 600 }}>{(prob * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 4, background: "#0f172a", borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${prob * 100}%`,
          background: isPick ? "#38bdf8" : "#334155",
          borderRadius: 2,
        }} />
      </div>
    </div>
  );
}
