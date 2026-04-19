import React, { useEffect, useState } from "react";
import { api } from "./api";
import { Prediction, Performance } from "./types";
import GameCard from "./components/GameCard";
import StatsBar from "./components/StatsBar";
import TrendChart from "./components/TrendChart";

type Tab = "today" | "recent" | "performance";

function Header({ tab, setTab }: { tab: Tab; setTab: (t: Tab) => void }) {
  return (
    <div style={{
      background: "#0f172a",
      borderBottom: "1px solid #1e293b",
      position: "sticky",
      top: 0,
      zIndex: 10,
    }}>
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "0 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 4, paddingTop: 14 }}>
          <span style={{ fontSize: 20, marginRight: 8 }}>🏒</span>
          <span style={{ fontWeight: 800, fontSize: 17, color: "#f1f5f9", letterSpacing: -0.3 }}>
            NHL Model
          </span>
        </div>
        <div style={{ display: "flex", gap: 0, marginTop: 12 }}>
          {(["today", "recent", "performance"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: "10px 16px",
                fontSize: 14,
                fontWeight: 600,
                color: tab === t ? "#38bdf8" : "#64748b",
                borderBottom: tab === t ? "2px solid #38bdf8" : "2px solid transparent",
                textTransform: "capitalize",
              }}
            >
              {t === "today" ? "Today" : t === "recent" ? "Recent" : "Performance"}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function LoadingSpinner() {
  return (
    <div style={{ textAlign: "center", padding: "60px 0", color: "#64748b" }}>
      Loading...
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div style={{
      textAlign: "center", padding: "60px 0",
      color: "#475569", fontSize: 15,
    }}>
      {message}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 28 }}>
      <h2 style={{ fontSize: 13, fontWeight: 600, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 12 }}>
        {title}
      </h2>
      {children}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("today");
  const [today, setToday] = useState<Prediction[] | null>(null);
  const [recent, setRecent] = useState<Prediction[] | null>(null);
  const [perf, setPerf] = useState<Performance | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);

    const fetch = async () => {
      try {
        if (tab === "today" && today === null) {
          setToday(await api.getToday());
        } else if (tab === "recent" && recent === null) {
          setRecent(await api.getRecent(14));
        } else if (tab === "performance" && perf === null) {
          setPerf(await api.getPerformance());
        }
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    fetch();
  }, [tab]); // eslint-disable-line

  const content = () => {
    if (loading) return <LoadingSpinner />;
    if (error) return <EmptyState message={`Error: ${error}`} />;

    if (tab === "today") {
      if (!today || today.length === 0) return <EmptyState message="No predictions for today yet." />;
      const valueBets = today.filter((p) => p.is_value_bet);
      return (
        <>
          {valueBets.length > 0 && (
            <Section title={`★ ${valueBets.length} value bet${valueBets.length > 1 ? "s" : ""} today`}>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {valueBets.map((p) => <GameCard key={p.game_id} p={p} />)}
              </div>
            </Section>
          )}
          <Section title={`All games (${today.length})`}>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {today.filter((p) => !p.is_value_bet).map((p) => <GameCard key={p.game_id} p={p} />)}
            </div>
          </Section>
        </>
      );
    }

    if (tab === "recent") {
      if (!recent || recent.length === 0) return <EmptyState message="No recent predictions." />;
      const grouped: Record<string, Prediction[]> = {};
      recent.forEach((p) => {
        grouped[p.game_date] = [...(grouped[p.game_date] || []), p];
      });
      return (
        <>
          {Object.entries(grouped)
            .sort(([a], [b]) => b.localeCompare(a))
            .map(([date, preds]) => {
              const correct = preds.filter((p) => p.correct).length;
              const resolved = preds.filter((p) => p.actual_home_win !== null).length;
              const d = new Date(date + "T12:00:00");
              const label = d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
              return (
                <Section key={date} title={`${label}${resolved > 0 ? `  ·  ${correct}/${resolved}` : ""}`}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {preds.map((p) => <GameCard key={p.game_id} p={p} />)}
                  </div>
                </Section>
              );
            })}
        </>
      );
    }

    if (tab === "performance") {
      if (!perf) return <EmptyState message="No performance data yet. Resolve some predictions first." />;
      return (
        <>
          <StatsBar perf={perf} />
          <Section title="Daily accuracy trend">
            <div style={{ background: "#1e293b", borderRadius: 12, padding: "16px 8px" }}>
              <TrendChart data={perf.daily_breakdown ?? []} />
            </div>
          </Section>
        </>
      );
    }
  };

  return (
    <div style={{ minHeight: "100vh" }}>
      <Header tab={tab} setTab={setTab} />
      <main style={{ maxWidth: 720, margin: "0 auto", padding: "24px 16px" }}>
        {content()}
      </main>
    </div>
  );
}
