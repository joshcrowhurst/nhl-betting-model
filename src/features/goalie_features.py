"""
Starting-goalie features.

The team-level goalie feature (goalie_sv_pct_diff) averages whoever was in net
over the team's last 10 games, so it can't tell a starter from a backup. These
features use the goalie actually starting the game:

  - starter_sv_pct_diff: home starter's save% minus away starter's, over each
    goalie's own previous starts (up to LOOKBACK_STARTS, any team, any season),
    shrunk toward the league average so a backup with 3 starts isn't treated
    as elite or terrible.
  - home/away_starter_share_l10: share of the team's last 10 games this goalie
    started. Low = backup or newly acquired goalie.

Historical starters come from boxscores (the goalie flagged `starter`).
For today's games they come from src/data/starting_goalies.py.
Only starts strictly before the game's date are used.
"""

import numpy as np
import pandas as pd

LOOKBACK_STARTS = 40
SHRINK_SHOTS = 600.0          # ~20 games of shots at the league average
LEAGUE_SV_FALLBACK = 0.903
TEAM_WINDOW = 10


class GoalieHistory:
    """Per-goalie and per-team start logs, indexed for fast 'before date' lookups."""

    def __init__(self, games: pd.DataFrame):
        frames = []
        for side in ("home", "away"):
            cols = {f"{side}_goalie_id": "goalie_id", f"{side}_goalie_saves": "saves",
                    f"{side}_goalie_sa": "sa", f"{side}_team": "team"}
            if not all(c in games.columns for c in cols):
                continue
            f = games[["date", *cols]].rename(columns=cols)
            frames.append(f)
        starts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["date", "goalie_id", "saves", "sa", "team"])
        starts = starts.dropna(subset=["goalie_id", "sa"])
        starts = starts[starts["sa"] > 0]
        starts["goalie_id"] = starts["goalie_id"].astype("int64")
        starts = starts.sort_values("date", kind="stable").reset_index(drop=True)

        self._by_goalie = {gid: g for gid, g in starts.groupby("goalie_id")}
        self._by_team = {t: g for t, g in starts.groupby("team")}
        self._dates = starts["date"].values
        self._cum_saves = starts["saves"].cumsum().values
        self._cum_sa = starts["sa"].cumsum().values

    def _league_sv(self, before) -> float:
        n = np.searchsorted(self._dates, np.datetime64(before), side="left")
        if n == 0:
            return LEAGUE_SV_FALLBACK
        lo = max(0, n - 4000)  # roughly the last 1.5 seasons of starts
        saves = self._cum_saves[n - 1] - (self._cum_saves[lo - 1] if lo else 0)
        sa = self._cum_sa[n - 1] - (self._cum_sa[lo - 1] if lo else 0)
        return float(saves / sa) if sa else LEAGUE_SV_FALLBACK

    def starter_sv_pct(self, goalie_id, before) -> float:
        if goalie_id is None or pd.isna(goalie_id):
            return np.nan
        league = self._league_sv(before)
        g = self._by_goalie.get(int(goalie_id))
        if g is None:
            return league  # never started in our data: assume league average
        n = np.searchsorted(g["date"].values, np.datetime64(before), side="left")
        recent = g.iloc[max(0, n - LOOKBACK_STARTS):n]
        return float((recent["saves"].sum() + SHRINK_SHOTS * league) / (recent["sa"].sum() + SHRINK_SHOTS))

    def start_share(self, team, goalie_id, before) -> float:
        if goalie_id is None or pd.isna(goalie_id):
            return np.nan
        t = self._by_team.get(team)
        if t is None:
            return np.nan
        n = np.searchsorted(t["date"].values, np.datetime64(before), side="left")
        recent = t.iloc[max(0, n - TEAM_WINDOW):n]
        if recent.empty:
            return np.nan
        return float((recent["goalie_id"] == int(goalie_id)).mean())

    def usual_starter(self, team, before) -> int | None:
        """Goalie with the most starts in the team's last 10 games (ties: most recent)."""
        t = self._by_team.get(team)
        if t is None:
            return None
        n = np.searchsorted(t["date"].values, np.datetime64(before), side="left")
        recent = t.iloc[max(0, n - TEAM_WINDOW):n]
        if recent.empty:
            return None
        counts = recent.groupby("goalie_id").agg(n=("date", "size"), last=("date", "max"))
        return int(counts.sort_values(["n", "last"], ascending=False).index[0])


def starter_features(hist: GoalieHistory, row: pd.Series) -> dict:
    date = row["date"]
    home_id, away_id = row.get("home_starter_id"), row.get("away_starter_id")
    home_sv = hist.starter_sv_pct(home_id, date)
    away_sv = hist.starter_sv_pct(away_id, date)
    return {
        "home_starter_sv_pct": home_sv,
        "away_starter_sv_pct": away_sv,
        "starter_sv_pct_diff": home_sv - away_sv if not (np.isnan(home_sv) or np.isnan(away_sv)) else np.nan,
        "home_starter_share_l10": hist.start_share(row["home_team"], home_id, date),
        "away_starter_share_l10": hist.start_share(row["away_team"], away_id, date),
    }
