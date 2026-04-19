"""
Batch-fetches boxscores for a list of game IDs and extracts team-level stats.
Results are cached to a single parquet per season to avoid thousands of tiny files.

Stats extracted per game:
  - home/away shots on goal (sog)
  - home/away goalie save% (starter only)
  - home/away goalie saves, shots against
  - home/away hits, blocked shots, faceoff win%
  - home/away power play goals, power play shots against (from goalie data)
  - outcome type: REG / OT / SO
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NHL_API_BASE, RAW_DIR

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "nhl-betting-model/1.0"})

BATCH_CACHE_DIR = RAW_DIR / "boxscore_batches"
BATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_enriched_game_stats(schedule_df: pd.DataFrame, season: str, delay: float = 0.15) -> pd.DataFrame:
    """
    Given a schedule DataFrame for one season, fetches boxscores for all completed
    games and returns a DataFrame of per-game team stats.

    Caches the entire season to one parquet file — only fetches missing games.
    """
    cache_path = BATCH_CACHE_DIR / f"enriched_{season}.parquet"

    completed = schedule_df[schedule_df["game_state"] == "OFF"]["game_id"].tolist()
    if not completed:
        return pd.DataFrame()

    # Load existing cache
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        already_fetched = set(cached["game_id"].tolist())
    else:
        cached = pd.DataFrame()
        already_fetched = set()

    missing = [gid for gid in completed if gid not in already_fetched]
    logger.info(f"Season {season}: {len(already_fetched)} cached, {len(missing)} to fetch")

    if not missing:
        return cached

    new_rows = []
    for i, game_id in enumerate(missing):
        row = _fetch_boxscore_stats(game_id)
        if row:
            new_rows.append(row)
        if (i + 1) % 50 == 0:
            logger.info(f"  Fetched {i+1}/{len(missing)} boxscores...")
        time.sleep(delay)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df
        combined.to_parquet(cache_path, index=False)
        logger.info(f"Saved enriched stats for {len(combined)} games ({season})")
        return combined

    return cached


def _fetch_boxscore_stats(game_id: int) -> dict | None:
    url = f"{NHL_API_BASE}/gamecenter/{game_id}/boxscore"
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug(f"Boxscore fetch failed for {game_id}: {e}")
        return None

    row = {"game_id": game_id}

    # Team-level SOG
    for side, key in (("homeTeam", "home"), ("awayTeam", "away")):
        team = data.get(side, {})
        row[f"{key}_sog"] = team.get("sog")

    # Goalie stats (starter only)
    pbg = data.get("playerByGameStats", {})
    for side, key in (("homeTeam", "home"), ("awayTeam", "away")):
        goalies = pbg.get(side, {}).get("goalies", [])
        starter = next((g for g in goalies if g.get("starter")), None)
        if starter:
            row[f"{key}_goalie_sv_pct"] = starter.get("savePctg")
            row[f"{key}_goalie_saves"] = starter.get("saves")
            row[f"{key}_goalie_sa"] = starter.get("shotsAgainst")
            row[f"{key}_goalie_ga"] = starter.get("goalsAgainst")
            row[f"{key}_goalie_toi"] = _toi_to_minutes(starter.get("toi", "0:00"))
            # PP from goalie data
            _parse_pp_shots(starter, key, row)
        else:
            for col in ["goalie_sv_pct", "goalie_saves", "goalie_sa", "goalie_ga", "goalie_toi", "pp_shots_against"]:
                row[f"{key}_{col}"] = None

    # Skater aggregates: hits, blocked shots, faceoff %, giveaways, takeaways, PP goals
    for side, key in (("homeTeam", "home"), ("awayTeam", "away")):
        forwards = pbg.get(side, {}).get("forwards", [])
        defense = pbg.get(side, {}).get("defense", [])
        skaters = forwards + defense
        if skaters:
            row[f"{key}_hits"] = sum(s.get("hits", 0) or 0 for s in skaters)
            row[f"{key}_blocked"] = sum(s.get("blockedShots", 0) or 0 for s in skaters)
            row[f"{key}_giveaways"] = sum(s.get("giveaways", 0) or 0 for s in skaters)
            row[f"{key}_takeaways"] = sum(s.get("takeaways", 0) or 0 for s in skaters)
            row[f"{key}_pp_goals"] = sum(s.get("powerPlayGoals", 0) or 0 for s in skaters)
            faceoffs = [s.get("faceoffWinningPctg") for s in skaters if s.get("faceoffWinningPctg") is not None]
            row[f"{key}_faceoff_pct"] = sum(faceoffs) / len(faceoffs) if faceoffs else None
        else:
            for col in ["hits", "blocked", "giveaways", "takeaways", "pp_goals", "faceoff_pct"]:
                row[f"{key}_{col}"] = None

    # Outcome type
    outcome = data.get("gameOutcome", {})
    row["outcome_type"] = outcome.get("lastPeriodType", "REG")  # REG, OT, SO

    return row


def _parse_pp_shots(goalie: dict, key: str, row: dict) -> None:
    """Parse 'powerPlayShotsAgainst' string like '2/3' → shots against on PP."""
    raw = goalie.get("powerPlayShotsAgainst", "")
    if raw and "/" in str(raw):
        parts = str(raw).split("/")
        try:
            row[f"{key}_pp_shots_against"] = int(parts[1])
        except (ValueError, IndexError):
            row[f"{key}_pp_shots_against"] = None
    else:
        row[f"{key}_pp_shots_against"] = None


def _toi_to_minutes(toi: str) -> float:
    """Convert 'MM:SS' to float minutes."""
    try:
        parts = toi.split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except Exception:
        return 0.0


def load_enriched_stats(seasons: list[str]) -> pd.DataFrame:
    """Load cached enriched stats for multiple seasons."""
    frames = []
    for season in seasons:
        path = BATCH_CACHE_DIR / f"enriched_{season}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
