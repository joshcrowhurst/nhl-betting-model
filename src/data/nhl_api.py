"""
Fetches game schedules and results from the NHL Stats API (api-web.nhle.com).
No API key required. Covers seasons from 2010-11 onward.
"""

import time
import logging
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NHL_API_BASE, RAW_DIR

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "nhl-betting-model/1.0"})


def _get(url: str, params: dict = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            logger.warning(f"Retry {attempt+1}/{retries} for {url}: {e}")
            time.sleep(2 ** attempt)


def get_season_schedule(season: str) -> pd.DataFrame:
    """
    season: '20232024' format
    Returns DataFrame with one row per regular-season game.

    Uses date-based pagination: starts at Oct 1 of the season's first year,
    follows nextStartDate until past May (end of playoffs).
    """
    cache_path = RAW_DIR / f"schedule_{season}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    year = int(season[:4])
    start_date = f"{year}-10-01"
    end_date = f"{year + 1}-05-31"

    logger.info(f"Fetching schedule for season {season} via date pagination")
    rows = []
    current_date = start_date

    while current_date and current_date <= end_date:
        try:
            data = _get(f"{NHL_API_BASE}/schedule/{current_date}")
        except Exception as e:
            logger.warning(f"Failed to fetch schedule for {current_date}: {e}")
            break

        for day in data.get("gameWeek", []):
            game_date = day.get("date")  # date lives on the day, not the game
            for game in day.get("games", []):
                if game.get("gameType") != 2:
                    continue
                if str(game.get("season", "")) != season:
                    continue
                rows.append({
                    "game_id": game["id"],
                    "season": season,
                    "date": game_date,
                    "home_team": game["homeTeam"]["abbrev"],
                    "away_team": game["awayTeam"]["abbrev"],
                    "home_score": game["homeTeam"].get("score"),
                    "away_score": game["awayTeam"].get("score"),
                    "game_state": game.get("gameState"),
                    "period": game.get("periodDescriptor", {}).get("number"),
                })

        next_date = data.get("nextStartDate")
        if not next_date or next_date == current_date:
            break
        current_date = next_date
        time.sleep(0.1)  # be polite

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset="game_id")
        df["date"] = pd.to_datetime(df["date"])
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)
        df.loc[df["home_score"].isna(), "home_win"] = None
        df.to_parquet(cache_path, index=False)
        logger.info(f"Saved {len(df)} regular season games for {season}")
    else:
        logger.warning(f"No games found for season {season}")
    return df


def get_games_for_date(target_date: date, game_types: tuple = (2, 3)) -> pd.DataFrame:
    """
    Fetch all games on a specific date, including playoffs (gameType 3).
    Does NOT cache — used for daily prediction runs.
    game_types: (2,) = regular season only, (2,3) = reg + playoffs
    """
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        data = _get(f"{NHL_API_BASE}/schedule/{date_str}")
    except Exception as e:
        logger.warning(f"Could not fetch schedule for {date_str}: {e}")
        return pd.DataFrame()

    rows = []
    for day in data.get("gameWeek", []):
        game_date = day.get("date")
        if game_date != date_str:   # only the requested date, not the full week
            continue
        for game in day.get("games", []):
            if game.get("gameType") not in game_types:
                continue
            rows.append({
                "game_id": game["id"],
                "season": str(game.get("season", "")),
                "date": game_date,
                "game_type": game.get("gameType"),
                "home_team": game["homeTeam"]["abbrev"],
                "away_team": game["awayTeam"]["abbrev"],
                "home_score": game["homeTeam"].get("score"),
                "away_score": game["awayTeam"].get("score"),
                "game_state": game.get("gameState"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)
        df.loc[df["home_score"].isna(), "home_win"] = None
    return df


def get_boxscore(game_id: int) -> Optional[dict]:
    """Returns raw boxscore dict for a single game."""
    cache_path = RAW_DIR / f"boxscore_{game_id}.json"
    if cache_path.exists():
        import json
        return json.loads(cache_path.read_text())

    url = f"{NHL_API_BASE}/gamecenter/{game_id}/boxscore"
    try:
        data = _get(url)
        import json
        cache_path.write_text(json.dumps(data))
        return data
    except Exception as e:
        logger.warning(f"Could not fetch boxscore {game_id}: {e}")
        return None


def get_team_stats_from_boxscore(game_id: int) -> Optional[dict]:
    """Extracts per-team summary stats from a boxscore."""
    raw = get_boxscore(game_id)
    if raw is None:
        return None

    out = {}
    for side in ("homeTeam", "awayTeam"):
        team = raw.get(side, {})
        key = "home" if side == "homeTeam" else "away"
        out[f"{key}_sog"] = team.get("sog")
        out[f"{key}_pim"] = team.get("pim")
        out[f"{key}_hits"] = team.get("hits")
        out[f"{key}_blocked"] = team.get("blockedShots")
        out[f"{key}_faceoff_pct"] = team.get("faceoffWinningPctg")
        out[f"{key}_pp_pct"] = team.get("powerPlayPctg")
        out[f"{key}_pk_pct"] = team.get("penaltyKillPctg")
    return out


def get_multiple_seasons(seasons: list[str]) -> pd.DataFrame:
    """Load and concatenate schedules for multiple seasons."""
    frames = []
    for s in seasons:
        try:
            df = get_season_schedule(s)
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed to load season {s}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_goalie_stats(season: str) -> pd.DataFrame:
    """Fetch goalie stats summary for a season."""
    cache_path = RAW_DIR / f"goalies_{season}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = f"{NHL_API_BASE}/skater-stats-leaders/{season}/2"
    # Goalies come from a different endpoint
    url = f"{NHL_API_BASE}/goalie-stats-leaders/{season}/2?categories=wins&limit=-1"
    try:
        data = _get(url)
        rows = []
        for g in data.get("data", []):
            rows.append({
                "season": season,
                "goalie_id": g.get("playerId"),
                "name": g.get("fullName"),
                "team": g.get("teamAbbrevs"),
                "gp": g.get("gamesPlayed"),
                "sv_pct": g.get("savePctg"),
                "gaa": g.get("goalsAgainstAverage"),
                "wins": g.get("wins"),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        logger.warning(f"Could not fetch goalie stats for {season}: {e}")
        return pd.DataFrame()


def current_season_code() -> str:
    """Returns current season code like '20242025'."""
    today = date.today()
    year = today.year
    if today.month >= 9:
        return f"{year}{year+1}"
    return f"{year-1}{year}"


def season_range(start: str, end: str = None) -> list[str]:
    """
    Generate list of season codes from start to end (inclusive).
    start/end: '20182019' format
    """
    def _year(s): return int(s[:4])
    if end is None:
        end = current_season_code()
    return [f"{y}{y+1}" for y in range(_year(start), _year(end) + 1)]
