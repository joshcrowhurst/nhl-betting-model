"""
Today's starting goalies.

Primary source: Daily Faceoff's starting-goalies page, which marks each starter
Confirmed / Likely / Unconfirmed. It's a Next.js page, so the data sits in the
__NEXT_DATA__ JSON blob; we search that blob for game objects instead of
hard-coding a key path, since the layout can change.

Names are mapped to NHL player IDs via each team's current roster.

Fallback (or when a team isn't found): the team's usual starter — the goalie
with the most starts in its last 10 games — marked "projected".
"""

import json
import logging
import re
import unicodedata
from datetime import date

import pandas as pd
import requests

from config import NHL_API_BASE
from src.data.odds_api import TEAM_NAME_TO_ABBREV
from src.features.goalie_features import GoalieHistory

logger = logging.getLogger(__name__)

DFO_URL = "https://www.dailyfaceoff.com/starting-goalies/{date}"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal NHL model; once-daily fetch)"}
STATUS_WORDS = ("confirmed", "likely", "expected", "unconfirmed", "projected")

# Daily Faceoff uses full names / slugs; add the extra spellings we might see.
_TEAM_LOOKUP = {**{k.lower(): v for k, v in TEAM_NAME_TO_ABBREV.items()},
                "utah mammoth": "UTA", "utah hc": "UTA", "montréal canadiens": "MTL"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def _team_abbrev(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    if v.upper() in set(TEAM_NAME_TO_ABBREV.values()):
        return v.upper()
    v = v.lower().replace("-", " ")
    if v in _TEAM_LOOKUP:
        return _TEAM_LOOKUP[v]
    for name, abbrev in _TEAM_LOOKUP.items():  # slugs like "calgary-flames"
        if _norm(name) == _norm(v):
            return abbrev
    return None


def _status(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.lower()
    for w in STATUS_WORDS:
        if w in v:
            return "likely" if w == "expected" else w
    return None


def _find_games(node, found: list) -> None:
    """Collect dicts that look like one game: home/away goalie name keys."""
    if isinstance(node, dict):
        keys = {k.lower(): k for k in node}
        hg = next((keys[k] for k in keys if "home" in k and "goalie" in k and "name" in k), None)
        ag = next((keys[k] for k in keys if "away" in k and "goalie" in k and "name" in k), None)
        if hg and ag:
            found.append(node)
        for v in node.values():
            _find_games(v, found)
    elif isinstance(node, list):
        for v in node:
            _find_games(v, found)


def _pick(node: dict, side: str, *must: str, exclude: tuple = ()):
    for k, v in node.items():
        kl = k.lower()
        if kl.startswith(side) and all(m in kl for m in must) and not any(e in kl for e in exclude):
            return v
    return None


def fetch_daily_faceoff(game_date: date) -> list[dict]:
    """Returns [{home_team, away_team, home_name, away_name, home_status, away_status}]."""
    resp = requests.get(DFO_URL.format(date=game_date.isoformat()), headers=HEADERS, timeout=20)
    resp.raise_for_status()
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text, re.S)
    if not m:
        raise ValueError("Daily Faceoff page has no __NEXT_DATA__ blob")
    found: list = []
    _find_games(json.loads(m.group(1)), found)

    out = []
    for g in found:
        home_team = (_team_abbrev(_pick(g, "home", "team", "abbrev")) or _team_abbrev(_pick(g, "home", "team", "name"))
                     or _team_abbrev(_pick(g, "home", "team", "slug")))
        away_team = (_team_abbrev(_pick(g, "away", "team", "abbrev")) or _team_abbrev(_pick(g, "away", "team", "name"))
                     or _team_abbrev(_pick(g, "away", "team", "slug")))
        if not (home_team and away_team):
            continue
        out.append({
            "home_team": home_team, "away_team": away_team,
            "home_name": _pick(g, "home", "goalie", "name"), "away_name": _pick(g, "away", "goalie", "name"),
            "home_status": _status(_pick(g, "home", "strength", "name") or _pick(g, "home", "status")),
            "away_status": _status(_pick(g, "away", "strength", "name") or _pick(g, "away", "status")),
        })
    logger.info(f"Daily Faceoff: {len(out)} game(s) parsed from {len(found)} candidate object(s)")
    return out


def _roster_goalies(team: str) -> dict[str, tuple[int, str]]:
    """normalized full name / last name -> (player_id, display name)."""
    data = requests.get(f"{NHL_API_BASE}/roster/{team}/current", timeout=15).json()
    out = {}
    for g in data.get("goalies", []):
        first = (g.get("firstName") or {}).get("default", "")
        last = (g.get("lastName") or {}).get("default", "")
        entry = (int(g["id"]), f"{first} {last}".strip())
        out[_norm(f"{first}{last}")] = entry
        out.setdefault(_norm(last), entry)
    return out


def _resolve_id(name: str, team: str, rosters: dict) -> tuple[int, str] | None:
    if not name:
        return None
    if team not in rosters:
        try:
            rosters[team] = _roster_goalies(team)
        except Exception as e:
            logger.warning(f"Roster fetch failed for {team}: {e}")
            rosters[team] = {}
    r = rosters[team]
    full = _norm(name)
    last = _norm(name.split()[-1])
    return r.get(full) or r.get(last)


def get_starters(game_date: date, todays: pd.DataFrame, hist: GoalieHistory, names: dict[int, str]) -> pd.DataFrame:
    """
    todays: today's games (game_id, home_team, away_team).
    hist:   GoalieHistory built from completed games (for the fallback).
    names:  goalie_id -> display name from boxscore history.
    Returns one row per game with {home,away}_starter_{id,name,status}.
    """
    try:
        dfo = {(g["home_team"], g["away_team"]): g for g in fetch_daily_faceoff(game_date)}
    except Exception as e:
        logger.warning(f"Daily Faceoff unavailable, using usual starters: {e}")
        dfo = {}

    rosters: dict = {}
    day = pd.Timestamp(game_date)
    rows = []
    for _, game in todays.iterrows():
        row = {"game_id": game["game_id"]}
        g = dfo.get((game["home_team"], game["away_team"]))
        for side in ("home", "away"):
            team = game[f"{side}_team"]
            gid = name = status = None
            if g and g.get(f"{side}_name"):
                hit = _resolve_id(g[f"{side}_name"], team, rosters)
                if hit:
                    gid, name = hit
                    status = g.get(f"{side}_status") or "unconfirmed"
                else:
                    logger.warning(f"Couldn't match {g[f'{side}_name']} to a {team} goalie")
            if gid is None:
                gid = hist.usual_starter(team, day)
                name = names.get(gid) if gid else None
                status = "projected" if gid else None
            row.update({f"{side}_starter_id": gid, f"{side}_starter_name": name, f"{side}_starter_status": status})
        rows.append(row)
    return pd.DataFrame(rows)
