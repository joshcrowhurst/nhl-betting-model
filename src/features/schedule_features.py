"""
Schedule-derived features:
  - Bounce-back motivation (previous game margin)
  - Schedule density (back-to-back, 3-in-4 nights)
  - Travel distance and fatigue (miles flown over rolling window)

All features computed strictly from prior games — no lookahead.
"""

import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Arena coordinates (lat, lon) for all 32 NHL franchises
# Keyed by team abbreviation used in the NHL API
# ---------------------------------------------------------------------------
ARENA_COORDS = {
    "ANA": (33.8078, -117.8764),   # Honda Center, Anaheim
    "BOS": (42.3662, -71.0621),    # TD Garden, Boston
    "BUF": (42.8750, -78.8764),    # KeyBank Center, Buffalo
    "CGY": (51.0375, -114.0519),   # Scotiabank Saddledome, Calgary
    "CAR": (35.8033, -78.7219),    # PNC Arena, Raleigh
    "CHI": (41.8807, -87.6742),    # United Center, Chicago
    "COL": (39.7489, -105.0077),   # Ball Arena, Denver
    "CBJ": (39.9692, -83.0061),    # Nationwide Arena, Columbus
    "DAL": (32.7905, -96.8103),    # American Airlines Center, Dallas
    "DET": (42.3411, -83.0548),    # Little Caesars Arena, Detroit
    "EDM": (53.5461, -113.4938),   # Rogers Place, Edmonton
    "FLA": (26.1583, -80.3256),    # Amerant Bank Arena, Sunrise
    "LAK": (34.0430, -118.2673),   # Crypto.com Arena, Los Angeles
    "MIN": (44.9447, -93.1014),    # Xcel Energy Center, St. Paul
    "MTL": (45.4960, -73.5694),    # Bell Centre, Montreal
    "NSH": (36.1593, -86.7786),    # Bridgestone Arena, Nashville
    "NJD": (40.7337, -74.1713),    # Prudential Center, Newark
    "NYI": (40.7223, -73.7236),    # UBS Arena, Elmont
    "NYR": (40.7505, -73.9934),    # Madison Square Garden, New York
    "OTT": (45.2969, -75.9278),    # Canadian Tire Centre, Ottawa
    "PHI": (39.9012, -75.1720),    # Wells Fargo Center, Philadelphia
    "PIT": (40.4395, -79.9892),    # PPG Paints Arena, Pittsburgh
    "SEA": (47.6222, -122.3544),   # Climate Pledge Arena, Seattle
    "SJS": (37.3329, -121.9011),   # SAP Center, San Jose
    "STL": (38.6267, -90.2025),    # Enterprise Center, St. Louis
    "TBL": (27.9428, -82.4519),    # Amalie Arena, Tampa
    "TOR": (43.6435, -79.3791),    # Scotiabank Arena, Toronto
    "UTA": (40.7683, -111.9011),   # Delta Center, Salt Lake City (Utah HC)
    "VAN": (49.2778, -123.1088),   # Rogers Arena, Vancouver
    "VGK": (36.1028, -115.1784),   # T-Mobile Arena, Las Vegas
    "WSH": (38.8981, -77.0209),    # Capital One Arena, Washington
    "WPG": (49.8928, -97.1437),    # Canada Life Centre, Winnipeg
    # Legacy abbreviations that may appear in older seasons
    "ARI": (33.5318, -112.2612),   # Gila River Arena / Mullett Arena, Arizona
    "PHX": (33.5318, -112.2612),
}


def haversine_miles(coord1: tuple, coord2: tuple) -> float:
    """Great-circle distance in miles between two (lat, lon) pairs."""
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(a))


def get_schedule_features(
    team: str,
    team_games: pd.DataFrame,
    game_date: pd.Timestamp,
) -> dict:
    """
    Compute all schedule-derived features for a team heading into a game.

    team_games: DataFrame from _team_games() — must include 'date', 'is_home',
                'opponent', 'gf', 'ga', 'won' columns, sorted by date ascending.
    game_date:  the date of the game being predicted (strictly excluded).
    """
    prior = team_games[team_games["date"] < game_date].sort_values("date")

    f = {}

    # ---- Bounce-back / motivation ----------------------------------------
    if prior.empty:
        f["prev_margin"] = 0.0
        f["blowout_loss_prev"] = 0
        f["avg_margin_l3"] = 0.0
    else:
        last = prior.iloc[-1]
        margin = float(last["gf"]) - float(last["ga"])
        f["prev_margin"] = margin
        f["blowout_loss_prev"] = int(margin <= -3)
        last3 = prior.tail(3)
        f["avg_margin_l3"] = float((last3["gf"] - last3["ga"]).mean())

    # ---- Schedule density / fatigue --------------------------------------
    last_7 = prior[prior["date"] >= game_date - pd.Timedelta(days=7)]
    last_4 = prior[prior["date"] >= game_date - pd.Timedelta(days=4)]
    last_2 = prior[prior["date"] >= game_date - pd.Timedelta(days=2)]

    f["games_last_7_days"] = len(last_7)
    f["games_last_4_days"] = len(last_4)
    f["is_b2b"] = int(
        not prior.empty and (game_date - prior.iloc[-1]["date"]).days <= 1
    )
    f["is_3in4"] = int(len(last_4) >= 2 and f["is_b2b"])

    # ---- Travel distance --------------------------------------------------
    f["travel_miles_last_7"] = _travel_miles(prior, team, game_date, days=7)
    f["direct_travel_miles"] = _direct_travel_to_game(prior, team, game_date)

    return f


def _current_arena(game_row: pd.Series, team: str) -> str:
    """Return the arena (team abbrev) where the team played in a given game."""
    return team if game_row["is_home"] else game_row.get("opponent", team)


def _travel_miles(prior: pd.DataFrame, team: str, game_date: pd.Timestamp, days: int) -> float:
    """Total miles traveled by a team in the last N days before game_date."""
    window = prior[prior["date"] >= game_date - pd.Timedelta(days=days)].sort_values("date")
    if len(window) < 1:
        return 0.0

    locations = [_current_arena(r, team) for _, r in window.iterrows()]
    # Add the team's home arena as the starting point if first game in window was at home
    # (approximation: we start from wherever they played first in the window)
    total = 0.0
    for i in range(1, len(locations)):
        a, b = locations[i - 1], locations[i]
        if a != b and a in ARENA_COORDS and b in ARENA_COORDS:
            total += haversine_miles(ARENA_COORDS[a], ARENA_COORDS[b])
    return round(total, 1)


def _direct_travel_to_game(prior: pd.DataFrame, team: str, game_date: pd.Timestamp) -> float:
    """
    Miles from the team's last game location to their current game location.
    For home teams this is 0 (or travel back home from road trip).
    We approximate: distance from last-game arena to home arena if they're
    returning home, else distance from last arena to away arena.
    This is computed as distance from last arena to home arena as a proxy —
    the actual destination is determined by the caller (home vs away).
    """
    if prior.empty:
        return 0.0
    last = prior.iloc[-1]
    last_arena = _current_arena(last, team)
    if last_arena not in ARENA_COORDS or team not in ARENA_COORDS:
        return 0.0
    return round(haversine_miles(ARENA_COORDS[last_arena], ARENA_COORDS[team]), 1)


def enrich_team_games_with_opponent(
    prior: pd.DataFrame,
    team: str,
) -> pd.DataFrame:
    """
    Add 'opponent' column to a team_games DataFrame so travel can be computed.
    home games: opponent = away_team; away games: opponent = home_team.
    """
    df = prior.copy()
    if "home_team" in df.columns and "away_team" in df.columns:
        df["opponent"] = np.where(df["home_team"] == team, df["away_team"], df["home_team"])
    return df
