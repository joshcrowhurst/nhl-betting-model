"""
Fetches NHL moneyline odds from The Odds API (the-odds-api.com).
Requires ODDS_API_KEY environment variable (set in .env).
"""

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

import requests
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT, ODDS_REGIONS, ODDS_MARKETS, RAW_DIR

logger = logging.getLogger(__name__)

SESSION = requests.Session()

# ---------------------------------------------------------------------------
# Team name → NHL API abbreviation mapping
# Odds API returns full names; NHL API uses abbrevs
# ---------------------------------------------------------------------------
TEAM_NAME_TO_ABBREV = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "Seattle Kraken": "SEA",
    "San Jose Sharks": "SJS",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}

# Alternate spellings seen in odds feeds
TEAM_NAME_TO_ABBREV.update({
    "Montréal Canadiens": "MTL",
    "St Louis Blues": "STL",
    "Utah HC": "UTA",
})

ABBREV_TO_TEAM_NAME = {v: k for k, v in TEAM_NAME_TO_ABBREV.items()}

# Credits left on the API key, from the last response header (None until a call is made)
LAST_REMAINING: int | None = None


def name_to_abbrev(name: str) -> str | None:
    return TEAM_NAME_TO_ABBREV.get(name)


def _get(endpoint: str, params: dict = None) -> dict | list:
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY not set. Add it to your .env file.")
    params = params or {}
    params["apiKey"] = ODDS_API_KEY
    url = f"{ODDS_API_BASE}/{endpoint}"
    resp = SESSION.get(url, params=params, timeout=15)
    global LAST_REMAINING
    remaining = resp.headers.get("x-requests-remaining")
    if remaining:
        try:
            LAST_REMAINING = int(float(remaining))
        except ValueError:
            pass
        logger.info(f"Odds API credits remaining: {remaining} (last call used {resp.headers.get('x-requests-last')})")
    resp.raise_for_status()
    return resp.json()


def get_current_odds() -> pd.DataFrame:
    """Fetch live/upcoming NHL moneyline odds. Does NOT cache (live data)."""
    data = _get(
        f"sports/{ODDS_SPORT}/odds",
        params={"regions": ODDS_REGIONS, "markets": ODDS_MARKETS, "oddsFormat": "american"},
    )
    df = _parse_odds(data)
    return _add_abbrevs(df)


def get_historical_odds(at: datetime) -> pd.DataFrame:
    """
    Odds as they stood at `at` (UTC), from the historical endpoint.
    Paid plans only; each call costs 10 credits per region and market.
    Snapshots are cached on disk, so re-running a backfill is free.
    """
    ts = pd.Timestamp(at).tz_convert("UTC") if pd.Timestamp(at).tzinfo else pd.Timestamp(at).tz_localize("UTC")
    iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    cache_dir = RAW_DIR / "odds_hist"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{iso.replace(':', '')}.json"

    if cache_path.exists():
        data = json.loads(cache_path.read_text())
    else:
        data = _get(
            f"historical/sports/{ODDS_SPORT}/odds",
            params={"regions": ODDS_REGIONS, "markets": ODDS_MARKETS,
                    "oddsFormat": "american", "date": iso},
        )
        cache_path.write_text(json.dumps(data))
        time.sleep(0.2)

    raw = data.get("data", []) if isinstance(data, dict) else data
    df = _parse_odds(raw)
    if not df.empty:
        df["snapshot"] = data.get("timestamp", iso) if isinstance(data, dict) else iso
    return _add_abbrevs(df)


def get_consensus_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-bookmaker odds to consensus (median) per game."""
    if df.empty:
        return df
    grp_cols = ["event_id", "home_team", "away_team", "home_abbrev", "away_abbrev", "commence_time"]
    grp_cols = [c for c in grp_cols if c in df.columns]
    df = df.dropna(subset=["home_odds_american", "away_odds_american"])
    # Take the median of decimal odds, not American: American odds jump from
    # -100 to +100, so the median of e.g. -105 and +102 would be -1.5.
    df = df.assign(home_odds_decimal=df["home_odds_american"].map(american_to_decimal),
                   away_odds_decimal=df["away_odds_american"].map(american_to_decimal))
    out = (
        df.groupby(grp_cols)
        .agg(
            home_odds_median=("home_odds_decimal", "median"),
            away_odds_median=("away_odds_decimal", "median"),
            home_no_vig_prob=("home_no_vig_prob", "median"),
            away_no_vig_prob=("away_no_vig_prob", "median"),
            num_books=("bookmaker", "count"),
        )
        .reset_index()
    )
    for side in ("home", "away"):
        out[f"{side}_odds_median"] = out[f"{side}_odds_median"].map(decimal_to_american)
    # Best available price on each side (for American odds, higher is always
    # better for the bettor) and which bookmaker offers it.
    for side in ("home", "away"):
        idx = df.groupby(grp_cols)[f"{side}_odds_american"].idxmax()
        best = df.loc[idx, grp_cols + [f"{side}_odds_american", "bookmaker"]].rename(
            columns={f"{side}_odds_american": f"best_{side}_odds", "bookmaker": f"best_{side}_book"})
        out = out.merge(best, on=grp_cols, how="left")
    return out


def match_odds_to_games(consensus: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Join consensus odds onto a games DataFrame (from NHL API) by abbrev + date.
    games must have: home_team (abbrev), away_team (abbrev), date (Timestamp)
    Returns games with odds columns appended.
    """
    if consensus.empty:
        return games

    odds = consensus.copy()
    # Match on the Eastern-time date: NHL schedule dates are local, and a 10 PM ET
    # puck drop is already the next day in UTC.
    odds["date"] = (pd.to_datetime(odds["commence_time"], utc=True)
                    .dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None))
    # Drop full-name columns before renaming abbrevs to home_team/away_team
    odds = odds.drop(columns=["home_team", "away_team"], errors="ignore")
    odds = odds.rename(columns={
        "home_abbrev": "home_team",
        "away_abbrev": "away_team",
        "home_no_vig_prob": "market_home_prob",
        "away_no_vig_prob": "market_away_prob",
        "home_odds_median": "home_odds",
        "away_odds_median": "away_odds",
    })
    keep = ["date", "home_team", "away_team", "market_home_prob",
            "market_away_prob", "home_odds", "away_odds", "num_books",
            "best_home_odds", "best_home_book", "best_away_odds", "best_away_book", "commence_time"]
    odds = odds[[c for c in keep if c in odds.columns]]
    # Doubleheaders don't happen in the NHL, so date + teams identifies a game.
    odds = odds.drop_duplicates(subset=["date", "home_team", "away_team"])
    games = games.drop(columns=[c for c in odds.columns if c in games.columns
                                and c not in ("date", "home_team", "away_team")])

    return games.merge(odds, on=["date", "home_team", "away_team"], how="left")


def compute_ev(model_prob: float, odds_american: float) -> float | None:
    """
    Expected value per $1 wagered.
    EV > 0 = value bet. EV < 0 = edge to the house.
    """
    decimal = american_to_decimal(odds_american)
    if decimal is None:
        return None
    return round(model_prob * (decimal - 1) - (1 - model_prob), 4)


def format_odds_line(home_team: str, away_team: str, prediction: dict) -> str:
    """
    Format a single game prediction for display.
    Value bet = the side where model probability > market probability (positive EV).
    This may differ from the model's overall pick (the team with >50% win prob).
    """
    home_prob = prediction["home_win_prob"]
    away_prob = 1 - home_prob
    winner = prediction["predicted_winner"]

    line = f"  {away_team} @ {home_team}\n"
    line += f"    Model:  {home_team} {home_prob:.1%}  |  {away_team} {away_prob:.1%}\n"

    if prediction.get("market_home_prob"):
        mhp = prediction["market_home_prob"]
        map_ = 1 - mhp
        home_edge = home_prob - mhp
        away_edge = away_prob - map_
        home_odds = prediction.get("home_odds")
        away_odds = prediction.get("away_odds")

        line += f"    Market: {home_team} {mhp:.1%}  |  {away_team} {map_:.1%}\n"

        # Compute EV for both sides
        home_ev = compute_ev(home_prob, home_odds) if home_odds else None
        away_ev = compute_ev(away_prob, away_odds) if away_odds else None

        # Value side = whichever has positive EV (model edge > vig)
        # If both negative, show the less-negative side for info
        if home_ev is not None and away_ev is not None:
            if home_ev > 0:
                value_team, value_odds, value_edge, value_ev = home_team, home_odds, home_edge, home_ev
            elif away_ev > 0:
                value_team, value_odds, value_edge, value_ev = away_team, away_odds, away_edge, away_ev
            else:
                # No value — show both EVs as info
                line += (f"    EV:     {home_team} {_fmt_odds(home_odds)} {home_ev:+.3f}  |  "
                         f"{away_team} {_fmt_odds(away_odds)} {away_ev:+.3f}  (no value)\n")
                line += f"    Pick:   {winner}\n"
                return line

            edge_str = f"+{value_edge:.1%}" if value_edge > 0 else f"{value_edge:.1%}"
            ev_str = f"+{value_ev:.3f}" if value_ev > 0 else f"{value_ev:.3f}"
            value_tag = "  ★ VALUE BET" if value_ev > 0 else ""
            line += (f"    Bet:    {value_team} {_fmt_odds(value_odds)}  "
                     f"edge {edge_str}  EV {ev_str}{value_tag}\n")
        elif home_ev is not None:
            edge_str = f"+{home_edge:.1%}" if home_edge > 0 else f"{home_edge:.1%}"
            ev_str = f"+{home_ev:.3f}" if home_ev > 0 else f"{home_ev:.3f}"
            tag = "  ★ VALUE BET" if home_ev > 0 else ""
            line += f"    Bet:    {home_team} {_fmt_odds(home_odds)}  edge {edge_str}  EV {ev_str}{tag}\n"
    else:
        line += f"    Odds:   not available\n"

    line += f"    Pick:   {winner}\n"
    return line


def _fmt_odds(o: float) -> str:
    if o is None:
        return "N/A"
    return f"+{int(o)}" if o > 0 else str(int(o))


def _parse_odds(data: list) -> pd.DataFrame:
    rows = []
    for event in data:
        home = event.get("home_team")
        away = event.get("away_team")
        commence = event.get("commence_time")
        event_id = event.get("id")

        for bookmaker in event.get("bookmakers", []):
            book = bookmaker.get("key")
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                rows.append({
                    "event_id": event_id,
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "bookmaker": book,
                    "home_odds_american": outcomes.get(home),
                    "away_odds_american": outcomes.get(away),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"])
        df["home_implied_prob"] = df["home_odds_american"].apply(_american_to_prob)
        df["away_implied_prob"] = df["away_odds_american"].apply(_american_to_prob)
        total = df["home_implied_prob"] + df["away_implied_prob"]
        df["home_no_vig_prob"] = df["home_implied_prob"] / total
        df["away_no_vig_prob"] = df["away_implied_prob"] / total
    return df


def _add_abbrevs(df: pd.DataFrame) -> pd.DataFrame:
    """Add home_abbrev / away_abbrev columns using the name mapping."""
    if df.empty:
        return df
    df["home_abbrev"] = df["home_team"].map(TEAM_NAME_TO_ABBREV)
    df["away_abbrev"] = df["away_team"].map(TEAM_NAME_TO_ABBREV)
    unmapped = df[df["home_abbrev"].isna()]["home_team"].unique().tolist()
    if unmapped:
        logger.warning(f"Unmapped team names (update TEAM_NAME_TO_ABBREV): {unmapped}")
    return df


def _american_to_prob(odds: float) -> float:
    if odds is None or pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def american_to_decimal(odds: float) -> float:
    if odds is None or pd.isna(odds):
        return None
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def decimal_to_american(odds: float) -> float:
    if odds is None or pd.isna(odds):
        return None
    if odds >= 2:
        return round((odds - 1) * 100, 1)
    return round(-100 / (odds - 1), 1)
