"""
Builds the feature matrix from raw game schedule data.
All features are computed strictly from games PRIOR to the target game
to prevent lookahead bias.

Feature groups:
  - Rolling win rate (last N games)
  - Rolling goals for/against
  - Rolling goalie save% (from boxscore enrichment)
  - Rolling shot ratio (proxy for possession/Corsi)
  - Rolling hits, blocked shots, faceoff%
  - Elo ratings (accumulated team strength)
  - Home/away splits
  - Rest days
  - Head-to-head record
  - Market implied probability (when odds available)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ROLLING_WINDOW_GAMES, MIN_GAMES_FOR_FEATURES
from src.features.schedule_features import get_schedule_features, enrich_team_games_with_opponent

logger = logging.getLogger(__name__)

ELO_BASE = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 50.0  # home team gets this added to their Elo for expected win calc

FEATURE_COLS = [
    # Elo — single feature encoding accumulated team strength + home advantage
    "elo_home_win_prob",

    # Goalie matchup — net save% advantage
    "goalie_sv_pct_diff",

    # Possession — shot share differential (proxy for Corsi)
    "shot_ratio_diff",

    # Recent goal differential — short-term form (last 10), complements Elo's long-term view
    "home_gd_per_game_l10",
    "away_gd_per_game_l10",

    # Venue-split form — how each team performs in their specific role
    "home_win_rate_home_l10",
    "away_win_rate_away_l10",

    # Bounce-back / momentum
    "margin_momentum_diff",       # home 3-game avg margin minus away
    "home_blowout_loss_prev",
    "away_blowout_loss_prev",

    # Head-to-head history
    "h2h_home_win_rate",

    # Faceoff — zone time and possession proxy
    "home_faceoff_pct_l10",
    "away_faceoff_pct_l10",

    # Rest and schedule fatigue
    "rest_advantage",
    "home_is_b2b",
    "away_is_b2b",
    "home_is_3in4",
    "away_is_3in4",
    "away_games_last_7_days",

    # Travel — direct distance flown by away team to reach this game
    "away_direct_travel_miles",

    # Market (optional — NaN when no odds)
    "market_home_prob",
]


def build_features(
    games: pd.DataFrame,
    enriched: pd.DataFrame = None,
    odds: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    games:    schedule DataFrame from nhl_api — one row per completed game
    enriched: optional boxscore stats from boxscore_enricher — merged by game_id
    odds:     optional consensus odds DataFrame

    Returns feature DataFrame with target column 'home_win'.
    """
    if games.empty or "date" not in games.columns:
        logger.warning("build_features: empty or malformed games DataFrame")
        return pd.DataFrame()

    games = games.copy()
    games = games.sort_values("date").reset_index(drop=True)
    games = games[games["home_win"].notna()].copy()
    games["home_win"] = games["home_win"].astype(int)

    # Merge enriched boxscore data if provided
    if enriched is not None and not enriched.empty:
        games = games.merge(enriched, on="game_id", how="left")
    else:
        for col in ["home_sog", "away_sog", "home_goalie_sv_pct", "away_goalie_sv_pct",
                    "home_faceoff_pct", "away_faceoff_pct", "home_hits", "away_hits"]:
            games[col] = np.nan

    # Compute Elo ratings in chronological order (mutates a dict in-place)
    elo_ratings = _compute_elo_series(games)

    feature_rows = []
    for idx, row in games.iterrows():
        prior = games[games["date"] < row["date"]]
        features = _build_row_features(row, prior, elo_ratings)
        if features is not None:
            features["game_id"] = row["game_id"]
            features["date"] = row["date"]
            features["season"] = row["season"]
            features["home_team"] = row["home_team"]
            features["away_team"] = row["away_team"]
            features["home_win"] = row["home_win"]
            feature_rows.append(features)

    df = pd.DataFrame(feature_rows)
    if df.empty:
        return df

    if odds is not None and not odds.empty:
        df = _merge_odds(df, odds)
    if "market_home_prob" not in df.columns:
        df["market_home_prob"] = np.nan

    return df.reset_index(drop=True)


def _build_row_features(row: pd.Series, prior: pd.DataFrame, elo_ratings: dict) -> dict | None:
    home = row["home_team"]
    away = row["away_team"]

    home_games = _team_games(prior, home)
    away_games = _team_games(prior, away)

    if len(home_games) < MIN_GAMES_FOR_FEATURES or len(away_games) < MIN_GAMES_FOR_FEATURES:
        return None

    # Add opponent column for travel calculations (already have home_team/away_team from prior slice)
    home_games_rich = enrich_team_games_with_opponent(home_games, home)
    away_games_rich = enrich_team_games_with_opponent(away_games, away)

    home_l10 = home_games.tail(ROLLING_WINDOW_GAMES)
    away_l10 = away_games.tail(ROLLING_WINDOW_GAMES)
    home_home_l10 = home_games[home_games["is_home"]].tail(ROLLING_WINDOW_GAMES)
    away_away_l10 = away_games[~away_games["is_home"]].tail(ROLLING_WINDOW_GAMES)

    def safe_mean(s):
        s = s.dropna()
        return float(s.mean()) if len(s) > 0 else np.nan

    f = {
        # Rolling win/goals
        "home_win_rate_l10": safe_mean(home_l10["won"]),
        "home_gf_per_game_l10": safe_mean(home_l10["gf"]),
        "home_ga_per_game_l10": safe_mean(home_l10["ga"]),
        "home_gd_per_game_l10": safe_mean(home_l10["gf"] - home_l10["ga"]),
        "home_win_rate_home_l10": safe_mean(home_home_l10["won"]) if len(home_home_l10) >= 3 else np.nan,
        "home_gf_home_l10": safe_mean(home_home_l10["gf"]) if len(home_home_l10) >= 3 else np.nan,
        "home_ga_home_l10": safe_mean(home_home_l10["ga"]) if len(home_home_l10) >= 3 else np.nan,

        "away_win_rate_l10": safe_mean(away_l10["won"]),
        "away_gf_per_game_l10": safe_mean(away_l10["gf"]),
        "away_ga_per_game_l10": safe_mean(away_l10["ga"]),
        "away_gd_per_game_l10": safe_mean(away_l10["gf"] - away_l10["ga"]),
        "away_win_rate_away_l10": safe_mean(away_away_l10["won"]) if len(away_away_l10) >= 3 else np.nan,
        "away_gf_away_l10": safe_mean(away_away_l10["gf"]) if len(away_away_l10) >= 3 else np.nan,
        "away_ga_away_l10": safe_mean(away_away_l10["ga"]) if len(away_away_l10) >= 3 else np.nan,

        "home_rest_days": _rest_days(home_games, row["date"]),
        "away_rest_days": _rest_days(away_games, row["date"]),
    }

    f["win_rate_diff"] = f["home_win_rate_l10"] - f["away_win_rate_l10"]
    f["gf_diff"] = f["home_gf_per_game_l10"] - f["away_gf_per_game_l10"]
    f["ga_diff"] = f["home_ga_per_game_l10"] - f["away_ga_per_game_l10"]
    f["gd_diff"] = f["home_gd_per_game_l10"] - f["away_gd_per_game_l10"]
    f["rest_advantage"] = f["home_rest_days"] - f["away_rest_days"]

    # H2H
    h2h = prior[
        ((prior["home_team"] == home) & (prior["away_team"] == away)) |
        ((prior["home_team"] == away) & (prior["away_team"] == home))
    ].tail(10)
    if len(h2h) >= 2:
        home_h2h_wins = (
            ((h2h["home_team"] == home) & (h2h["home_win"] == 1)) |
            ((h2h["away_team"] == home) & (h2h["home_win"] == 0))
        ).sum()
        f["h2h_home_win_rate"] = home_h2h_wins / len(h2h)
    else:
        f["h2h_home_win_rate"] = np.nan

    # Elo (pre-game ratings)
    game_key = row["game_id"]
    elo_snapshot = elo_ratings.get(game_key, {})
    home_elo = elo_snapshot.get(home, ELO_BASE)
    away_elo = elo_snapshot.get(away, ELO_BASE)
    f["home_elo"] = home_elo
    f["away_elo"] = away_elo
    f["elo_diff"] = home_elo - away_elo
    f["elo_home_win_prob"] = _elo_win_prob(home_elo + ELO_HOME_ADVANTAGE, away_elo)

    # Goalie save% rolling
    f["home_goalie_sv_pct_l10"] = safe_mean(home_l10["goalie_sv_pct"]) if "goalie_sv_pct" in home_l10.columns else np.nan
    f["away_goalie_sv_pct_l10"] = safe_mean(away_l10["goalie_sv_pct"]) if "goalie_sv_pct" in away_l10.columns else np.nan
    f["goalie_sv_pct_diff"] = (
        f["home_goalie_sv_pct_l10"] - f["away_goalie_sv_pct_l10"]
        if not (np.isnan(f["home_goalie_sv_pct_l10"]) or np.isnan(f["away_goalie_sv_pct_l10"]))
        else np.nan
    )

    # Shot ratio rolling
    f["home_shot_ratio_l10"] = safe_mean(home_l10["shot_ratio"]) if "shot_ratio" in home_l10.columns else np.nan
    f["away_shot_ratio_l10"] = safe_mean(away_l10["shot_ratio"]) if "shot_ratio" in away_l10.columns else np.nan
    f["shot_ratio_diff"] = (
        f["home_shot_ratio_l10"] - f["away_shot_ratio_l10"]
        if not (np.isnan(f["home_shot_ratio_l10"]) or np.isnan(f["away_shot_ratio_l10"]))
        else np.nan
    )

    # Physical / possession
    f["home_faceoff_pct_l10"] = safe_mean(home_l10["faceoff_pct"]) if "faceoff_pct" in home_l10.columns else np.nan
    f["away_faceoff_pct_l10"] = safe_mean(away_l10["faceoff_pct"]) if "faceoff_pct" in away_l10.columns else np.nan
    f["home_hits_per_game_l10"] = safe_mean(home_l10["hits"]) if "hits" in home_l10.columns else np.nan
    f["away_hits_per_game_l10"] = safe_mean(away_l10["hits"]) if "hits" in away_l10.columns else np.nan

    # Schedule features (bounce-back, fatigue, travel)
    game_date = row["date"]
    hsf = get_schedule_features(home, home_games_rich, game_date)
    asf = get_schedule_features(away, away_games_rich, game_date)

    f["home_prev_margin"]        = hsf["prev_margin"]
    f["away_prev_margin"]        = asf["prev_margin"]
    f["home_blowout_loss_prev"]  = hsf["blowout_loss_prev"]
    f["away_blowout_loss_prev"]  = asf["blowout_loss_prev"]
    f["home_avg_margin_l3"]      = hsf["avg_margin_l3"]
    f["away_avg_margin_l3"]      = asf["avg_margin_l3"]
    f["margin_momentum_diff"]    = hsf["avg_margin_l3"] - asf["avg_margin_l3"]

    f["home_is_b2b"]             = hsf["is_b2b"]
    f["away_is_b2b"]             = asf["is_b2b"]
    f["home_is_3in4"]            = hsf["is_3in4"]
    f["away_is_3in4"]            = asf["is_3in4"]
    f["home_games_last_7_days"]  = hsf["games_last_7_days"]
    f["away_games_last_7_days"]  = asf["games_last_7_days"]
    f["b2b_advantage"]           = asf["is_b2b"] - hsf["is_b2b"]

    f["home_travel_miles_last_7"]  = hsf["travel_miles_last_7"]
    f["away_travel_miles_last_7"]  = asf["travel_miles_last_7"]
    f["away_direct_travel_miles"]  = asf["direct_travel_miles"]
    f["travel_fatigue_diff"]       = asf["travel_miles_last_7"] - hsf["travel_miles_last_7"]

    return f


def _team_games(prior: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return all prior games for a team with perspective columns."""
    home = prior[prior["home_team"] == team].copy()
    home["gf"] = home["home_score"]
    home["ga"] = home["away_score"]
    home["won"] = home["home_win"].astype(int)
    home["is_home"] = True
    if "home_goalie_sv_pct" in prior.columns:
        home["goalie_sv_pct"] = home["home_goalie_sv_pct"]
    if "home_sog" in prior.columns and "away_sog" in prior.columns:
        total = home["home_sog"] + home["away_sog"]
        home["shot_ratio"] = home["home_sog"] / total.replace(0, np.nan)
    if "home_faceoff_pct" in prior.columns:
        home["faceoff_pct"] = home["home_faceoff_pct"]
    if "home_hits" in prior.columns:
        home["hits"] = home["home_hits"]

    away = prior[prior["away_team"] == team].copy()
    away["gf"] = away["away_score"]
    away["ga"] = away["home_score"]
    away["won"] = (1 - away["home_win"]).astype(int)
    away["is_home"] = False
    if "away_goalie_sv_pct" in prior.columns:
        away["goalie_sv_pct"] = away["away_goalie_sv_pct"]
    if "home_sog" in prior.columns and "away_sog" in prior.columns:
        total = away["home_sog"] + away["away_sog"]
        away["shot_ratio"] = away["away_sog"] / total.replace(0, np.nan)
    if "away_faceoff_pct" in prior.columns:
        away["faceoff_pct"] = away["away_faceoff_pct"]
    if "away_hits" in prior.columns:
        away["hits"] = away["away_hits"]

    return pd.concat([home, away], ignore_index=True).sort_values("date")


def _compute_elo_series(games: pd.DataFrame) -> dict:
    """
    Walk through games chronologically and record each team's Elo BEFORE each game.
    Returns dict: {game_id: {team_abbrev: elo_rating}}
    """
    ratings = {}  # team -> current Elo
    snapshots = {}  # game_id -> {team: elo before game}

    for _, row in games.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_elo = ratings.get(home, ELO_BASE)
        away_elo = ratings.get(away, ELO_BASE)

        snapshots[row["game_id"]] = {home: home_elo, away: away_elo}

        expected_home = _elo_win_prob(home_elo + ELO_HOME_ADVANTAGE, away_elo)
        actual_home = float(row["home_win"])

        ratings[home] = home_elo + ELO_K * (actual_home - expected_home)
        ratings[away] = away_elo + ELO_K * ((1 - actual_home) - (1 - expected_home))

    return snapshots


def _elo_win_prob(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _rest_days(team_games: pd.DataFrame, game_date: pd.Timestamp) -> float:
    if team_games.empty:
        return 7.0
    last_game = team_games["date"].max()
    delta = (game_date - last_game).days
    return min(float(delta), 10.0)


def _merge_odds(features: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    odds = odds.copy()
    odds["date"] = pd.to_datetime(odds["commence_time"]).dt.normalize()
    odds = odds.rename(columns={"home_no_vig_prob": "market_home_prob"})
    odds = odds[["date", "home_team", "away_team", "market_home_prob"]]
    return features.merge(odds, on=["date", "home_team", "away_team"], how="left")


def get_feature_cols(include_market: bool = True) -> list[str]:
    cols = [c for c in FEATURE_COLS if c != "market_home_prob"]
    if include_market:
        cols.append("market_home_prob")
    return cols
