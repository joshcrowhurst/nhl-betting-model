"""
Daily pipeline run by GitHub Actions (see .github/workflows/daily.yml).

Steps, each safe to re-run:
  resolve  - fill in results for finished games we predicted
  retrain  - retrain the model on 6 seasons (weekly, or when no usable model exists)
  predict  - predict today's not-yet-started games and email them

All dates are US Eastern, which is how the NHL schedules games.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from config import MODELS_DIR, ODDS_API_KEY
from src.data.nhl_api import get_multiple_seasons, season_range, current_season_code, get_games_for_date
from src.data.boxscore_enricher import get_enriched_game_stats
from src.data.odds_api import get_current_odds, get_consensus_odds, match_odds_to_games, compute_ev
from src.features.feature_engineer import build_features, get_feature_cols
from src.models.moneyline_model import MoneylineModel
from src.pipeline import store

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MODEL_PATH = MODELS_DIR / "moneyline_latest.pkl"
HISTORY_SEASONS = 3
TRAIN_SEASONS = 6
RESOLVE_LOOKBACK_DAYS = 14
NOT_STARTED = {"FUT", "PRE"}
FINISHED = {"OFF", "FINAL"}

# Optional boxscore features that may be missing; XGBoost handles their NaNs.
OPTIONAL_PREFIXES = (
    "home_goalie", "away_goalie", "goalie_", "home_shot", "away_shot", "shot_ratio",
    "home_faceoff", "away_faceoff", "home_hits", "away_hits", "market_",
)


def today_et() -> date:
    return datetime.now(ET).date()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_history(num_seasons: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    current = current_season_code()
    start_year = int(current[:4]) - num_seasons + 1
    seasons = season_range(f"{start_year}{start_year + 1}", current)
    games = get_multiple_seasons(seasons)
    frames = []
    for season in seasons:
        if games.empty:
            break
        enriched = get_enriched_game_stats(games[games["season"] == season], season)
        if not enriched.empty:
            frames.append(enriched)
    enriched = pd.concat(frames, ignore_index=True) if frames else None
    return games, enriched


# ---------------------------------------------------------------------------
# Retrain
# ---------------------------------------------------------------------------

def retrain() -> MoneylineModel:
    games, enriched = _load_history(TRAIN_SEASONS)
    features = build_features(games, enriched=enriched)
    base_cols = [c for c in get_feature_cols(include_market=False) if not c.startswith(OPTIONAL_PREFIXES)]
    valid = features.dropna(subset=["home_win"]).dropna(subset=base_cols)

    model = MoneylineModel(include_market=False)
    metrics = model.train(valid, valid["home_win"], calibrate=True)
    model.save(MODEL_PATH)
    store.log_run("retrain", "success", games_trained=len(valid), metrics=metrics)
    logger.info(f"Retrained on {len(valid)} games: {metrics}")
    return model


def load_or_train_model() -> MoneylineModel:
    if MODEL_PATH.exists():
        try:
            return MoneylineModel.load(MODEL_PATH)
        except Exception as e:  # e.g. pickle from an incompatible library version
            logger.warning(f"Saved model unusable ({e}); retraining")
    return retrain()


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------

def resolve() -> int:
    preds = store.load_predictions()
    pending = preds[preds["actual_home_win"].isna()]
    cutoff = (today_et() - timedelta(days=RESOLVE_LOOKBACK_DAYS)).isoformat()
    dates = sorted(d for d in pending["game_date"].unique() if d >= cutoff and d <= today_et().isoformat())

    resolved = 0
    for d in dates:
        day = get_games_for_date(date.fromisoformat(d), game_types=(2, 3))
        if day.empty:
            continue
        finished = day[day["game_state"].isin(FINISHED)].set_index("game_id")
        for idx in pending.index[pending["game_date"] == d]:
            gid = int(preds.at[idx, "game_id"])
            if gid not in finished.index:
                continue
            g = finished.loc[gid]
            home_won = bool(g["home_score"] > g["away_score"])
            preds.at[idx, "home_score"] = int(g["home_score"])
            preds.at[idx, "away_score"] = int(g["away_score"])
            preds.at[idx, "actual_home_win"] = int(home_won)
            preds.at[idx, "correct"] = (preds.at[idx, "home_win_prob"] > 0.5) == home_won
            if isinstance(preds.at[idx, "value_team"], str):
                value_is_home = preds.at[idx, "value_team"] == preds.at[idx, "home_team"]
                preds.at[idx, "value_bet_correct"] = home_won if value_is_home else not home_won
            preds.at[idx, "resolved_at"] = _now_iso()
            resolved += 1

    if resolved:
        store.save_predictions(preds)
    store.log_run("resolve", "success", resolved=resolved)
    logger.info(f"Resolved {resolved} game(s)")
    return resolved


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def _fetch_odds() -> pd.DataFrame | None:
    if not ODDS_API_KEY:
        logger.info("ODDS_API_KEY not set — predicting without odds")
        return None
    try:
        return get_consensus_odds(get_current_odds())
    except Exception as e:
        logger.warning(f"Odds fetch failed — predicting without odds: {e}")
        return None


def _sf(val) -> float | None:
    try:
        return None if val is None or pd.isna(val) else float(val)
    except (TypeError, ValueError):
        return None


def predict(model: MoneylineModel, game_date: date | None = None) -> pd.DataFrame:
    """Predict not-yet-started games on game_date. Returns the new prediction rows."""
    game_date = game_date or today_et()
    preds = store.load_predictions()
    already = set(preds["game_id"].astype(int))

    todays = get_games_for_date(game_date, game_types=(2, 3))
    if todays.empty:
        store.log_run("predict", "success", date=str(game_date), predicted=0, note="no games")
        logger.info(f"No games on {game_date}")
        return pd.DataFrame(columns=store.COLUMNS)

    targets = todays[todays["game_state"].isin(NOT_STARTED) & ~todays["game_id"].isin(already)].copy()
    if targets.empty:
        store.log_run("predict", "success", date=str(game_date), predicted=0, note="nothing new")
        logger.info("All of today's games are already predicted or under way")
        return pd.DataFrame(columns=store.COLUMNS)

    games, enriched = _load_history(HISTORY_SEASONS)
    day_ts = pd.Timestamp(game_date)
    prior = games[games["date"] < day_ts]
    combined = pd.concat([prior, targets.assign(home_win=0.0)], ignore_index=True)
    features = build_features(combined, enriched=enriched, target_ids=targets["game_id"])

    consensus = _fetch_odds()
    if consensus is not None and not consensus.empty:
        targets = match_odds_to_games(consensus, targets.assign(date=day_ts))

    rows = []
    for _, game in targets.iterrows():
        feat = features[features["game_id"] == game["game_id"]] if not features.empty else features
        if feat.empty:
            logger.warning(f"Skipping {game['game_id']} — not enough history")
            continue
        p_home = float(model.predict_proba(feat)[0])
        home_odds, away_odds = _sf(game.get("home_odds")), _sf(game.get("away_odds"))
        home_ev = compute_ev(p_home, home_odds) if home_odds else None
        away_ev = compute_ev(1 - p_home, away_odds) if away_odds else None

        value_team = value_odds = value_ev = None
        if home_ev is not None and home_ev > 0 and (away_ev is None or home_ev >= away_ev):
            value_team, value_odds, value_ev = game["home_team"], home_odds, home_ev
        elif away_ev is not None and away_ev > 0:
            value_team, value_odds, value_ev = game["away_team"], away_odds, away_ev

        rows.append({
            "game_id": int(game["game_id"]),
            "game_date": game_date.isoformat(),
            "season": str(game.get("season")),
            "game_type": int(game.get("game_type", 2)),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_win_prob": round(p_home, 4),
            "away_win_prob": round(1 - p_home, 4),
            "predicted_winner": game["home_team"] if p_home > 0.5 else game["away_team"],
            "market_home_prob": _sf(game.get("market_home_prob")),
            "home_odds": home_odds,
            "away_odds": away_odds,
            "home_ev": home_ev,
            "away_ev": away_ev,
            "is_value_bet": value_team is not None,
            "value_team": value_team,
            "value_odds": value_odds,
            "value_ev": value_ev,
            "logged_at": _now_iso(),
        })

    new = pd.DataFrame(rows, columns=store.COLUMNS)
    if not new.empty:
        store.save_predictions(pd.concat([preds, new], ignore_index=True) if not preds.empty else new)
    store.log_run("predict", "success", date=str(game_date), predicted=len(new),
                  odds=consensus is not None and not consensus.empty)
    logger.info(f"Predicted {len(new)} game(s) for {game_date}")
    return new


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(tasks: set[str], send_email: bool = True, force_email: bool = False) -> None:
    failures = []

    def step(name, fn):
        try:
            return fn()
        except Exception as e:
            logger.exception(f"{name} failed")
            store.log_run(name, "error", error=str(e))
            failures.append(name)
            return None

    if "resolve" in tasks:
        step("resolve", resolve)

    model = None
    if "retrain" in tasks:
        model = step("retrain", retrain)
    if "predict" in tasks:
        model = model or step("load_model", load_or_train_model)
        new = step("predict", lambda: predict(model)) if model is not None else None
        if send_email and ((new is not None and not new.empty) or force_email):
            from src.pipeline.emailer import send_daily_email
            step("email", lambda: send_daily_email(today_et()))

    if failures:
        raise SystemExit(f"Failed steps: {', '.join(failures)}")
