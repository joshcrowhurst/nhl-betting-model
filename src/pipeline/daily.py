"""
Daily pipeline run by GitHub Actions (see .github/workflows/daily.yml).

Steps, each safe to re-run:
  resolve  - fill in results for finished games we predicted
  retrain  - retrain the model on 6 seasons (weekly, or when no usable model exists)
  predict  - predict today's not-yet-started games and email them

All dates are US Eastern, which is how the NHL schedules games.
"""

import logging
import os
import warnings
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from config import MODELS_DIR, ODDS_API_KEY
from src.data.nhl_api import get_multiple_seasons, season_range, current_season_code, get_games_for_date
from src.data.boxscore_enricher import get_enriched_game_stats
from src.data.odds_api import get_current_odds, get_consensus_odds, match_odds_to_games
from src.data.starting_goalies import get_starters
from src.features.feature_engineer import build_features, get_feature_cols, OPTIONAL_FEATURE_PREFIXES, FEATURE_VERSION
from src.features.goalie_features import GoalieHistory
from src.models.moneyline_model import MoneylineModel
from src.models.explain import contributions, rationale
from src.pipeline import store, pricing
from src.models import blend

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MODEL_PATH = MODELS_DIR / "moneyline_latest.pkl"
HISTORY_SEASONS = 3
TRAIN_SEASONS = 6
RESOLVE_LOOKBACK_DAYS = 14
NOT_STARTED = {"FUT", "PRE"}
FINISHED = {"OFF", "FINAL"}



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
    base_cols = [c for c in get_feature_cols(include_market=False) if not c.startswith(OPTIONAL_FEATURE_PREFIXES)]
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
            model = MoneylineModel.load(MODEL_PATH)
            if (model.feature_cols == get_feature_cols(include_market=False)
                    and getattr(model, "feature_version", 1) == FEATURE_VERSION):
                return model
            logger.info("Saved model was trained on a different feature set; retraining")
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
            for team_col, out_col in (("value_team", "value_bet_correct"), ("shop_team", "shop_bet_correct")):
                if isinstance(preds.at[idx, team_col], str):
                    on_home = preds.at[idx, team_col] == preds.at[idx, "home_team"]
                    preds.at[idx, out_col] = home_won if on_home else not home_won
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


def _iso(val) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return pd.Timestamp(val).tz_convert("UTC").isoformat() if pd.Timestamp(val).tzinfo else str(val)


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

    # Tonight's starters (Daily Faceoff, else each team's usual starter)
    prior_box = prior.merge(enriched, on="game_id", how="left") if enriched is not None else prior
    names = {}
    for side in ("home", "away"):
        if f"{side}_goalie_id" in prior_box.columns:
            pairs = prior_box[[f"{side}_goalie_id", f"{side}_goalie_name"]].dropna()
            names.update(dict(zip(pairs.iloc[:, 0].astype("int64"), pairs.iloc[:, 1])))
    starters = get_starters(game_date, targets, GoalieHistory(prior_box), names)
    targets = targets.merge(starters, on="game_id", how="left")
    combined = pd.concat([prior, targets.assign(home_win=0.0)], ignore_index=True)
    features = build_features(combined, enriched=enriched, target_ids=targets["game_id"])

    consensus = _fetch_odds()
    if consensus is not None and not consensus.empty:
        targets = match_odds_to_games(consensus, targets.assign(date=day_ts))

    contribs = contributions(model, features) if not features.empty else None
    coefs = blend.load(pricing.BLEND_PATH)

    rows = []
    for _, game in targets.iterrows():
        feat = features[features["game_id"] == game["game_id"]] if not features.empty else features
        if feat.empty:
            logger.warning(f"Skipping {game['game_id']} — not enough history")
            continue
        p_home = float(model.predict_proba(feat)[0])
        favoured = game["home_team"] if p_home > 0.5 else game["away_team"]
        try:
            names = {side: game.get(f"{side}_starter_name") for side in ("home", "away")
                     if isinstance(game.get(f"{side}_starter_name"), str)}
            why = rationale(feat.iloc[0], contribs.loc[feat.index[0]], game["home_team"], game["away_team"],
                            favoured, names)
        except Exception as e:  # an explanation must never block a prediction
            logger.warning(f"Rationale failed for {game['game_id']}: {e}")
            why = None
        priced = pricing.price(p_home, game.to_dict(), coefs)

        rows.append({
            "game_id": int(game["game_id"]),
            "game_date": game_date.isoformat(),
            "season": str(game.get("season")),
            "game_type": int(game.get("game_type", 2)),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_win_prob": round(p_home, 4),
            "away_win_prob": round(1 - p_home, 4),
            "predicted_winner": favoured,
            "rationale": why,
            **priced,
            "odds_source": "live" if priced["home_odds"] is not None else None,
            "commence_time": _iso(game.get("commence_time")),
            **{f"{side}_starter{suffix}": game.get(f"{side}_starter{suffix}")
               for side in ("home", "away") for suffix in ("_name", "_status")},
            "logged_at": _now_iso(),
        })

    new = pd.DataFrame(rows, columns=store.COLUMNS)
    if not new.empty:
        with warnings.catch_warnings():  # all-empty columns (e.g. no odds yet) are expected
            warnings.simplefilter("ignore", FutureWarning)
            combined = pd.concat([preds, new], ignore_index=True) if not preds.empty else new
        store.save_predictions(combined)
    store.log_run("predict", "success", date=str(game_date), predicted=len(new),
                  odds=consensus is not None and not consensus.empty)
    logger.info(f"Predicted {len(new)} game(s) for {game_date}")
    return new


# ---------------------------------------------------------------------------
# Close: record odds shortly before puck drop
# ---------------------------------------------------------------------------

def close(game_date: date | None = None) -> int:
    """
    Snapshot current odds for today's predicted games that haven't started, as
    their closing line. Run shortly before puck drop (Cloud Scheduler: 6:40 PM
    and 9:40 PM ET); a later snapshot overwrites an earlier one, so each game
    keeps the last price before it started. Comparing the price at pick time
    with the close (closing-line value) is the quickest test of a real edge.
    """
    game_date = game_date or today_et()
    preds = store.load_predictions()
    mask = preds["game_date"] == game_date.isoformat()
    if not mask.any():
        store.log_run("close", "success", updated=0, note="no predictions today")
        return 0

    todays = get_games_for_date(game_date, game_types=(2, 3))
    not_started = set(todays.loc[todays["game_state"].isin(NOT_STARTED), "game_id"]) if not todays.empty else set()
    idxs = [i for i in preds.index[mask] if int(preds.at[i, "game_id"]) in not_started]
    if not idxs:
        store.log_run("close", "success", updated=0, note="all games started")
        return 0

    consensus = _fetch_odds()
    if consensus is None or consensus.empty:
        store.log_run("close", "success", updated=0, note="no odds")
        return 0
    games = preds.loc[idxs, ["game_id", "home_team", "away_team"]].assign(date=pd.Timestamp(game_date))
    matched = match_odds_to_games(consensus, games).set_index("game_id")

    updated = 0
    now = _now_iso()
    for i in idxs:
        gid = preds.at[i, "game_id"]
        if gid not in matched.index or pd.isna(matched.at[gid, "home_odds"]):
            continue
        m = matched.loc[gid]
        preds.at[i, "close_home_odds"] = float(m["home_odds"])
        preds.at[i, "close_away_odds"] = float(m["away_odds"])
        preds.at[i, "close_home_prob"] = round(float(m["market_home_prob"]), 4)
        preds.at[i, "close_at"] = now
        updated += 1
    store.save_predictions(preds)
    store.log_run("close", "success", updated=updated)
    logger.info(f"Recorded closing odds for {updated} game(s)")
    return updated


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

    if "close" in tasks:
        step("close", close)

    # Historical odds (paid Odds API plan). ODDS_MAX_CREDITS caps what one run may spend.
    max_credits = int(os.getenv("ODDS_MAX_CREDITS", "12000"))
    if "backfill" in tasks:
        from src.pipeline.market import backfill_predictions
        step("backfill", lambda: backfill_predictions(max_credits))
    if "market" in tasks:
        from src.pipeline.market import compare_market
        step("market", lambda: compare_market(max_credits=max_credits))

    if failures:
        raise SystemExit(f"Failed steps: {', '.join(failures)}")
