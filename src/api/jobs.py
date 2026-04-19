"""
Job logic invoked by Cloud Scheduler via POST /jobs/{predict|resolve|retrain}.
Each job is idempotent — safe to re-run if Cloud Scheduler retries.
"""

import logging
import os
import pickle
import tempfile
from datetime import date, datetime

import pandas as pd
from google.cloud import storage

from src.database.connection import get_db
from src.database.models import ModelRun, Prediction, PerformanceCache
from src.data.nhl_api import get_multiple_seasons, season_range, current_season_code, get_games_for_date
from src.data.boxscore_enricher import get_enriched_game_stats
from src.data.odds_api import get_current_odds, get_consensus_odds, match_odds_to_games, compute_ev
from src.features.feature_engineer import build_features
from src.models.moneyline_model import MoneylineModel

logger = logging.getLogger(__name__)

GCS_BUCKET = os.getenv("GCS_BUCKET", "nhl-betting-model")
MODEL_BLOB = os.getenv("MODEL_BLOB", "models/moneyline_latest.pkl")


# ---------------------------------------------------------------------------
# Model storage helpers
# ---------------------------------------------------------------------------

def load_model_from_gcs() -> MoneylineModel:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(MODEL_BLOB)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        blob.download_to_filename(f.name)
        with open(f.name, "rb") as fh:
            model = pickle.load(fh)
    logger.info(f"Model loaded from gs://{GCS_BUCKET}/{MODEL_BLOB}")
    return model


def save_model_to_gcs(model: MoneylineModel) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(model, f)
        tmp_path = f.name
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(MODEL_BLOB)
    blob.upload_from_filename(tmp_path)
    logger.info(f"Model saved to gs://{GCS_BUCKET}/{MODEL_BLOB}")


# ---------------------------------------------------------------------------
# Fetch helpers (reused across jobs)
# ---------------------------------------------------------------------------

def _fetch_games(num_seasons: int = 3):
    current = current_season_code()
    start_year = int(current[:4]) - num_seasons + 1
    seasons = season_range(f"{start_year}{start_year+1}", current)
    df = get_multiple_seasons(seasons)
    return df, seasons


def _fetch_enriched(df: pd.DataFrame, seasons: list) -> pd.DataFrame | None:
    frames = []
    for season in seasons:
        season_df = df[df["season"] == season]
        enriched = get_enriched_game_stats(season_df, season)
        if not enriched.empty:
            frames.append(enriched)
    return pd.concat(frames, ignore_index=True) if frames else None


# ---------------------------------------------------------------------------
# PREDICT job
# ---------------------------------------------------------------------------

def run_predict_job() -> dict:
    started = datetime.utcnow()
    log_run("predict", "running")

    try:
        model = load_model_from_gcs()
        df, seasons = _fetch_games(num_seasons=3)
        enriched = _fetch_enriched(df, seasons)

        today = date.today()
        today_ts = pd.Timestamp(today)
        today_games = get_games_for_date(today, game_types=(2, 3))

        if today_games.empty:
            logger.info("No games today — predict job done early")
            log_run("predict", "success", games_processed=0,
                    details={"note": "no games today"})
            return {"status": "success", "games": 0}

        df = pd.concat(
            [df, today_games[~today_games["game_id"].isin(df["game_id"])]],
            ignore_index=True
        )

        # Fetch odds
        consensus = None
        try:
            raw_odds = get_current_odds()
            consensus = get_consensus_odds(raw_odds)
        except Exception as e:
            logger.warning(f"Odds fetch failed: {e}")

        # Match odds to today's games
        prior_games = df[df["date"] < today_ts].copy()
        todays = df[df["date"] == today_ts].copy()
        if consensus is not None and not consensus.empty:
            todays["date"] = today_ts
            todays = match_odds_to_games(consensus, todays)

        saved = 0
        predictions_for_email = []

        with get_db() as db:
            for _, game in todays.iterrows():
                # Skip if already predicted today
                existing = db.query(Prediction).filter(
                    Prediction.game_id == int(game["game_id"])
                ).first()
                if existing:
                    continue

                game_row = game.copy()
                game_row["home_win"] = 0.0
                combined = pd.concat([prior_games, pd.DataFrame([game_row])], ignore_index=True)
                features = build_features(combined, enriched=enriched)
                game_features = features[features["game_id"] == game["game_id"]]
                if game_features.empty:
                    logger.warning(f"Skipping {game['game_id']} — insufficient history")
                    continue

                prob = float(model.predict_proba(game_features)[0])
                away_prob = 1 - prob

                market_home_prob = _sf(game.get("market_home_prob"))
                home_odds = _sf(game.get("home_odds"))
                away_odds = _sf(game.get("away_odds"))
                home_ev = compute_ev(prob, home_odds) if home_odds else None
                away_ev = compute_ev(away_prob, away_odds) if away_odds else None

                # Determine value bet side
                is_value = bool(
                    (home_ev is not None and home_ev > 0) or
                    (away_ev is not None and away_ev > 0)
                )
                if home_ev and home_ev > 0:
                    value_team, value_odds, value_ev = game["home_team"], home_odds, home_ev
                elif away_ev and away_ev > 0:
                    value_team, value_odds, value_ev = game["away_team"], away_odds, away_ev
                else:
                    value_team = value_odds = value_ev = None

                p = Prediction(
                    game_id=int(game["game_id"]),
                    game_date=today,
                    season=game.get("season"),
                    game_type=int(game.get("game_type", 2)),
                    home_team=game["home_team"],
                    away_team=game["away_team"],
                    home_win_prob=round(prob, 4),
                    away_win_prob=round(away_prob, 4),
                    predicted_winner=game["home_team"] if prob > 0.5 else game["away_team"],
                    market_home_prob=market_home_prob,
                    home_odds=home_odds,
                    away_odds=away_odds,
                    home_ev=home_ev,
                    away_ev=away_ev,
                    is_value_bet=is_value,
                    value_team=value_team,
                    value_odds=value_odds,
                    value_ev=value_ev,
                )
                db.add(p)
                saved += 1
                predictions_for_email.append(p)

        log_run("predict", "success", games_processed=saved)
        logger.info(f"Predict job complete — {saved} games saved")

        # Send email (import here to avoid circular)
        if predictions_for_email:
            from src.api.email_sender import send_predictions_email
            send_predictions_email(predictions_for_email, today)

        return {"status": "success", "games": saved}

    except Exception as e:
        logger.exception(f"Predict job failed: {e}")
        log_run("predict", "error", details={"error": str(e)})
        raise


# ---------------------------------------------------------------------------
# RESOLVE job
# ---------------------------------------------------------------------------

def run_resolve_job() -> dict:
    log_run("resolve", "running")

    try:
        df, _ = _fetch_games(num_seasons=1)
        completed = df[df["home_win"].notna()].set_index("game_id")

        resolved = 0
        with get_db() as db:
            unresolved = db.query(Prediction).filter(
                Prediction.actual_home_win.is_(None)
            ).all()

            for pred in unresolved:
                if pred.game_id in completed.index:
                    actual = int(completed.loc[pred.game_id, "home_win"])
                    pred.actual_home_win = actual
                    pred.resolved_at = datetime.utcnow()
                    pred.correct = (pred.home_win_prob > 0.5) == bool(actual)
                    if pred.value_team:
                        home_won = bool(actual)
                        value_is_home = pred.value_team == pred.home_team
                        pred.value_bet_correct = home_won if value_is_home else not home_won
                    resolved += 1

            if resolved > 0:
                _update_performance_cache(db)

        log_run("resolve", "success", games_processed=resolved)
        return {"status": "success", "resolved": resolved}

    except Exception as e:
        logger.exception(f"Resolve job failed: {e}")
        log_run("resolve", "error", details={"error": str(e)})
        raise


# ---------------------------------------------------------------------------
# RETRAIN job
# ---------------------------------------------------------------------------

def run_retrain_job() -> dict:
    log_run("retrain", "running")

    try:
        df, seasons = _fetch_games(num_seasons=6)
        enriched = _fetch_enriched(df, seasons)
        features = build_features(df, enriched=enriched)

        from src.features.feature_engineer import get_feature_cols
        feat_cols = get_feature_cols(include_market=False)
        base_cols = [c for c in feat_cols if not c.startswith((
            "home_goalie", "away_goalie", "goalie_", "home_shot",
            "away_shot", "shot_ratio", "home_faceoff", "away_faceoff",
            "home_hits", "away_hits", "market_"
        ))]
        valid = features.dropna(subset=["home_win"]).dropna(subset=base_cols)

        model = MoneylineModel(include_market=False)
        metrics = model.train(valid, valid["home_win"], calibrate=True)
        save_model_to_gcs(model)

        log_run("retrain", "success", games_processed=len(valid),
                details={"metrics": metrics})
        return {"status": "success", "games_trained": len(valid), "metrics": metrics}

    except Exception as e:
        logger.exception(f"Retrain job failed: {e}")
        log_run("retrain", "error", details={"error": str(e)})
        raise


# ---------------------------------------------------------------------------
# Performance cache
# ---------------------------------------------------------------------------

def _update_performance_cache(db) -> None:
    all_preds = db.query(Prediction).all()
    resolved = [p for p in all_preds if p.actual_home_win is not None]
    value_bets = [p for p in resolved if p.is_value_bet and p.value_bet_correct is not None]

    if not resolved:
        return

    accuracy = sum(1 for p in resolved if p.correct) / len(resolved)
    vb_correct = sum(1 for p in value_bets if p.value_bet_correct)
    vb_accuracy = vb_correct / len(value_bets) if value_bets else None

    # Simple flat-stake ROI: each bet stakes $1, positive odds pay decimal-1
    roi = None
    if value_bets:
        from src.data.odds_api import american_to_decimal
        total_stake = len(value_bets)
        total_return = sum(
            american_to_decimal(p.value_odds) if p.value_bet_correct else 0
            for p in value_bets
        )
        roi = (total_return - total_stake) / total_stake

    # Daily breakdown for trend
    by_date = {}
    for p in resolved:
        d = str(p.game_date)
        if d not in by_date:
            by_date[d] = {"date": d, "correct": 0, "total": 0, "vb_correct": 0, "vb_total": 0}
        by_date[d]["total"] += 1
        if p.correct:
            by_date[d]["correct"] += 1
        if p.is_value_bet:
            by_date[d]["vb_total"] += 1
            if p.value_bet_correct:
                by_date[d]["vb_correct"] += 1
    daily = sorted(by_date.values(), key=lambda x: x["date"])

    cache = PerformanceCache(
        total_predictions=len(all_preds),
        resolved_predictions=len(resolved),
        correct_predictions=sum(1 for p in resolved if p.correct),
        accuracy=round(accuracy, 4),
        total_value_bets=len(value_bets),
        value_bets_correct=vb_correct,
        value_bet_accuracy=round(vb_accuracy, 4) if vb_accuracy else None,
        value_bet_roi=round(roi, 4) if roi else None,
        total_ev=round(sum(p.value_ev for p in value_bets if p.value_ev), 4),
        daily_breakdown=daily,
    )
    db.add(cache)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_run(run_type: str, status: str, games_processed: int = 0, details: dict = None):
    try:
        with get_db() as db:
            run = ModelRun(
                run_type=run_type,
                status=status,
                games_processed=games_processed,
                details=details or {},
                completed_at=datetime.utcnow() if status != "running" else None,
            )
            db.add(run)
    except Exception as e:
        logger.warning(f"Could not log run: {e}")


def _sf(val) -> float | None:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)
    except Exception:
        return None
