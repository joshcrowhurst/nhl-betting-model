"""
Walk-forward backtest engine.

Methodology:
  - Train on all games UP TO a cutoff date
  - Predict on next N games (test window)
  - Slide the window forward, retrain, repeat
  - Never sees future data during training (no lookahead bias)

Output: DataFrame of predictions with actual outcomes, plus summary metrics.
"""

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BACKTEST_START_SEASON, WALK_FORWARD_RETRAIN_FREQ
from src.features.feature_engineer import build_features, get_feature_cols
from src.models.moneyline_model import MoneylineModel, _score

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    start_date: str = "2021-10-01"    # first prediction date
    min_train_games: int = 500        # minimum games before first model fit
    retrain_every: int = WALK_FORWARD_RETRAIN_FREQ  # retrain after N test games
    include_market: bool = False       # include market odds feature


@dataclass
class BacktestResult:
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics_by_window: list = field(default_factory=list)
    overall_metrics: dict = field(default_factory=dict)
    betting_summary: dict = field(default_factory=dict)
    final_model: object = field(default=None)  # last trained model in the walk-forward


def run_backtest(
    features_df: pd.DataFrame,
    config: BacktestConfig = None,
) -> BacktestResult:
    """
    features_df: output of build_features() — must contain date, home_win, and all feature cols
    Returns BacktestResult with all predictions and performance summary.
    """
    if config is None:
        config = BacktestConfig()

    feature_cols = get_feature_cols(include_market=config.include_market)
    features_df = features_df.copy().sort_values("date").reset_index(drop=True)
    start_dt = pd.Timestamp(config.start_date)

    # Only require base features to be non-null — XGBoost handles NaN for optional ones
    base_cols = [c for c in feature_cols if c in features_df.columns
                 and not c.startswith(("home_goalie", "away_goalie", "goalie_", "home_shot", "away_shot",
                                       "shot_ratio", "home_faceoff", "away_faceoff",
                                       "home_hits", "away_hits", "market_"))]
    train_mask = (features_df["date"] < start_dt) & features_df[base_cols].notna().all(axis=1)
    if train_mask.sum() < config.min_train_games:
        raise ValueError(
            f"Only {train_mask.sum()} training games before {config.start_date}. "
            f"Need at least {config.min_train_games}."
        )

    test_df = features_df[features_df["date"] >= start_dt].copy()
    all_predictions = []
    window_metrics = []
    games_since_retrain = 0
    current_model = None

    logger.info(f"Starting walk-forward backtest from {config.start_date}")
    logger.info(f"Test games: {len(test_df)}, retrain every {config.retrain_every} games")

    for i, (idx, row) in enumerate(test_df.iterrows()):
        # Retrain if needed
        if games_since_retrain == 0 or games_since_retrain >= config.retrain_every:
            train_data = features_df[
                (features_df["date"] < row["date"]) &
                features_df[base_cols].notna().all(axis=1)
            ]
            if len(train_data) >= config.min_train_games:
                current_model = MoneylineModel(include_market=config.include_market)
                current_model.train(train_data, train_data["home_win"], calibrate=True)
                games_since_retrain = 0
                logger.debug(f"Retrained on {len(train_data)} games up to {row['date'].date()}")

        if current_model is None:
            continue

        row_df = pd.DataFrame([row])
        try:
            prob = current_model.predict_proba(row_df)[0]
        except Exception as e:
            logger.warning(f"Prediction failed for game {row.get('game_id')}: {e}")
            continue

        all_predictions.append({
            "game_id": row.get("game_id"),
            "date": row["date"],
            "season": row.get("season"),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_win_prob": round(prob, 4),
            "away_win_prob": round(1 - prob, 4),
            "predicted_winner": row["home_team"] if prob > 0.5 else row["away_team"],
            "actual_home_win": int(row["home_win"]),
            "correct": int((prob > 0.5) == bool(row["home_win"])),
            "market_home_prob": row.get("market_home_prob"),
        })
        games_since_retrain += 1

        # Collect window metrics every retrain cycle
        if games_since_retrain == config.retrain_every and len(all_predictions) >= 20:
            window_preds = pd.DataFrame(all_predictions[-config.retrain_every:])
            m = _score(window_preds["actual_home_win"], window_preds["home_win_prob"], label=f"window_{i}")
            m["end_date"] = str(row["date"].date())
            window_metrics.append(m)

    preds_df = pd.DataFrame(all_predictions)
    if preds_df.empty:
        return BacktestResult()

    overall = _score(preds_df["actual_home_win"], preds_df["home_win_prob"], label="overall")
    overall["accuracy"] = preds_df["correct"].mean()
    overall["n_games"] = len(preds_df)

    betting = _betting_summary(preds_df)
    logger.info(f"Backtest complete. Overall AUC: {overall['auc']}, Accuracy: {overall['accuracy']:.3f}")

    return BacktestResult(
        predictions=preds_df,
        metrics_by_window=window_metrics,
        overall_metrics=overall,
        betting_summary=betting,
        final_model=current_model,
    )


def _betting_summary(preds: pd.DataFrame) -> dict:
    """
    Simulated flat-stake betting: bet $1 on model's predicted winner each game.
    Requires 'market_home_prob' to compute expected value bets.
    Returns ROI and other betting metrics.
    """
    total_games = len(preds)
    correct = preds["correct"].sum()
    accuracy = correct / total_games

    summary = {
        "total_games": total_games,
        "correct_predictions": int(correct),
        "accuracy": round(accuracy, 4),
        "date_range": f"{preds['date'].min().date()} → {preds['date'].max().date()}",
    }

    # If market odds available, compute value betting stats
    if "market_home_prob" in preds.columns and preds["market_home_prob"].notna().sum() > 50:
        # Edge = model prob minus market no-vig prob
        model_home_edge = preds["home_win_prob"] - preds["market_home_prob"]
        value_bets = preds[model_home_edge.abs() > 0.03]  # >3% edge threshold
        if len(value_bets) > 0:
            vb_accuracy = value_bets["correct"].mean()
            summary["value_bets"] = len(value_bets)
            summary["value_bet_accuracy"] = round(vb_accuracy, 4)

    return summary


def print_backtest_report(result: BacktestResult) -> None:
    m = result.overall_metrics
    b = result.betting_summary
    print("\n" + "="*55)
    print("  BACKTEST REPORT")
    print("="*55)
    print(f"  Period:     {b.get('date_range', 'N/A')}")
    print(f"  Games:      {m.get('n_games', 0)}")
    print(f"  Accuracy:   {m.get('accuracy', 0):.1%}")
    print(f"  ROC-AUC:    {m.get('auc', 0):.4f}")
    print(f"  Log Loss:   {m.get('log_loss', 0):.4f}")
    print(f"  Brier:      {m.get('brier', 0):.4f}")
    if "value_bets" in b:
        print(f"\n  Value Bets: {b['value_bets']} ({b.get('value_bet_accuracy', 0):.1%} acc)")
    print("="*55 + "\n")
