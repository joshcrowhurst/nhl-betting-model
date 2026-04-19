"""
Main entry point. Run this to fetch data, train, backtest, or forward-test.

Usage examples:
  python run.py backtest              # run walk-forward backtest
  python run.py predict               # predict today's games
  python run.py resolve               # fill in yesterday's results
  python run.py summary               # show forward test P&L
  python run.py fetch --seasons 3     # just fetch/cache data
"""

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads ODDS_API_KEY from .env if present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from config import MODELS_DIR
from src.data.nhl_api import get_multiple_seasons, season_range, current_season_code, get_games_for_date
from src.data.boxscore_enricher import get_enriched_game_stats, load_enriched_stats
from src.data.odds_api import get_current_odds, get_consensus_odds, match_odds_to_games, format_odds_line, compute_ev
from src.features.feature_engineer import build_features
from src.models.moneyline_model import MoneylineModel
from src.backtest.backtest_engine import run_backtest, BacktestConfig, print_backtest_report
from src.forward_test.forward_tester import predict_today, resolve_outcomes, summary


def fetch_data(num_seasons: int = 5):
    current = current_season_code()
    start_year = int(current[:4]) - num_seasons + 1
    start = f"{start_year}{start_year+1}"
    seasons = season_range(start, current)
    logger.info(f"Fetching seasons: {seasons}")
    df = get_multiple_seasons(seasons)
    logger.info(f"Total games fetched: {len(df)}")
    return df, seasons


def fetch_enriched(df, seasons, skip_enrichment: bool = False):
    if skip_enrichment:
        return None
    logger.info("Fetching boxscore enrichment (goalie sv%, SOG, hits, faceoff%)...")
    frames = []
    for season in seasons:
        season_df = df[df["season"] == season]
        enriched = get_enriched_game_stats(season_df, season)
        if not enriched.empty:
            frames.append(enriched)
    import pandas as pd
    return pd.concat(frames, ignore_index=True) if frames else None


def cmd_backtest(args) -> None:
    df, seasons = fetch_data(num_seasons=args.seasons)
    if df.empty:
        logger.error("No game data fetched — check your network connection.")
        sys.exit(1)
    enriched = fetch_enriched(df, seasons, skip_enrichment=args.skip_enrichment)
    logger.info("Building features...")
    features = build_features(df, enriched=enriched)
    if features.empty:
        logger.error("Feature builder returned empty — not enough historical data.")
        sys.exit(1)
    logger.info(f"Feature rows: {len(features)}")

    config = BacktestConfig(
        start_date=args.start_date,
        include_market=False,
    )
    result = run_backtest(features, config=config)
    print_backtest_report(result)

    # Save predictions for inspection
    out = Path("data/processed/backtest_predictions.parquet")
    result.predictions.to_parquet(out, index=False)
    logger.info(f"Predictions saved to {out}")


def cmd_predict(args) -> None:
    import pandas as pd
    from datetime import date as date_type

    model_path = MODELS_DIR / "moneyline_latest.pkl"
    if not model_path.exists():
        logger.error("No trained model found. Run 'python run.py train' first.")
        sys.exit(1)

    model = MoneylineModel.load(model_path)

    # Historical games for feature building (regular season only)
    df, seasons = fetch_data(num_seasons=3)
    enriched = fetch_enriched(df, seasons)

    # Today's games — include playoffs (gameType 3)
    today = date_type.today()
    today_games = get_games_for_date(today, game_types=(2, 3))
    if not today_games.empty:
        # Merge today's games into df so forward_tester can find them
        df = pd.concat([df, today_games[~today_games["game_id"].isin(df["game_id"])]], ignore_index=True)
        logger.info(f"Found {len(today_games)} game(s) today (incl. playoffs)")

    # Fetch current odds
    consensus = None
    try:
        logger.info("Fetching current odds...")
        raw_odds = get_current_odds()
        consensus = get_consensus_odds(raw_odds)
        logger.info(f"Got odds for {len(consensus)} games")
    except Exception as e:
        logger.warning(f"Could not fetch odds: {e}")

    predictions = predict_today(model, df, enriched=enriched, consensus_odds=consensus)

    if not predictions:
        print("No new games to predict today.")
        return

    print(f"\n{'='*58}")
    print(f"  PREDICTIONS  —  {len(predictions)} game(s) today")
    print(f"{'='*58}")

    value_bets = []
    for p in predictions:
        print(format_odds_line(p["home_team"], p["away_team"], p))
        home_ev = p.get("home_ev")
        away_ev = p.get("away_ev")
        if (home_ev is not None and home_ev > 0) or (away_ev is not None and away_ev > 0):
            value_bets.append(p)

    print("─" * 58)
    if value_bets:
        print(f"  ★  {len(value_bets)} value bet(s) identified (EV > 0, marked above)\n")
    else:
        print("  No value bets today — model edge doesn't overcome the vig\n")


def cmd_resolve(args) -> None:
    df, _ = fetch_data(num_seasons=1)
    resolved = resolve_outcomes(df)
    print(f"Resolved {resolved} game outcomes.")


def cmd_summary(args) -> None:
    summary()


def cmd_importance(args) -> None:
    """Train a model on all data and print feature importance table."""
    df, seasons = fetch_data(num_seasons=args.seasons)
    enriched = fetch_enriched(df, seasons, skip_enrichment=args.skip_enrichment)
    features = build_features(df, enriched=enriched)

    from src.features.feature_engineer import get_feature_cols
    feat_cols = get_feature_cols(include_market=False)
    valid = features.dropna(subset=["home_win"])
    # Don't drop rows with NaN optional features — XGBoost handles them
    valid = valid.dropna(subset=[c for c in feat_cols
                                  if not c.startswith(("home_goalie", "away_goalie", "goalie_",
                                                       "home_shot", "away_shot", "shot_ratio",
                                                       "home_faceoff", "away_faceoff",
                                                       "home_hits", "away_hits", "market_"))])

    logger.info(f"Training on {len(valid)} games for importance analysis...")
    model = MoneylineModel(include_market=False)
    model.train(valid, valid["home_win"], calibrate=True)

    imp = model.feature_importance()
    if imp.empty:
        print("Could not extract feature importances.")
        return

    imp = imp.sort_values(ascending=False)
    total = imp.sum()

    print("\n" + "=" * 52)
    print(f"  FEATURE IMPORTANCE  (trained on {len(valid):,} games)")
    print("=" * 52)
    print(f"  {'Feature':<38} {'Importance':>8}")
    print("-" * 52)
    cumulative = 0.0
    for feat, score in imp.items():
        pct = score / total * 100
        cumulative += pct
        bar = "█" * int(pct / 2)
        print(f"  {feat:<38} {pct:>6.1f}%  {bar}")
        if cumulative > 95:
            remaining = len(imp) - list(imp.index).index(feat) - 1
            if remaining > 0:
                print(f"  ... {remaining} more features (< 5% combined)")
            break
    print("=" * 52 + "\n")


def cmd_train_and_save(args) -> None:
    """Train on all available data and save model (used before forward testing)."""
    df, seasons = fetch_data(num_seasons=args.seasons)
    enriched = fetch_enriched(df, seasons, skip_enrichment=args.skip_enrichment)
    features = build_features(df, enriched=enriched)
    features = features.dropna(subset=["home_win"])

    from src.features.feature_engineer import get_feature_cols
    feat_cols = get_feature_cols(include_market=False)
    features = features.dropna(subset=feat_cols)

    logger.info(f"Training on {len(features)} games...")
    model = MoneylineModel(include_market=False)
    metrics = model.train(features, features["home_win"])
    logger.info(f"Train metrics: {metrics}")
    path = model.save()
    print(f"Model saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NHL Betting Model")
    sub = parser.add_subparsers(dest="command")

    p_backtest = sub.add_parser("backtest", help="Run walk-forward backtest")
    p_backtest.add_argument("--seasons", type=int, default=6, help="Seasons of data to use")
    p_backtest.add_argument("--start-date", default="2022-10-01", help="First prediction date")
    p_backtest.add_argument("--skip-enrichment", action="store_true", help="Skip boxscore fetch (faster, fewer features)")
    p_backtest.set_defaults(func=cmd_backtest)

    p_train = sub.add_parser("train", help="Train and save model on all data")
    p_train.add_argument("--seasons", type=int, default=5)
    p_train.add_argument("--skip-enrichment", action="store_true")
    p_train.set_defaults(func=cmd_train_and_save)

    p_imp = sub.add_parser("importance", help="Show feature importance table")
    p_imp.add_argument("--seasons", type=int, default=6)
    p_imp.add_argument("--skip-enrichment", action="store_true")
    p_imp.set_defaults(func=cmd_importance)

    p_predict = sub.add_parser("predict", help="Predict today's games")
    p_predict.set_defaults(func=cmd_predict)

    p_resolve = sub.add_parser("resolve", help="Resolve yesterday's predictions")
    p_resolve.set_defaults(func=cmd_resolve)

    p_summary = sub.add_parser("summary", help="Show forward test summary")
    p_summary.set_defaults(func=cmd_summary)

    p_fetch = sub.add_parser("fetch", help="Fetch and cache NHL data")
    p_fetch.add_argument("--seasons", type=int, default=5)
    p_fetch.set_defaults(func=lambda a: fetch_data(a.seasons))

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)
