"""
Forward tester — logs model predictions for upcoming games before they're played,
then resolves outcomes once results come in.

Workflow:
  1. Run predict_today() before games start → writes pending predictions to JSONL log
  2. Run resolve_outcomes() after games finish → fills in actuals
  3. Run summary() any time to see live P&L

Log format (one JSON object per line):
  {game_id, date, home_team, away_team, home_win_prob, predicted_winner,
   market_home_prob, home_odds, away_odds, home_ev, away_ev,
   actual_home_win, resolved_at, correct}
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FORWARD_TEST_LOG
from src.features.feature_engineer import build_features
from src.models.moneyline_model import MoneylineModel
from src.data.odds_api import compute_ev, match_odds_to_games

logger = logging.getLogger(__name__)


def predict_today(
    model: MoneylineModel,
    all_games: pd.DataFrame,
    enriched: pd.DataFrame = None,
    consensus_odds: pd.DataFrame = None,
    target_date: date = None,
) -> list[dict]:
    """
    Generate and log predictions for games on target_date (defaults to today).
    consensus_odds: output of get_consensus_odds() from odds_api — optional.
    Returns list of prediction dicts (also written to JSONL log).
    Skips any games already logged.
    """
    if target_date is None:
        target_date = date.today()

    target_ts = pd.Timestamp(target_date)
    prior_games = all_games[all_games["date"] < target_ts].copy()
    # Only games on exactly today's date
    today_games = all_games[all_games["date"] == target_ts].copy()

    if today_games.empty:
        logger.info(f"No games found for {target_date}")
        return []

    # Match consensus odds to today's games
    if consensus_odds is not None and not consensus_odds.empty:
        today_games["date"] = target_ts
        today_games = match_odds_to_games(consensus_odds, today_games)

    existing_ids = _load_logged_ids()
    predictions = []

    for _, game in today_games.iterrows():
        if game["game_id"] in existing_ids:
            logger.debug(f"Skipping already-logged game {game['game_id']}")
            continue

        # build_features drops rows with null home_win, so temporarily set a dummy
        # value so today's game passes through the filter. It only uses prior dates
        # for its own features so the dummy doesn't affect anything.
        game_row = game.copy()
        game_row["home_win"] = 0.0

        combined = pd.concat([prior_games, pd.DataFrame([game_row])], ignore_index=True)
        features = build_features(combined, enriched=enriched)
        game_features = features[features["game_id"] == game["game_id"]]

        if game_features.empty:
            logger.warning(f"Could not build features for {game['game_id']} — insufficient history")
            continue

        prob = float(model.predict_proba(game_features)[0])
        away_prob = 1 - prob

        # Odds and EV
        market_home_prob = _safe_float(game.get("market_home_prob"))
        home_odds = _safe_float(game.get("home_odds"))
        away_odds = _safe_float(game.get("away_odds"))
        home_ev = compute_ev(prob, home_odds) if home_odds else None
        away_ev = compute_ev(away_prob, away_odds) if away_odds else None

        entry = {
            "game_id": int(game["game_id"]),
            "date": str(target_date),
            "season": game.get("season"),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_win_prob": round(prob, 4),
            "away_win_prob": round(away_prob, 4),
            "predicted_winner": game["home_team"] if prob > 0.5 else game["away_team"],
            "market_home_prob": market_home_prob,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "home_ev": home_ev,
            "away_ev": away_ev,
            "logged_at": datetime.utcnow().isoformat(),
            "actual_home_win": None,
            "resolved_at": None,
            "correct": None,
        }

        _append_log(entry)
        predictions.append(entry)
        logger.info(
            f"  {game['away_team']} @ {game['home_team']} → "
            f"home {prob:.1%}"
            + (f"  EV home {home_ev:+.3f} / away {away_ev:+.3f}" if home_ev is not None else "")
        )

    return predictions


def resolve_outcomes(all_games: pd.DataFrame) -> int:
    """Fill in actual outcomes for completed games. Returns count resolved."""
    log = _load_log()
    if not log:
        logger.info("No predictions to resolve.")
        return 0

    completed = all_games[all_games["home_win"].notna()].set_index("game_id")
    resolved_count = 0
    updated_log = []

    for entry in log:
        if entry.get("actual_home_win") is not None:
            updated_log.append(entry)
            continue

        gid = entry["game_id"]
        if gid in completed.index:
            actual = int(completed.loc[gid, "home_win"])
            entry["actual_home_win"] = actual
            entry["resolved_at"] = datetime.utcnow().isoformat()
            entry["correct"] = int((entry["home_win_prob"] > 0.5) == bool(actual))
            resolved_count += 1
            result = "HOME WIN" if actual else "AWAY WIN"
            mark = "✓" if entry["correct"] else "✗"
            logger.info(f"  {entry['away_team']} @ {entry['home_team']}: {result} — {mark}")
        updated_log.append(entry)

    _write_log(updated_log)
    logger.info(f"Resolved {resolved_count} new outcomes.")
    return resolved_count


def summary() -> pd.DataFrame:
    """Print forward test performance summary and return full log as DataFrame."""
    log = _load_log()
    if not log:
        print("No forward test predictions logged yet.")
        return pd.DataFrame()

    df = pd.DataFrame(log)
    df["date"] = pd.to_datetime(df["date"])
    resolved = df[df["actual_home_win"].notna()].copy()
    pending = df[df["actual_home_win"].isna()]

    print("\n" + "=" * 58)
    print("  FORWARD TEST SUMMARY")
    print("=" * 58)
    print(f"  Total predictions : {len(df)}")
    print(f"  Resolved          : {len(resolved)}")
    print(f"  Pending           : {len(pending)}")

    if len(resolved) >= 5:
        acc = resolved["correct"].mean()
        print(f"\n  Accuracy          : {acc:.1%} ({int(resolved['correct'].sum())}/{len(resolved)})")

        # Value bet performance
        if "home_ev" in resolved.columns:
            has_ev = resolved[resolved["home_ev"].notna() | resolved["away_ev"].notna()]
            value = has_ev[
                ((has_ev["home_win_prob"] >= 0.5) & (has_ev["home_ev"] > 0)) |
                ((has_ev["home_win_prob"] < 0.5) & (has_ev["away_ev"] > 0))
            ]
            if len(value) > 0:
                print(f"  Value bet acc     : {value['correct'].mean():.1%} ({len(value)} bets)")

        # Market edge
        if "market_home_prob" in resolved.columns and resolved["market_home_prob"].notna().sum() >= 5:
            resolved["model_edge"] = (resolved["home_win_prob"] - resolved["market_home_prob"]).abs()
            high_edge = resolved[resolved["model_edge"] > 0.05]
            if len(high_edge) > 0:
                print(f"  High-edge acc     : {high_edge['correct'].mean():.1%} ({len(high_edge)} bets, >5% edge)")

    if len(pending) > 0:
        print(f"\n  Upcoming / pending:")
        for _, r in pending.iterrows():
            home_ev = _safe_float(r.get("home_ev"))
            away_ev = _safe_float(r.get("away_ev"))
            home_odds = _safe_float(r.get("home_odds"))
            away_odds = _safe_float(r.get("away_odds"))

            # Show the value side if one exists, otherwise show pick side
            if home_ev is not None and away_ev is not None:
                if home_ev > 0:
                    bet_str = f"  ★ {r['home_team']} {_fmt_odds(home_odds)} EV {home_ev:+.3f}"
                elif away_ev > 0:
                    bet_str = f"  ★ {r['away_team']} {_fmt_odds(away_odds)} EV {away_ev:+.3f}"
                else:
                    bet_str = f"  no value"
            elif home_odds or away_odds:
                bet_str = f"  no value"
            else:
                bet_str = ""

            print(f"    {r['date'].date()} | {r['away_team']} @ {r['home_team']} "
                  f"→ {r['predicted_winner']} ({r['home_win_prob']:.1%}){bet_str}")

    print("=" * 58 + "\n")
    return df


def _fmt_odds(o) -> str:
    try:
        if o is None or (isinstance(o, float) and pd.isna(o)):
            return ""
        return f"+{int(o)}" if o > 0 else str(int(o))
    except (TypeError, ValueError):
        return ""


def _safe_float(val) -> float | None:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)
    except Exception:
        return None


def _load_log() -> list[dict]:
    if not FORWARD_TEST_LOG.exists():
        return []
    entries = []
    with open(FORWARD_TEST_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_logged_ids() -> set:
    return {e["game_id"] for e in _load_log()}


def _append_log(entry: dict) -> None:
    FORWARD_TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FORWARD_TEST_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _write_log(entries: list[dict]) -> None:
    FORWARD_TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FORWARD_TEST_LOG, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
