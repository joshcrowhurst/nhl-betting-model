"""
Historical odds (paid Odds API plans): backfilling odds for saved predictions,
and backtesting the model against the market.

  backfill_predictions  Fill in odds for predictions made without them (as of
                        the time each prediction was logged), plus closing odds
                        (a snapshot 5 minutes before each start time).
  compare_market        Walk-forward backtest joined to the odds available each
                        morning: model vs market vs Benter-style blend, and
                        value-bet / price-shop results. Saves the blend for the
                        daily pipeline when it beats the model.

Historical snapshots cost 10 credits each and are cached on disk (in the state
branch), so re-running is free. Every function takes a credit budget and stops
fetching when it's spent or when the key is close to running out.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import log_loss

from config import RAW_DIR
from src import betting
from src.data import odds_api
from src.data.odds_api import get_consensus_odds, get_historical_odds, match_odds_to_games
from src.models import blend
from src.pipeline import pricing, store

logger = logging.getLogger(__name__)

CREDITS_PER_SNAPSHOT = 10
KEEP_IN_RESERVE = 300           # leave enough for live daily odds
MORNING_UTC_HOUR = 15           # ~11 AM ET: odds as they'd be when picks go out
SEASON_CACHE = RAW_DIR / "odds_hist"


class Budget:
    def __init__(self, max_credits: int):
        self.left = max_credits
        self.used = 0

    def allow(self, cached: bool) -> bool:
        if cached:
            return True
        if self.left < CREDITS_PER_SNAPSHOT:
            return False
        if odds_api.LAST_REMAINING is not None and odds_api.LAST_REMAINING < KEEP_IN_RESERVE:
            return False
        return True

    def spend(self, cached: bool) -> None:
        if not cached:
            self.left -= CREDITS_PER_SNAPSHOT
            self.used += CREDITS_PER_SNAPSHOT


def _is_cached(at: pd.Timestamp) -> bool:
    iso = at.strftime("%Y-%m-%dT%H:%M:%SZ")
    return (SEASON_CACHE / f"{iso.replace(':', '')}.json").exists()


def _snapshot(at: pd.Timestamp, budget: Budget) -> pd.DataFrame | None:
    """Consensus odds at `at`, or None if the budget doesn't allow the call."""
    at = at.tz_convert("UTC") if at.tzinfo else at.tz_localize("UTC")
    at = at.floor("min")
    cached = _is_cached(at)
    if not budget.allow(cached):
        return None
    raw = get_historical_odds(at.to_pydatetime())
    budget.spend(cached)
    return get_consensus_odds(raw) if not raw.empty else pd.DataFrame()


def _plan_error(e: Exception) -> str | None:
    if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code in (401, 403, 422):
        return (f"Historical odds request refused ({e.response.status_code}): "
                f"{e.response.text[:200]}. Historical odds need a paid Odds API plan.")
    return None


# ---------------------------------------------------------------------------
# Backfill saved predictions
# ---------------------------------------------------------------------------

def backfill_predictions(max_credits: int = 2000) -> dict:
    preds = store.load_predictions()
    budget = Budget(max_credits)
    filled = closed = 0
    coefs = None  # these picks were made without a blend; price them the same way

    try:
        # 1) Opening odds as of when each prediction was logged
        missing = preds[preds["home_odds"].isna() & preds["logged_at"].notna()]
        for logged_at, group in missing.groupby("logged_at"):
            cons = _snapshot(pd.Timestamp(logged_at), budget)
            if cons is None:
                break
            if cons.empty:
                continue
            games = group[["game_id", "home_team", "away_team", "game_date"]].assign(
                date=pd.to_datetime(group["game_date"]))
            matched = match_odds_to_games(cons, games).set_index("game_id")
            for idx in group.index:
                gid = preds.at[idx, "game_id"]
                if gid not in matched.index or pd.isna(matched.at[gid, "home_odds"]):
                    continue
                m = matched.loc[gid].to_dict()
                m.update(home_team=preds.at[idx, "home_team"], away_team=preds.at[idx, "away_team"])
                for k, v in pricing.price(float(preds.at[idx, "home_win_prob"]), m, coefs).items():
                    preds.at[idx, k] = v
                preds.at[idx, "odds_source"] = "historical"
                preds.at[idx, "commence_time"] = pd.Timestamp(m["commence_time"]).isoformat()
                filled += 1

        # 2) Closing odds: one snapshot 5 minutes before each distinct start time
        need_close = preds[preds["close_home_prob"].isna() & preds["commence_time"].notna()
                           & preds["actual_home_win"].notna()]
        for start, group in need_close.groupby("commence_time"):
            cons = _snapshot(pd.Timestamp(start) - timedelta(minutes=5), budget)
            if cons is None:
                break
            if cons.empty:
                continue
            games = group[["game_id", "home_team", "away_team"]].assign(date=pd.to_datetime(group["game_date"]))
            matched = match_odds_to_games(cons, games).set_index("game_id")
            for idx in group.index:
                gid = preds.at[idx, "game_id"]
                if gid in matched.index and not pd.isna(matched.at[gid, "home_odds"]):
                    preds.at[idx, "close_home_odds"] = float(matched.at[gid, "home_odds"])
                    preds.at[idx, "close_away_odds"] = float(matched.at[gid, "away_odds"])
                    preds.at[idx, "close_home_prob"] = round(float(matched.at[gid, "market_home_prob"]), 4)
                    preds.at[idx, "close_at"] = str(start)
                    closed += 1
    except requests.HTTPError as e:
        msg = _plan_error(e)
        if not msg:
            raise
        logger.error(msg)
        store.save_predictions(preds)
        store.log_run("backfill", "error", error=msg, filled=filled, closed=closed)
        return {"filled": filled, "closed": closed, "error": msg}

    store.save_predictions(preds)
    result = {"filled": filled, "closed": closed, "credits_used": budget.used,
              "credits_remaining": odds_api.LAST_REMAINING}
    store.log_run("backfill", "success", **result)
    logger.info(f"Backfill: {result}")
    return result


# ---------------------------------------------------------------------------
# Season odds for backtests
# ---------------------------------------------------------------------------

def season_odds(schedule: pd.DataFrame, season: str, budget: Budget) -> pd.DataFrame:
    """Morning consensus odds for every game in a season's schedule (cached per season)."""
    SEASON_CACHE.mkdir(parents=True, exist_ok=True)
    path = SEASON_CACHE / f"season_v2_{season}.parquet"  # v2: decimal-median consensus
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["game_id"])
    sched = schedule[schedule["season"] == season]
    todo = sched[~sched["game_id"].isin(have["game_id"])]
    rows = []
    for day, group in todo.groupby("date"):
        at = pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=MORNING_UTC_HOUR)
        cons = _snapshot(at, budget)
        if cons is None:
            logger.info(f"Credit budget reached at {day.date()} ({season})")
            break
        if cons.empty:
            continue
        m = match_odds_to_games(cons, group[["game_id", "date", "home_team", "away_team"]])
        rows.append(m[m["home_odds"].notna()])
    if rows:
        new = pd.concat(rows, ignore_index=True)
        keep = ["game_id", "market_home_prob", "home_odds", "away_odds",
                "best_home_odds", "best_home_book", "best_away_odds", "best_away_book"]
        have = pd.concat([have, new[keep]], ignore_index=True) if not have.empty else new[keep]
        have.to_parquet(path, index=False)
    return have


# ---------------------------------------------------------------------------
# Model vs market vs blend
# ---------------------------------------------------------------------------

def _bets(df: pd.DataFrame, prob_col: str) -> dict:
    """Value bets at the consensus price using probabilities in prob_col."""
    rows = []
    for _, r in df.iterrows():
        p = r[prob_col]
        he, ae = betting.ev(p, r["home_odds"]), betting.ev(1 - p, r["away_odds"])
        if he is not None and he > 0 and (ae is None or he >= ae):
            rows.append({"p": p, "odds": r["home_odds"], "won": r["actual_home_win"] == 1})
        elif ae is not None and ae > 0:
            rows.append({"p": 1 - p, "odds": r["away_odds"], "won": r["actual_home_win"] == 0})
    return betting.simulate(pd.DataFrame(rows, columns=["p", "odds", "won"]), "p", "odds", "won")


def _shop_bets(df: pd.DataFrame) -> dict:
    rows = []
    for _, r in df.iterrows():
        m = r["market_home_prob"]
        cands = [(betting.ev(m, r["best_home_odds"]), m, r["best_home_odds"], r["actual_home_win"] == 1),
                 (betting.ev(1 - m, r["best_away_odds"]), 1 - m, r["best_away_odds"], r["actual_home_win"] == 0)]
        cands = [c for c in cands if c[0] is not None and c[0] >= betting.SHOP_MIN_EV]
        if cands:
            _, p, o, w = max(cands)
            rows.append({"p": p, "odds": o, "won": w})
    return betting.simulate(pd.DataFrame(rows, columns=["p", "odds", "won"]), "p", "odds", "won")


def _ll(y, p) -> float:
    return round(float(log_loss(y, np.clip(p, 1e-4, 1 - 1e-4), labels=[0, 1])), 4)


def compare_market(start_date: str = "2023-10-01", max_credits: int = 12000, retrain_every: int = 100) -> dict:
    from src.backtest.backtest_engine import run_backtest, BacktestConfig
    from src.features.feature_engineer import build_features
    from src.pipeline.daily import _load_history, TRAIN_SEASONS

    games, enriched = _load_history(TRAIN_SEASONS)
    logger.info("Building features for the market backtest...")
    features = build_features(games, enriched=enriched)
    preds = run_backtest(features, BacktestConfig(start_date=start_date, retrain_every=retrain_every)).predictions
    preds["season"] = preds["season"].astype(str)

    budget = Budget(max_credits)
    frames = []
    try:
        for season in sorted(preds["season"].unique()):
            # Only games the backtest predicted (completed ones), never future dates.
            frames.append(season_odds(games[games["game_id"].isin(preds["game_id"])], season, budget))
    except requests.HTTPError as e:
        msg = _plan_error(e)
        if not msg:
            raise
        store.log_run("market", "error", error=msg)
        raise SystemExit(msg)
    odds = pd.concat(frames, ignore_index=True).drop_duplicates("game_id")
    # The backtest carries its own (empty) market column; use the backfilled one.
    preds = preds.drop(columns=[c for c in ("market_home_prob",) if c in preds.columns])
    df = preds.merge(odds, on="game_id", how="inner").dropna(subset=["market_home_prob", "home_odds", "away_odds"])
    if df.empty:
        raise SystemExit("No historical odds matched the backtest games.")

    # Walk-forward blend: fit on all earlier seasons, test on the next.
    df["blend_prob"] = np.nan
    seasons = sorted(df["season"].unique())
    for i, season in enumerate(seasons[1:], start=1):
        train = df[df["season"].isin(seasons[:i])]
        coefs = blend.fit(train["home_win_prob"], train["market_home_prob"], train["actual_home_win"])
        test = df["season"] == season
        df.loc[test, "blend_prob"] = blend.apply(coefs, df.loc[test, "home_win_prob"], df.loc[test, "market_home_prob"])

    lines = ["## Model vs market", "",
             f"Walk-forward predictions from {start_date}, joined to morning consensus odds "
             f"({len(df)} games with odds). Credits used this run: {budget.used}; "
             f"remaining on key: {odds_api.LAST_REMAINING}.", "",
             "| Season | Games | Model log loss | Market log loss | Blend log loss | "
             "Model value bets (ROI, Kelly) | Blend value bets (ROI, Kelly) | Price shop (ROI) |",
             "|---|---|---|---|---|---|---|---|"]
    per_season = {}
    for season in seasons + ["all"]:
        d = df if season == "all" else df[df["season"] == season]
        y = d["actual_home_win"].astype(int)
        has_blend = d["blend_prob"].notna()
        r = {
            "games": len(d),
            "model_ll": _ll(y, d["home_win_prob"]),
            "market_ll": _ll(y, d["market_home_prob"]),
            "blend_ll": _ll(y[has_blend], d.loc[has_blend, "blend_prob"]) if has_blend.any() else None,
            "model_bets": _bets(d, "home_win_prob"),
            "blend_bets": _bets(d[has_blend], "blend_prob") if has_blend.any() else None,
            "shop": _shop_bets(d),
        }
        per_season[season] = r

        def fmt(b):
            if not b or not b["bets"]:
                return "—"
            return f"{b['bets']} bets, {b['roi']:+.1%}, {b['kelly_growth']:+.1%}"
        lines.append(f"| {season} | {r['games']} | {r['model_ll']} | {r['market_ll']} | "
                     f"{r['blend_ll'] if r['blend_ll'] is not None else '—'} | {fmt(r['model_bets'])} | "
                     f"{fmt(r['blend_bets'])} | {fmt(r['shop'])} |")

    # Final blend on everything; deploy it only if it beat the model out of sample.
    oos = df[df["blend_prob"].notna()]
    deploy = (not oos.empty and _ll(oos["actual_home_win"], oos["blend_prob"])
              < _ll(oos["actual_home_win"], oos["home_win_prob"]))
    coefs = blend.fit(df["home_win_prob"], df["market_home_prob"], df["actual_home_win"])
    if deploy:
        blend.save(coefs, pricing.BLEND_PATH)
        verdict = (f"Blend deployed (weights: model {coefs['a_model']:.2f}, market {coefs['b_market']:.2f}). "
                   "Daily value bets now use the blended probability.")
    else:
        if pricing.BLEND_PATH.exists():
            pricing.BLEND_PATH.unlink()
        verdict = "Blend not deployed: it didn't beat the model out of sample. Value bets use the model alone."
    lines += ["", verdict, "",
              "Log loss: lower is better. ROI: flat 1-unit stakes at the consensus price. "
              "Kelly: bankroll change with quarter-Kelly stakes. Price shop: bets where one bookmaker's "
              f"price beats the consensus fair price by {betting.SHOP_MIN_EV:.0%}+."]
    report = "\n".join(lines)
    print(report)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")
    store.log_run("market", "success", games=len(df), credits_used=budget.used, blend_deployed=bool(deploy),
                  coefs=coefs, all=per_season["all"])
    return {"per_season": per_season, "coefs": coefs, "deployed": deploy}
