"""Performance stats shared by the email and the dashboard."""

import math

import pandas as pd

from src import betting


def _truthy(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _bet_stats(bets: pd.DataFrame, team_col: str, odds_col: str, won_col: str) -> dict:
    """Flat-stake record, ROI with a rough 95% range, and quarter-Kelly growth."""
    if bets.empty:
        return {"bets": 0, "won": 0, "units": 0.0, "roi": None, "roi_low": None, "roi_high": None,
                "kelly_growth": None}
    won = _truthy(bets[won_col])
    on_home = bets[team_col] == bets["home_team"]
    p_home = _num(bets["bet_home_prob"]).fillna(_num(bets["home_win_prob"]))
    frame = pd.DataFrame({"p": p_home.where(on_home, 1 - p_home), "odds": _num(bets[odds_col]), "won": won})
    stats = betting.simulate(frame, "p", "odds", "won")
    profit = [(betting.decimal(o) - 1) if w else -1.0 for o, w in zip(frame["odds"], frame["won"])]
    n = len(profit)
    # Normal approximation: with a few hundred bets a positive ROI can still be luck.
    half = 1.96 * pd.Series(profit).std(ddof=1) / math.sqrt(n) if n > 1 else None
    stats.update(roi_low=round(stats["roi"] - half, 4) if half is not None else None,
                 roi_high=round(stats["roi"] + half, 4) if half is not None else None)
    return stats


def performance(preds: pd.DataFrame) -> dict:
    """Pick record, value-bet and price-shop results, and closing-line value."""
    resolved = preds[preds["actual_home_win"].notna()]
    correct = int(_truthy(resolved["correct"]).sum())
    out = {
        "predicted": int(len(preds)),
        "resolved": int(len(resolved)),
        "correct": correct,
        "accuracy": round(correct / len(resolved), 4) if len(resolved) else None,
    }

    vb = _bet_stats(resolved[_truthy(resolved["is_value_bet"])], "value_team", "value_odds", "value_bet_correct")
    out.update({
        "value_bets": vb["bets"], "value_bets_won": vb["won"], "units": vb["units"], "roi": vb["roi"],
        "roi_low": vb["roi_low"], "roi_high": vb["roi_high"], "kelly_growth": vb["kelly_growth"],
    })
    shop = resolved[resolved["shop_team"].map(lambda v: isinstance(v, str))]
    sb = _bet_stats(shop, "shop_team", "shop_odds", "shop_bet_correct")
    out.update({"shop_bets": sb["bets"], "shop_won": sb["won"], "shop_units": sb["units"], "shop_roi": sb["roi"]})

    # Closing-line value of value bets: expected profit of the price taken,
    # judged by the closing fair probability. Positive on average = beating the market.
    vbc = preds[_truthy(preds["is_value_bet"]) & _num(preds["close_home_prob"]).notna()]
    clvs = []
    for _, r in vbc.iterrows():
        close_p = float(r["close_home_prob"]) if r["value_team"] == r["home_team"] else 1 - float(r["close_home_prob"])
        c = betting.clv(r["value_odds"], close_p)
        if c is not None:
            clvs.append(c)
    out.update({
        "clv_bets": len(clvs),
        "clv_avg": round(sum(clvs) / len(clvs), 4) if clvs else None,
        "clv_beat_pct": round(sum(c > 0 for c in clvs) / len(clvs), 4) if clvs else None,
    })
    return out


def daily_series(preds: pd.DataFrame) -> list[dict]:
    """Per-date results with running totals, for the trend chart."""
    resolved = preds[preds["actual_home_win"].notna()].copy()
    if resolved.empty:
        return []
    resolved["is_correct"] = _truthy(resolved["correct"])
    out, cum_c, cum_n = [], 0, 0
    for d, g in resolved.groupby("game_date"):
        cum_c += int(g["is_correct"].sum())
        cum_n += len(g)
        out.append({"date": d, "correct": int(g["is_correct"].sum()), "games": len(g),
                    "cum_accuracy": round(cum_c / cum_n, 4), "cum_games": cum_n})
    return out
