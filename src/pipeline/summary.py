"""Performance stats shared by the email and the dashboard."""

import pandas as pd

from src.data.odds_api import american_to_decimal


def _truthy(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))


def performance(preds: pd.DataFrame) -> dict:
    """Record, accuracy and flat 1-unit value-bet P&L for a set of predictions."""
    resolved = preds[preds["actual_home_win"].notna()]
    correct = int(_truthy(resolved["correct"]).sum())
    out = {
        "predicted": int(len(preds)),
        "resolved": int(len(resolved)),
        "correct": correct,
        "accuracy": round(correct / len(resolved), 4) if len(resolved) else None,
    }

    bets = resolved[_truthy(resolved["is_value_bet"])]
    won = _truthy(bets["value_bet_correct"])
    units = sum(
        (american_to_decimal(o) - 1) if w else -1.0
        for o, w in zip(bets["value_odds"], won)
    )
    out.update({
        "value_bets": int(len(bets)),
        "value_bets_won": int(won.sum()),
        "units": round(units, 2),
        "roi": round(units / len(bets), 4) if len(bets) else None,
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
