"""
Betting math shared by the daily pipeline, the dashboard and the backtests.

All odds are American. Probabilities are win probabilities for the side being
bet. "Fair" probabilities are bookmaker-implied probabilities with the margin
(vig) removed.
"""

import math

import pandas as pd

KELLY_FRACTION = 0.25    # quarter Kelly: much lower variance for a small loss in growth
KELLY_CAP = 0.03         # never stake more than 3% of bankroll on one game
SHOP_MIN_EV = 0.02       # best price must beat the consensus fair price by 2%


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def decimal(american) -> float | None:
    if not _ok(american):
        return None
    a = float(american)
    if abs(a) < 100:  # not a valid American price
        return None
    return a / 100 + 1 if a > 0 else 100 / abs(a) + 1


def ev(prob, american) -> float | None:
    """Expected profit per 1 unit staked."""
    d = decimal(american)
    if d is None or not _ok(prob):
        return None
    return round(float(prob) * d - 1, 4)


def kelly_stake(prob, american, fraction: float = KELLY_FRACTION, cap: float = KELLY_CAP) -> float:
    """Fraction of bankroll to stake (0 when there's no edge)."""
    d = decimal(american)
    if d is None or not _ok(prob) or d <= 1:
        return 0.0
    full = (float(prob) * d - 1) / (d - 1)
    return round(min(max(full * fraction, 0.0), cap), 4)


def clv(bet_american, close_fair_prob) -> float | None:
    """Closing-line value: the bet's expected profit judged by the closing fair price."""
    return ev(close_fair_prob, bet_american)


def simulate(bets: pd.DataFrame, prob_col: str, odds_col: str, won_col: str) -> dict:
    """
    Flat 1-unit and quarter-Kelly results for a set of bets.
    bets: one row per bet with the bettor's probability, the price taken and
    whether it won.
    """
    if bets.empty:
        return {"bets": 0, "won": 0, "units": 0.0, "roi": None, "kelly_growth": None}
    won = bets[won_col].astype(bool)
    profit = [(decimal(o) - 1) if w else -1.0 for o, w in zip(bets[odds_col], won)]
    log_growth = 0.0
    for p, o, w in zip(bets[prob_col], bets[odds_col], won):
        stake = kelly_stake(p, o)
        log_growth += math.log1p(stake * (decimal(o) - 1) if w else -stake)
    return {
        "bets": int(len(bets)),
        "won": int(won.sum()),
        "units": round(sum(profit), 2),
        "roi": round(sum(profit) / len(bets), 4),
        "kelly_growth": round(math.expm1(log_growth), 4),   # bankroll change, e.g. 0.08 = +8%
    }
