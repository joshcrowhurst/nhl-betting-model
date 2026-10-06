"""
Turns a model probability plus market odds into the betting fields stored for
each prediction: expected value, the value-bet side and its Kelly stake, and
any price-shopping opportunity. Used for live predictions and for predictions
whose odds are backfilled from the historical endpoint.
"""

import math

from config import MODELS_DIR
from src import betting
from src.models import blend

BLEND_PATH = MODELS_DIR / "blend.json"


def _num(x):
    try:
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
    except (TypeError, ValueError):
        return None


def price(p_model_home: float, game: dict, coefs: dict | None) -> dict:
    """
    p_model_home: model's home-win probability.
    game: dict with home_team, away_team and (optionally) home_odds, away_odds,
          market_home_prob, best_{home,away}_odds/_book.
    coefs: market-blend coefficients, or None to bet on the model alone.
    """
    home, away = game["home_team"], game["away_team"]
    home_odds, away_odds = _num(game.get("home_odds")), _num(game.get("away_odds"))
    market = _num(game.get("market_home_prob"))

    p_bet, source = p_model_home, "model"
    if coefs and market is not None:
        p_bet, source = float(blend.apply(coefs, [p_model_home], [market])[0]), "blend"

    out = {
        "market_home_prob": market,
        "home_odds": home_odds,
        "away_odds": away_odds,
        "bet_home_prob": round(p_bet, 4) if home_odds is not None else None,
        "prob_source": source if home_odds is not None else None,
        "home_ev": betting.ev(p_bet, home_odds),
        "away_ev": betting.ev(1 - p_bet, away_odds),
        "is_value_bet": False,
        "value_team": None, "value_odds": None, "value_ev": None, "value_stake": None,
        "shop_team": None, "shop_odds": None, "shop_book": None, "shop_ev": None,
    }
    for side in ("home", "away"):
        out[f"best_{side}_odds"] = _num(game.get(f"best_{side}_odds"))
        b = game.get(f"best_{side}_book")
        out[f"best_{side}_book"] = b if isinstance(b, str) else None

    # Value bet: the side with positive expected value at the consensus price.
    h, a = out["home_ev"], out["away_ev"]
    if h is not None and h > 0 and (a is None or h >= a):
        team, odds, e, prob = home, home_odds, h, p_bet
    elif a is not None and a > 0:
        team, odds, e, prob = away, away_odds, a, 1 - p_bet
    else:
        team = None
    if team:
        out.update(is_value_bet=True, value_team=team, value_odds=odds, value_ev=e,
                   value_stake=betting.kelly_stake(prob, odds))

    # Price shopping: a bookmaker's best price beats the consensus fair price,
    # whatever the model thinks.
    if market is not None:
        best = []
        for side, team_, fair in (("home", home, market), ("away", away, 1 - market)):
            e = betting.ev(fair, out[f"best_{side}_odds"])
            if e is not None and e >= betting.SHOP_MIN_EV:
                best.append((e, team_, side))
        if best:
            e, team_, side = max(best)
            out.update(shop_team=team_, shop_odds=out[f"best_{side}_odds"],
                       shop_book=out[f"best_{side}_book"], shop_ev=e)
    return out
