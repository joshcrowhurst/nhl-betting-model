"""
Benter-style blend of the model's probability with the betting market's.

Benter (1994) found his fundamental model and the public odds each carried
information the other lacked, and combined them with a second-stage logistic
regression:

    logit(p) = a * logit(p_model) + b * logit(p_market) + c

The coefficients are fit on out-of-sample model predictions (from the
walk-forward backtest) joined to the odds available at prediction time, so
the blend learns how much to trust the model relative to the market.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-4


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit(p_model, p_market, y) -> dict:
    X = np.column_stack([_logit(p_model), _logit(p_market)])
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(X, np.asarray(y, dtype=int))
    return {"a_model": float(lr.coef_[0][0]), "b_market": float(lr.coef_[0][1]),
            "c": float(lr.intercept_[0]), "n": int(len(X))}


def apply(coefs: dict, p_model, p_market) -> np.ndarray:
    z = coefs["a_model"] * _logit(p_model) + coefs["b_market"] * _logit(p_market) + coefs["c"]
    return 1 / (1 + np.exp(-z))


def save(coefs: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coefs, indent=1))


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
