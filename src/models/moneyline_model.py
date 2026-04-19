"""
XGBoost moneyline win-probability model.
Outputs calibrated probability that the home team wins.

Usage:
    model = MoneylineModel()
    model.train(X_train, y_train)
    probs = model.predict_proba(X_test)
    model.save("models_saved/model_v1.pkl")
    model = MoneylineModel.load("models_saved/model_v1.pkl")
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR
from src.features.feature_engineer import get_feature_cols

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_lambda": 1.5,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


class MoneylineModel:
    def __init__(self, params: dict = None, include_market: bool = False):
        """
        include_market: if True, uses market implied prob as a feature.
        Set False for pure model, True to blend with market.
        """
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.include_market = include_market
        self.feature_cols = get_feature_cols(include_market=include_market)
        self._model = None
        self.train_metrics = {}

    def _make_base(self) -> XGBClassifier:
        return XGBClassifier(**self.params)

    def train(self, X: pd.DataFrame, y: pd.Series, calibrate: bool = True) -> dict:
        """
        Fit the model. Returns dict of in-sample metrics.
        calibrate=True wraps XGB in isotonic regression calibration.
        """
        X_feat = X[self.feature_cols].copy()
        # XGBoost handles NaN natively — no imputation needed

        base = self._make_base()
        if calibrate:
            self._model = CalibratedClassifierCV(base, method="isotonic", cv=5)
        else:
            self._model = base

        self._model.fit(X_feat, y)
        probs = self._model.predict_proba(X_feat)[:, 1]
        self.train_metrics = _score(y, probs, label="train")
        logger.info(f"Train metrics: {self.train_metrics}")
        return self.train_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns array of home-win probabilities."""
        if self._model is None:
            raise RuntimeError("Model not trained yet.")
        X_feat = X[self.feature_cols].copy()
        return self._model.predict_proba(X_feat)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series, label: str = "eval") -> dict:
        probs = self.predict_proba(X)
        metrics = _score(y, probs, label=label)
        logger.info(f"{label} metrics: {metrics}")
        return metrics

    def feature_importance(self) -> pd.Series:
        """Returns feature importances averaged across calibration folds."""
        model = self._model
        importances = None

        # CalibratedClassifierCV wraps estimators in calibrated_classifiers_
        if hasattr(model, "calibrated_classifiers_"):
            fold_importances = []
            for cc in model.calibrated_classifiers_:
                est = cc.estimator
                if hasattr(est, "feature_importances_"):
                    fold_importances.append(est.feature_importances_)
            if fold_importances:
                importances = np.mean(fold_importances, axis=0)
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        if importances is None:
            return pd.Series(dtype=float)
        return pd.Series(importances, index=self.feature_cols).sort_values(ascending=False)

    def save(self, path: str | Path = None) -> Path:
        if path is None:
            path = MODELS_DIR / "moneyline_latest.pkl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MoneylineModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def cross_validate(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """
    Stratified k-fold cross-validation. Returns averaged metrics.
    Note: for time-series data use the backtest engine instead — this is
    useful for quick sanity checks only.
    """
    feature_cols = get_feature_cols(include_market=False)
    X_feat = X[feature_cols].copy()

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_feat, y)):
        X_tr, X_val = X_feat.iloc[train_idx], X_feat.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = MoneylineModel()
        model.train(X_tr, y_tr, calibrate=True)
        metrics = model.evaluate(X_val, y_val, label=f"fold_{fold}")
        fold_metrics.append(metrics)

    avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    logger.info(f"CV averages: {avg}")
    return avg


def _score(y_true, y_prob, label: str = "") -> dict:
    return {
        "label": label,
        "n": len(y_true),
        "auc": round(roc_auc_score(y_true, y_prob), 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
        "brier": round(brier_score_loss(y_true, y_prob), 4),
        "accuracy": round(((y_prob > 0.5) == y_true).mean(), 4),
    }
