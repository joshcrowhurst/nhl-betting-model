"""
File-based storage for predictions and run history, replacing Cloud SQL.

predictions.csv holds one row per predicted game (~1,300 rows a season), and
runs.jsonl holds one line per pipeline step. Both live in DATA_DIR, which the
GitHub Actions workflow points at the `state` branch.
"""

import json
from datetime import datetime, timezone

import pandas as pd

from config import PREDICTIONS_LOG, RUNS_LOG

COLUMNS = [
    "game_id", "game_date", "season", "game_type", "home_team", "away_team",
    "home_win_prob", "away_win_prob", "predicted_winner", "rationale",
    "home_starter_name", "home_starter_status", "away_starter_name", "away_starter_status",
    "market_home_prob", "home_odds", "away_odds", "home_ev", "away_ev",
    "is_value_bet", "value_team", "value_odds", "value_ev",
    "home_score", "away_score", "actual_home_win", "correct", "value_bet_correct",
    "logged_at", "resolved_at",
]

MAX_RUNS_KEPT = 500


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_LOG.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(PREDICTIONS_LOG, dtype={"season": str})
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[COLUMNS]


def save_predictions(df: pd.DataFrame) -> None:
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    df = df[COLUMNS].sort_values(["game_date", "game_id"])
    df.to_csv(PREDICTIONS_LOG, index=False)


def log_run(step: str, status: str, **details) -> None:
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "step": step, "status": status, **details}
    lines = load_runs()
    lines.append(entry)
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    RUNS_LOG.write_text("".join(json.dumps(e, default=str) + "\n" for e in lines[-MAX_RUNS_KEPT:]))


def load_runs() -> list[dict]:
    if not RUNS_LOG.exists():
        return []
    return [json.loads(l) for l in RUNS_LOG.read_text().splitlines() if l.strip()]
