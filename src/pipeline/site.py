"""
Builds the static GitHub Pages dashboard: site/index.html plus a data.json
snapshot of predictions, performance and recent runs.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import BASE_DIR
from src.pipeline import store
from src.pipeline.daily import today_et
from src.pipeline.summary import performance, daily_series

SITE_SRC = BASE_DIR / "site"


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in SITE_SRC.iterdir():
        shutil.copy(f, out_dir / f.name)

    preds = store.load_predictions()
    seasons = sorted(preds["season"].dropna().astype(str).unique(), reverse=True)
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "today": today_et().isoformat(),
        "seasons": {
            s: {"performance": performance(preds[preds["season"] == s]),
                "series": daily_series(preds[preds["season"] == s])}
            for s in seasons
        },
        "all_time": performance(preds),
        "predictions": _records(preds.sort_values(["game_date", "game_id"], ascending=[False, True])),
        "runs": store.load_runs()[-30:][::-1],
    }
    (out_dir / "data.json").write_text(json.dumps(data, default=str))
    return out_dir
