import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models_saved"

# NHL API (no key required)
NHL_API_BASE = "https://api-web.nhle.com/v1"

# The Odds API
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "icehockey_nhl"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h"  # moneyline

# Model settings
TRAIN_SEASONS_MIN = 3       # minimum seasons of data before first model fit
ROLLING_WINDOW_GAMES = 10   # recent-form window (games)
MIN_GAMES_FOR_FEATURES = 5  # games a team must have played before predicting

# Backtest settings
BACKTEST_START_SEASON = "20182019"
WALK_FORWARD_RETRAIN_FREQ = 30  # retrain every N games

# Forward test
FORWARD_TEST_LOG = LOGS_DIR / "forward_test.jsonl"

for d in [RAW_DIR, PROCESSED_DIR, LOGS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
