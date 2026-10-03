"""
Prints what the live data sources return, so the parsers can be checked
against reality. Run by the backtest workflow (the dev sandbox can't reach these).
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.starting_goalies import DFO_URL, HEADERS, _find_games, fetch_daily_faceoff, _roster_goalies


def show(title, obj, limit=2500):
    print(f"\n===== {title} =====")
    print(json.dumps(obj, indent=1, default=str)[:limit] if not isinstance(obj, str) else obj[:limit])


for d in [date.today(), date(2026, 4, 10)]:
    try:
        html = requests.get(DFO_URL.format(date=d.isoformat()), headers=HEADERS, timeout=20).text
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
        show(f"DFO {d}: has __NEXT_DATA__", bool(m))
        if m:
            data = json.loads(m.group(1))
            show(f"DFO {d}: pageProps keys", list(data.get("props", {}).get("pageProps", {}).keys()))
            found = []
            _find_games(data, found)
            show(f"DFO {d}: {len(found)} game objects; first one", found[0] if found else None)
        show(f"DFO {d}: parsed", fetch_daily_faceoff(d))
    except Exception as e:
        show(f"DFO {d}: ERROR", repr(e))

try:
    show("Roster CGY goalies", _roster_goalies("CGY"))
except Exception as e:
    show("Roster ERROR", repr(e))

try:
    box = requests.get("https://api-web.nhle.com/v1/gamecenter/2025021000/boxscore", timeout=15).json()
    show("Boxscore home goalies", box.get("playerByGameStats", {}).get("homeTeam", {}).get("goalies"))
except Exception as e:
    show("Boxscore ERROR", repr(e))
