"""
Daily email via Gmail SMTP (free; send from your own account to yourself).

Needs GMAIL_USER and GMAIL_APP_PASSWORD (a Google "app password", which
requires 2-step verification on the account). EMAIL_TO defaults to GMAIL_USER.
"""

import html
import logging
import os
import smtplib
from datetime import date
from email.message import EmailMessage

import pandas as pd

from src.pipeline import store
from src.pipeline.summary import performance, _truthy

logger = logging.getLogger(__name__)

RED = "#C8102E"
GOLD = "#F1BE48"
INK = "#1A1A1A"
MUTED = "#6B6460"
RULE = "#ECE6DF"
PAPER = "#FBF8F4"
WIN = "#1E7B3C"


def _odds(o) -> str:
    if o is None or pd.isna(o):
        return "—"
    o = int(round(float(o)))
    return f"+{o}" if o > 0 else str(o)


def _pct(p) -> str:
    if p is None or pd.isna(p):
        return "—"
    return ">99%" if p >= 0.995 else "<1%" if p < 0.005 else f"{float(p):.0%}"


def _goalie(p, side: str) -> str:
    """'Markstrom (confirmed)' style label for a starter."""
    name = p.get(f"{side}_starter_name")
    if not isinstance(name, str):
        return "TBD"
    status = p.get(f"{side}_starter_status")
    last = name.split()[-1]
    return html.escape(f"{last} ({status})" if isinstance(status, str) else last)


BOOKS = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
    "caesars": "Caesars", "betrivers": "BetRivers", "espnbet": "ESPN BET", "fanatics": "Fanatics",
    "bovada": "Bovada", "betonlineag": "BetOnline", "mybookieag": "MyBookie", "lowvig": "LowVig",
    "betus": "BetUS", "pointsbetus": "PointsBet", "hardrockbet": "Hard Rock", "ballybet": "Bally Bet",
}


def book_name(key) -> str:
    return BOOKS.get(key, str(key).replace("_", " ").title()) if isinstance(key, str) else "?"


def _cell(content: str, align: str = "left", extra: str = "") -> str:
    return (f'<td style="padding:10px 12px;border-bottom:1px solid {RULE};'
            f'text-align:{align};vertical-align:top;{extra}">{content}</td>')


def _head(cols: list[tuple[str, str]]) -> str:
    return "<tr>" + "".join(
        f'<th style="padding:8px 12px;text-align:{a};font-size:11px;letter-spacing:.06em;'
        f'text-transform:uppercase;color:{MUTED};border-bottom:2px solid {RED}">{t}</th>'
        for t, a in cols) + "</tr>"


def build_html(game_date: date, today: pd.DataFrame, last_results: pd.DataFrame,
               last_date: str | None, season_perf: dict, dashboard_url: str | None) -> str:
    esc = html.escape
    has_odds = today["home_odds"].notna().any()

    pick_rows = ""
    for _, p in today.iterrows():
        pick = p["predicted_winner"]
        prob = p["home_win_prob"] if pick == p["home_team"] else p["away_win_prob"]
        value = ""
        if _truthy(pd.Series([p["is_value_bet"]])).iloc[0]:
            stake = p.get("value_stake")
            stake_txt = f" · stake {float(stake):.1%}" if stake is not None and not pd.isna(stake) else ""
            value = (f'<span style="background:{GOLD};color:{INK};font-weight:700;padding:2px 6px;'
                     f'border-radius:4px;font-size:12px;white-space:nowrap">VALUE {esc(str(p["value_team"]))} '
                     f'{_odds(p["value_odds"])}{stake_txt}</span>')
        if isinstance(p.get("shop_team"), str):
            value += (f'<div style="font-size:12px;color:{MUTED};margin-top:4px">Best price: '
                      f'<strong style="color:{INK}">{esc(p["shop_team"])} {_odds(p["shop_odds"])}</strong> '
                      f'at {esc(book_name(p.get("shop_book")))} ({float(p["shop_ev"]):+.1%})</div>')
        goalies = ""
        if isinstance(p.get("away_starter_name"), str) or isinstance(p.get("home_starter_name"), str):
            goalies = (f'<div style="font-size:12px;color:{MUTED};margin-top:2px">'
                       f'{_goalie(p, "away")} vs {_goalie(p, "home")}</div>')
        # When a rationale row follows, it carries the divider instead.
        x = "border-bottom:0;" if isinstance(p.get("rationale"), str) else ""
        pick_rows += "<tr>" + _cell(f'{esc(p["away_team"])} <span style="color:{MUTED}">@</span> {esc(p["home_team"])}{goalies}', extra=x) \
            + _cell(f'<strong style="color:{RED}">{esc(pick)}</strong>', "center", x) \
            + _cell(_pct(prob), "center", x) \
            + (_cell(f'{_odds(p["away_odds"])} / {_odds(p["home_odds"])}', "center", x) if has_odds else "") \
            + _cell(value, extra=x) + "</tr>"
        if isinstance(p.get("rationale"), str):
            pick_rows += (f'<tr><td colspan="{5 if has_odds else 4}" style="padding:0 12px 12px;'
                          f'border-bottom:1px solid {RULE};font-size:13px;line-height:1.45;color:{MUTED}">'
                          f'{esc(p["rationale"])}</td></tr>')

    cols = [("Matchup", "left"), ("Pick", "center"), ("Win prob", "center")]
    if has_odds:
        cols.append(("Odds (away / home)", "center"))
    cols.append(("", "left"))
    picks_table = (f'<table style="width:100%;border-collapse:collapse;font-size:14px">'
                   f'{_head(cols)}{pick_rows}</table>')

    odds_note = (
        f'<p style="margin:12px 0 0;font-size:12px;color:{MUTED}">Odds are the median across US bookmakers. '
        f'Value = positive expected value at that price; stake = quarter-Kelly, as % of bankroll. '
        f'Best price = a bookmaker beating the consensus fair price, regardless of the model.</p>'
    ) if has_odds else (
        f'<p style="margin:12px 0 0;font-size:12px;color:{MUTED}">Odds unavailable today, '
        f'so no value-bet flags. Set the ODDS_API_KEY secret to enable them.</p>')

    results = ""
    if not last_results.empty:
        rows = ""
        for _, r in last_results.iterrows():
            ok = _truthy(pd.Series([r["correct"]])).iloc[0]
            mark = (f'<strong style="color:{WIN}">✓</strong>' if ok
                    else f'<strong style="color:{RED}">✗</strong>')
            score = f'{int(r["away_score"])}–{int(r["home_score"])}' if pd.notna(r["home_score"]) else ""
            rows += "<tr>" + _cell(f'{esc(r["away_team"])} @ {esc(r["home_team"])}') \
                + _cell(score, "center") + _cell(esc(r["predicted_winner"]), "center") \
                + _cell(mark, "center") + "</tr>"
        n_ok = int(_truthy(last_results["correct"]).sum())
        results = f"""
      <h2 style="font-size:15px;margin:28px 0 8px;color:{INK}">Last results · {date.fromisoformat(last_date).strftime('%a %b %-d')} · {n_ok}/{len(last_results)} correct</h2>
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        {_head([("Matchup", "left"), ("Score", "center"), ("Pick", "center"), ("", "center")])}{rows}
      </table>"""

    record = ""
    if season_perf["resolved"]:
        record = (f'Season: <strong>{season_perf["correct"]}–{season_perf["resolved"] - season_perf["correct"]}</strong> '
                  f'({season_perf["accuracy"]:.1%})')
        if season_perf["value_bets"]:
            record += (f' · Value bets {season_perf["value_bets_won"]}–'
                       f'{season_perf["value_bets"] - season_perf["value_bets_won"]}, '
                       f'{season_perf["units"]:+.2f}u')

    subtitle = " · ".join(x for x in (f"{len(today)} game{'s' if len(today) != 1 else ''}", record) if x)
    link = (f'<a href="{esc(dashboard_url)}" style="color:{RED};font-weight:600">Open the dashboard →</a>'
            if dashboard_url else "")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light only"></head>
<body style="margin:0;padding:20px 12px;background:{PAPER};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:{INK}">
  <div style="max-width:680px;margin:0 auto;background:#fff;border-radius:10px;overflow:hidden;border:1px solid {RULE}">
    <div style="background:{RED};padding:22px 24px;border-bottom:5px solid {GOLD}">
      <div style="color:{GOLD};font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase">NHL Model</div>
      <div style="color:#fff;font-size:22px;font-weight:800;margin-top:4px">{game_date.strftime("%A, %B %-d")}</div>
      <div style="color:#FFE3E8;font-size:13px;margin-top:6px">{subtitle}</div>
    </div>
    <div style="padding:20px 24px">
      {picks_table}
      {odds_note}
      {results}
      <p style="margin:28px 0 0;font-size:13px">{link}</p>
    </div>
    <div style="padding:14px 24px;background:{PAPER};font-size:11px;color:{MUTED};border-top:1px solid {RULE}">
      Model output for information only. Not financial advice.
    </div>
  </div>
</body></html>"""


def send_daily_email(game_date: date) -> None:
    user = os.getenv("GMAIL_USER", "")
    password = os.getenv("GMAIL_APP_PASSWORD", "")
    to = os.getenv("EMAIL_TO") or user
    if not (user and password):
        logger.warning("GMAIL_USER / GMAIL_APP_PASSWORD not set — skipping email")
        store.log_run("email", "skipped", reason="no credentials")
        return

    preds = store.load_predictions()
    today = preds[preds["game_date"] == game_date.isoformat()]
    if today.empty:
        logger.info("No predictions for today — no email")
        return
    earlier = preds[(preds["game_date"] < game_date.isoformat()) & preds["actual_home_win"].notna()]
    last_date = earlier["game_date"].max() if not earlier.empty else None
    last_results = earlier[earlier["game_date"] == last_date] if last_date else earlier
    season = today["season"].iloc[0]
    season_perf = performance(preds[preds["season"] == season])

    value_count = int(_truthy(today["is_value_bet"]).sum())
    subject = f"NHL picks · {game_date.strftime('%a %b %-d')} · {len(today)} games"
    if value_count:
        subject += f" · {value_count} value bet{'s' if value_count > 1 else ''}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content("Your daily NHL model picks — open in an HTML-capable mail client.")
    msg.add_alternative(
        build_html(game_date, today, last_results, last_date, season_perf, os.getenv("DASHBOARD_URL")),
        subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)
    store.log_run("email", "success", to_count=1, games=len(today))
    logger.info(f"Email sent: {subject}")
