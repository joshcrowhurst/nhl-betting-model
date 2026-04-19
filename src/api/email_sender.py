"""
SendGrid email sender for daily prediction digests.
"""

import logging
import os
from datetime import date

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

logger = logging.getLogger(__name__)

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
FROM_EMAIL = "josh.crowhurst@gmail.com"
TO_EMAIL = "josh.crowhurst@gmail.com"


def _fmt_odds(o) -> str:
    try:
        if o is None:
            return "N/A"
        return f"+{int(o)}" if o > 0 else str(int(o))
    except (TypeError, ValueError):
        return "N/A"


def _fmt_prob(p) -> str:
    try:
        return f"{float(p):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def _build_html(predictions: list, game_date: date) -> str:
    date_str = game_date.strftime("%A, %B %-d, %Y")

    rows = ""
    value_bets = []

    for p in predictions:
        home = p.home_team
        away = p.away_team
        home_prob = _fmt_prob(p.home_win_prob)
        away_prob = _fmt_prob(p.away_win_prob)
        winner = p.predicted_winner
        home_odds = _fmt_odds(p.home_odds)
        away_odds = _fmt_odds(p.away_odds)

        home_ev = p.home_ev
        away_ev = p.away_ev
        has_value = p.is_value_bet

        value_label = ""
        if has_value and p.value_team:
            ev_str = f"{p.value_ev:+.3f}" if p.value_ev else ""
            odds_str = _fmt_odds(p.value_odds)
            value_label = (
                f'<span style="color:#16a34a;font-weight:600">'
                f"★ {p.value_team} {odds_str} EV {ev_str}"
                f"</span>"
            )

        row_bg = "#f0fdf4" if has_value else "white"
        rows += f"""
        <tr style="background:{row_bg}">
          <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb">
            {away} <span style="color:#6b7280">@</span> {home}
          </td>
          <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;text-align:center">
            <strong>{winner}</strong>
          </td>
          <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;text-align:center">
            {home} {home_prob} / {away} {away_prob}
          </td>
          <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;text-align:center">
            {home} {home_odds} / {away} {away_odds}
          </td>
          <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb">
            {value_label}
          </td>
        </tr>"""

        if has_value:
            value_bets.append(p)

    value_section = ""
    if value_bets:
        items = ""
        for p in value_bets:
            ev_str = f"{p.value_ev:+.3f}" if p.value_ev else ""
            items += f"<li><strong>{p.value_team}</strong> ({_fmt_odds(p.value_odds)}) — EV {ev_str} — {p.away_team} @ {p.home_team}</li>"
        value_section = f"""
        <div style="margin:24px 0;padding:16px 20px;background:#f0fdf4;border-left:4px solid #16a34a;border-radius:4px">
          <p style="margin:0 0 8px;font-weight:700;color:#15803d">★ Value Bets Today ({len(value_bets)})</p>
          <ul style="margin:0;padding-left:20px;color:#166534">{items}</ul>
        </div>"""
    else:
        value_section = """
        <div style="margin:24px 0;padding:16px 20px;background:#f9fafb;border-left:4px solid #9ca3af;border-radius:4px">
          <p style="margin:0;color:#6b7280">No value bets today — model edge doesn't overcome the vig.</p>
        </div>"""

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;margin:0;padding:24px">
  <div style="max-width:700px;margin:0 auto;background:white;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1)">
    <div style="background:#1e3a5f;padding:24px 28px">
      <h1 style="color:white;margin:0;font-size:20px">NHL Model — {date_str}</h1>
      <p style="color:#93c5fd;margin:4px 0 0;font-size:14px">{len(predictions)} game(s) predicted</p>
    </div>
    <div style="padding:24px 28px">
      {value_section}
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead>
          <tr style="background:#f9fafb">
            <th style="padding:10px 14px;text-align:left;color:#6b7280;font-weight:600;border-bottom:2px solid #e5e7eb">Matchup</th>
            <th style="padding:10px 14px;text-align:center;color:#6b7280;font-weight:600;border-bottom:2px solid #e5e7eb">Pick</th>
            <th style="padding:10px 14px;text-align:center;color:#6b7280;font-weight:600;border-bottom:2px solid #e5e7eb">Probability</th>
            <th style="padding:10px 14px;text-align:center;color:#6b7280;font-weight:600;border-bottom:2px solid #e5e7eb">Odds</th>
            <th style="padding:10px 14px;text-align:left;color:#6b7280;font-weight:600;border-bottom:2px solid #e5e7eb">Value</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div style="padding:16px 28px;background:#f9fafb;font-size:12px;color:#9ca3af;border-top:1px solid #e5e7eb">
      NHL Betting Model — for informational purposes only. Not financial advice.
    </div>
  </div>
</body>
</html>"""


def send_predictions_email(predictions: list, game_date: date) -> None:
    if not SENDGRID_API_KEY:
        logger.warning("SENDGRID_API_KEY not set — skipping email")
        return

    html = _build_html(predictions, game_date)
    date_str = game_date.strftime("%b %-d")
    subject = f"NHL Predictions — {date_str} ({len(predictions)} games)"

    value_count = sum(1 for p in predictions if p.is_value_bet)
    if value_count:
        subject += f" ★ {value_count} value bet{'s' if value_count > 1 else ''}"

    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=TO_EMAIL,
        subject=subject,
        html_content=html,
    )

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        logger.info(f"Email sent — status {response.status_code}")
    except Exception as e:
        logger.error(f"Email send failed: {e}")
