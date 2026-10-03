"""
Plain-English rationale for each prediction.

XGBoost can split a prediction into per-feature contributions (SHAP values,
`pred_contribs=True`). We average them across the calibration folds, group
related features (goaltending, rest, form...), and turn the biggest groups
into short sentences that quote the underlying numbers. No extra service or
cost: it's computed from the model itself.

Contributions are in log-odds of a HOME win; positive favours the home team.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

# feature -> group. Groups are explained as one sentence.
GROUPS = {
    "elo_home_win_prob": "strength",
    "goalie_sv_pct_diff": "goalie_form",
    "starter_sv_pct_diff": "starter",
    "home_starter_share_l10": "backup",
    "away_starter_share_l10": "backup",
    "shot_ratio_diff": "shots",
    "home_gd_per_game_l10": "goal_diff",
    "away_gd_per_game_l10": "goal_diff",
    "home_win_rate_home_l10": "venue",
    "away_win_rate_away_l10": "venue",
    "margin_momentum_diff": "momentum",
    "home_blowout_loss_prev": "bounce_back",
    "away_blowout_loss_prev": "bounce_back",
    "h2h_home_win_rate": "h2h",
    "home_faceoff_pct_l10": "faceoffs",
    "away_faceoff_pct_l10": "faceoffs",
    "rest_advantage": "rest",
    "home_is_b2b": "rest",
    "away_is_b2b": "rest",
    "home_is_3in4": "rest",
    "away_is_3in4": "rest",
    "away_games_last_7_days": "rest",
    "away_direct_travel_miles": "travel",
}
MIN_EFFECT = 0.03       # log-odds; ignore groups smaller than this
AGAINST_RATIO = 0.6     # mention the main counter-factor if it's at least this big


def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature log-odds contributions toward a home win, one row per game."""
    feats = X[model.feature_cols]
    inner = model._model
    estimators = ([cc.estimator for cc in inner.calibrated_classifiers_]
                  if hasattr(inner, "calibrated_classifiers_") else [inner])
    dm = xgb.DMatrix(feats)
    contribs = np.mean([e.get_booster().predict(dm, pred_contribs=True) for e in estimators], axis=0)
    return pd.DataFrame(contribs[:, :-1], columns=model.feature_cols, index=X.index)  # drop bias


def _sv(x) -> str:
    return f"{x:.3f}".lstrip("0")


def _sentence(group: str, f: pd.Series, home: str, away: str, favoured: str, g: dict) -> str | None:
    """One sentence for a group, written from the point of view of `favoured`."""
    other = away if favoured == home else home
    fav_home = favoured == home

    def val(name):
        v = f.get(name)
        return None if v is None or pd.isna(v) else float(v)

    if group == "strength":
        p = val("elo_home_win_prob")
        if p is None:
            return None
        p = p if fav_home else 1 - p
        if p < 0.55:  # too small an edge to be worth a sentence
            return None
        return f"Over the longer run {favoured} has been the stronger team (rating edge alone makes them {p:.0%})."
    if group == "starter":
        hs, as_ = val("home_starter_sv_pct"), val("away_starter_sv_pct")
        if hs is None or as_ is None:
            return None
        fs, os_ = (hs, as_) if fav_home else (as_, hs)
        fn, on = (g["home"], g["away"]) if fav_home else (g["away"], g["home"])
        if fs >= os_:
            return f"{favoured}'s starter {fn} has a {_sv(fs)} save % over recent starts, vs {_sv(os_)} for {on}."
        return f"{other}'s starter {on} has struggled relative to {fn} ({_sv(os_)} vs {_sv(fs)} save %)."
    if group == "backup":
        hs, as_ = val("home_starter_share_l10"), val("away_starter_share_l10")
        o_share, o_name = (as_, g["away"]) if fav_home else (hs, g["home"])
        f_share, f_name = (hs, g["home"]) if fav_home else (as_, g["away"])
        if o_share is not None and o_share < 0.5:
            return f"{other} is likely in a backup goalie ({o_name} has started {round(o_share * 10)} of their last 10)."
        if f_share is not None and f_share >= 0.6:
            return f"{favoured} has its regular starter in net ({f_name}, {round(f_share * 10)} of the last 10 starts)."
        return None
    if group == "goalie_form":
        h, a = val("home_goalie_sv_pct_l10"), val("away_goalie_sv_pct_l10")
        if h is None or a is None:
            return None
        fv, ov = (h, a) if fav_home else (a, h)
        if fv <= ov:
            return None
        return f"{favoured}'s goaltending has been sharper over the last 10 games ({_sv(fv)} vs {_sv(ov)})."
    if group == "shots":
        h, a = val("home_shot_ratio_l10"), val("away_shot_ratio_l10")
        if h is None or a is None:
            return None
        fv, ov = (h, a) if fav_home else (a, h)
        if fv <= ov:
            return None
        return f"{favoured} has been controlling play, taking {fv:.0%} of shots lately vs {ov:.0%} for {other}."
    if group == "goal_diff":
        h, a = val("home_gd_per_game_l10"), val("away_gd_per_game_l10")
        if h is None or a is None:
            return None
        fv, ov = (h, a) if fav_home else (a, h)
        if fv <= ov:
            return None
        return f"Recent form favours {favoured}: {fv:+.1f} goals a game over their last 10, vs {ov:+.1f} for {other}."
    if group == "venue":
        h, a = val("home_win_rate_home_l10"), val("away_win_rate_away_l10")
        if fav_home and h is not None and h >= 0.5 and (a is None or h >= a):
            return f"{home} has won {h:.0%} of its recent home games" + (f", and {away} {a:.0%} on the road." if a is not None else ".")
        if not fav_home and a is not None and a >= 0.5 and (h is None or a >= h):
            return f"{away} has won {a:.0%} of its recent road games" + (f", and {home} {h:.0%} at home." if h is not None else ".")
        return None
    if group == "momentum":
        d = val("margin_momentum_diff")
        if d is None or (d if fav_home else -d) <= 0:
            return None
        return f"{favoured} has the better run of results over the last 3 games."
    if group == "bounce_back":
        hb, ab = val("home_blowout_loss_prev"), val("away_blowout_loss_prev")
        if (hb if fav_home else ab):
            return f"{favoured} is coming off a blowout loss, which teams tend to answer."
        if (ab if fav_home else hb):
            return f"{other} is coming off a blowout loss."
        return None
    if group == "h2h":
        r = val("h2h_home_win_rate")
        if r is None:
            return None
        r = r if fav_home else 1 - r
        if r <= 0.5:
            return None
        return f"{favoured} has won {r:.0%} of recent meetings between these teams."
    if group == "faceoffs":
        # The faceoff feature averages per-skater faceoff % including players
        # who took none, so its values (~18%) aren't real faceoff win rates
        # and shouldn't be quoted. Skip it until the feature is rebuilt.
        return None
    if group == "rest":
        hb, ab = val("home_is_b2b"), val("away_is_b2b")
        if (ab if fav_home else hb):
            return f"{other} is on the second night of a back-to-back."
        rest = val("rest_advantage")
        if rest:
            r = rest if fav_home else -rest
            # Skip big gaps: early in the season they're just offseason time.
            if 0 < r <= 3:
                return f"{favoured} is better rested ({int(r)} more day{'s' if r > 1 else ''} off)."
        g7 = val("away_games_last_7_days")
        if g7 and g7 >= 4:
            return f"{away} has a heavy schedule ({int(g7)} games in the last 7 days)."
        return None
    if group == "travel":
        miles = val("away_direct_travel_miles")
        if not miles or miles < 500 or not fav_home:
            return None
        return f"{away} travelled about {int(round(miles, -2)):,} miles to get here."
    return None


def rationale(features: pd.Series, contribs: pd.Series, home: str, away: str, favoured: str,
              goalie_names: dict | None = None) -> str:
    """2-3 sentences: the main reasons for `favoured`, plus the biggest factor against."""
    goalie_names = {"home": "the home starter", "away": "the away starter", **(goalie_names or {})}
    sign = 1.0 if favoured == home else -1.0
    by_group = {}
    for feat, c in contribs.items():
        grp = GROUPS.get(feat)
        if grp:
            by_group[grp] = by_group.get(grp, 0.0) + sign * float(c)

    supporting = sorted(((v, k) for k, v in by_group.items() if v > MIN_EFFECT), reverse=True)
    against = sorted(((v, k) for k, v in by_group.items() if v < -MIN_EFFECT))

    sentences = []
    for _, grp in supporting:
        s = _sentence(grp, features, home, away, favoured, goalie_names)
        if s:
            sentences.append(s)
        if len(sentences) == 2:
            break
    if not sentences:
        return f"A close call: no single factor stands out, with {favoured} slightly ahead overall."

    other = away if favoured == home else home
    if against and supporting and -against[0][0] >= AGAINST_RATIO * supporting[0][0]:
        s = _sentence(against[0][1], features, home, away, other, goalie_names)
        if s:
            if not s.startswith((home, away)):
                s = s[0].lower() + s[1:]
            sentences.append(f"In {other}'s favour: " + s)
    return " ".join(sentences)
