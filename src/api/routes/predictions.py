from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.database.connection import get_db
from src.database.models import Prediction

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


class PredictionOut(BaseModel):
    id: int
    game_id: int
    game_date: date
    season: Optional[str]
    game_type: Optional[int]
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    predicted_winner: Optional[str]
    market_home_prob: Optional[float]
    home_odds: Optional[float]
    away_odds: Optional[float]
    home_ev: Optional[float]
    away_ev: Optional[float]
    is_value_bet: Optional[bool]
    value_team: Optional[str]
    value_odds: Optional[float]
    value_ev: Optional[float]
    actual_home_win: Optional[int]
    correct: Optional[bool]
    value_bet_correct: Optional[bool]

    class Config:
        from_attributes = True


def _to_out(p: Prediction) -> PredictionOut:
    return PredictionOut.model_validate(p)


@router.get("/today", response_model=list[PredictionOut])
def get_today():
    today = date.today()
    with get_db() as db:
        rows = (
            db.query(Prediction)
            .filter(Prediction.game_date == today)
            .order_by(Prediction.id)
            .all()
        )
    return [_to_out(r) for r in rows]


@router.get("/recent", response_model=list[PredictionOut])
def get_recent(days: int = Query(7, ge=1, le=90)):
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=days)
    with get_db() as db:
        rows = (
            db.query(Prediction)
            .filter(Prediction.game_date >= cutoff)
            .order_by(Prediction.game_date.desc(), Prediction.id)
            .all()
        )
    return [_to_out(r) for r in rows]


@router.get("/date/{game_date}", response_model=list[PredictionOut])
def get_by_date(game_date: date):
    with get_db() as db:
        rows = (
            db.query(Prediction)
            .filter(Prediction.game_date == game_date)
            .order_by(Prediction.id)
            .all()
        )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No predictions for {game_date}")
    return [_to_out(r) for r in rows]
