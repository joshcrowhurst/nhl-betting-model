from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.database.connection import get_db
from src.database.models import PerformanceCache, ModelRun

router = APIRouter(prefix="/api/performance", tags=["performance"])


class PerformanceOut(BaseModel):
    total_predictions: int
    resolved_predictions: int
    correct_predictions: int
    accuracy: Optional[float]
    total_value_bets: int
    value_bets_correct: int
    value_bet_accuracy: Optional[float]
    value_bet_roi: Optional[float]
    total_ev: Optional[float]
    daily_breakdown: Optional[list]
    computed_at: Optional[str]


class ModelRunOut(BaseModel):
    id: int
    run_type: str
    status: Optional[str]
    games_processed: Optional[int]
    started_at: Optional[str]
    completed_at: Optional[str]
    details: Optional[dict]


@router.get("", response_model=Optional[PerformanceOut])
def get_performance():
    with get_db() as db:
        cache = (
            db.query(PerformanceCache)
            .order_by(PerformanceCache.computed_at.desc())
            .first()
        )
    if not cache:
        return None
    return PerformanceOut(
        total_predictions=cache.total_predictions or 0,
        resolved_predictions=cache.resolved_predictions or 0,
        correct_predictions=cache.correct_predictions or 0,
        accuracy=cache.accuracy,
        total_value_bets=cache.total_value_bets or 0,
        value_bets_correct=cache.value_bets_correct or 0,
        value_bet_accuracy=cache.value_bet_accuracy,
        value_bet_roi=cache.value_bet_roi,
        total_ev=cache.total_ev,
        daily_breakdown=cache.daily_breakdown,
        computed_at=cache.computed_at.isoformat() if cache.computed_at else None,
    )


@router.get("/trend", response_model=list[dict])
def get_trend():
    """Return daily breakdown array from the latest performance cache."""
    with get_db() as db:
        cache = (
            db.query(PerformanceCache)
            .order_by(PerformanceCache.computed_at.desc())
            .first()
        )
    if not cache or not cache.daily_breakdown:
        return []
    return cache.daily_breakdown


@router.get("/runs", response_model=list[ModelRunOut])
def get_runs(limit: int = 20):
    with get_db() as db:
        rows = (
            db.query(ModelRun)
            .order_by(ModelRun.started_at.desc())
            .limit(limit)
            .all()
        )
    return [
        ModelRunOut(
            id=r.id,
            run_type=r.run_type,
            status=r.status,
            games_processed=r.games_processed,
            started_at=r.started_at.isoformat() if r.started_at else None,
            completed_at=r.completed_at.isoformat() if r.completed_at else None,
            details=r.details,
        )
        for r in rows
    ]
