from datetime import datetime
from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, Integer,
    String, BigInteger, Text
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    game_id = Column(BigInteger, unique=True, nullable=False, index=True)
    game_date = Column(Date, nullable=False, index=True)
    season = Column(String(10))
    game_type = Column(Integer, default=2)   # 2=regular, 3=playoff
    home_team = Column(String(10), nullable=False)
    away_team = Column(String(10), nullable=False)

    # Model output
    home_win_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)
    predicted_winner = Column(String(10))

    # Market
    market_home_prob = Column(Float)
    home_odds = Column(Float)
    away_odds = Column(Float)
    home_ev = Column(Float)
    away_ev = Column(Float)

    # Value bet (whichever side has positive EV)
    is_value_bet = Column(Boolean, default=False)
    value_team = Column(String(10))
    value_odds = Column(Float)
    value_ev = Column(Float)

    # Resolution (filled in by resolve job)
    actual_home_win = Column(Integer)         # 0 or 1, NULL until resolved
    correct = Column(Boolean)                 # did model pick winner?
    value_bet_correct = Column(Boolean)       # did value bet win?

    logged_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True)
    run_type = Column(String(20), nullable=False)   # predict / resolve / retrain
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String(20))                     # success / error / partial
    games_processed = Column(Integer, default=0)
    details = Column(JSONB)


class PerformanceCache(Base):
    __tablename__ = "performance_cache"

    id = Column(Integer, primary_key=True)
    computed_at = Column(DateTime, default=datetime.utcnow)

    # Overall
    total_predictions = Column(Integer, default=0)
    resolved_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    accuracy = Column(Float)

    # Value bets
    total_value_bets = Column(Integer, default=0)
    value_bets_correct = Column(Integer, default=0)
    value_bet_accuracy = Column(Float)
    value_bet_roi = Column(Float)        # (winnings - stakes) / stakes
    total_ev = Column(Float)             # sum of all value bet EVs logged

    # Streaks
    current_correct_streak = Column(Integer, default=0)
    current_value_streak = Column(Integer, default=0)
    best_correct_streak = Column(Integer, default=0)

    # Serialised daily breakdown for trend chart (list of {date, accuracy, roi})
    daily_breakdown = Column(JSONB)
