"""
FastAPI application. Serves:
  - Job endpoints (POST /jobs/*) for Cloud Scheduler
  - Data API endpoints (GET /api/*) for the React frontend
  - Health check (GET /health)
"""

import logging
import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.database.connection import init_db, health_check
from src.api.routes.predictions import router as predictions_router
from src.api.routes.performance import router as performance_router

logger = logging.getLogger(__name__)

JOB_SECRET = os.getenv("JOB_SECRET", "")

app = FastAPI(title="NHL Betting Model API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(predictions_router)
app.include_router(performance_router)


@app.on_event("startup")
def on_startup():
    try:
        init_db()
    except Exception as e:
        logger.error(f"DB init failed on startup: {e}")


def _require_secret(x_job_secret: str | None):
    if JOB_SECRET and x_job_secret != JOB_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/health")
def health():
    db_ok = health_check()
    return {"status": "ok" if db_ok else "degraded", "db": db_ok}


@app.post("/jobs/predict")
def job_predict(x_job_secret: str | None = Header(default=None)):
    _require_secret(x_job_secret)
    from src.api.jobs import run_predict_job
    return run_predict_job()


@app.post("/jobs/resolve")
def job_resolve(x_job_secret: str | None = Header(default=None)):
    _require_secret(x_job_secret)
    from src.api.jobs import run_resolve_job
    return run_resolve_job()


@app.post("/jobs/retrain")
def job_retrain(x_job_secret: str | None = Header(default=None)):
    _require_secret(x_job_secret)
    from src.api.jobs import run_retrain_job
    return run_retrain_job()
