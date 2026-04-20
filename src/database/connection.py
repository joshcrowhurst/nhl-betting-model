"""
Cloud SQL (PostgreSQL) connection via SQLAlchemy.
In Cloud Run, connects via Unix socket using the Cloud SQL Auth Proxy
(automatically available as a sidecar when INSTANCE_CONNECTION_NAME is set).
Locally, connects via DATABASE_URL env var for development.
"""

import os
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    instance_connection = os.getenv("INSTANCE_CONNECTION_NAME")
    db_user = os.getenv("DB_USER", "nhl")
    db_pass = os.getenv("DB_PASS", "")
    db_name = os.getenv("DB_NAME", "nhl_betting")
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        # Local dev override
        url = database_url
    elif instance_connection:
        # Cloud Run + Cloud SQL Auth Proxy (unix socket)
        url = (
            f"postgresql+psycopg2://{db_user}:{db_pass}@/{db_name}"
            f"?host=/cloudsql/{instance_connection}"
        )
    else:
        raise RuntimeError(
            "Set DATABASE_URL (local) or INSTANCE_CONNECTION_NAME + DB_USER/PASS/NAME (Cloud Run)"
        )

    _engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=2)
    logger.info("Database engine created")
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False, expire_on_commit=False)
    return _SessionLocal


@contextmanager
def get_db() -> Session:
    """Context manager for a DB session — auto-commits or rolls back."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables ensured")


def health_check() -> bool:
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return False
