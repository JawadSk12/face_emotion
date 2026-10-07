# backend/database.py

import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# ─────────────────────────────────────────────
# SQLite configuration
# ─────────────────────────────────────────────

# DB: project_root/data/events.db
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "events.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ─────────────────────────────────────────────
# SQLAlchemy model(s)
# ─────────────────────────────────────────────

class Event(Base):
    """
    DB representation of a logged event.
    Mirrors CSV columns:
        timestamp, person, emotion
    """

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True, default=datetime.utcnow)
    person = Column(String(128), index=True, nullable=False)
    emotion = Column(String(64), index=True, nullable=False)


# ─────────────────────────────────────────────
# DB utilities
# ─────────────────────────────────────────────

def init_db():
    """Create tables if not exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency:
    Usage: `db: Session = Depends(get_db)`
    """
    from sqlalchemy.orm import Session

    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
