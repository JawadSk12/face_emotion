# backend/schemas.py

from datetime import datetime
from typing import List

from pydantic import BaseModel


# ─────────────────────────────────────────────
# Event schemas
# ─────────────────────────────────────────────

class EventBase(BaseModel):
    person: str
    emotion: str


class EventCreate(EventBase):
    """
    Schema for inserting events into DB.
    Timestamp is optional; backend can fill it.
    """
    timestamp: datetime | None = None


class Event(EventBase):
    """
    Schema for reading events from DB.
    """
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True  # SQLAlchemy model -> Pydantic


class EventsResponse(BaseModel):
    events: List[Event]
