# backend/websocket_handler.py

import asyncio
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session

from .database import get_db, Event as EventModel
from .schemas import Event as EventSchema

router = APIRouter()


async def fetch_new_events(db: Session, last_id: int) -> List[EventSchema]:
    """
    Fetch events with id > last_id, convert to Pydantic.
    """
    rows = (
        db.query(EventModel)
        .filter(EventModel.id > last_id)
        .order_by(EventModel.id.asc())
        .all()
    )
    return [EventSchema.model_validate(r) for r in rows]


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket, db: Session = Depends(get_db)):
    """
    WebSocket that pushes new events periodically.

    Protocol:
      - On connect: send recent events (last 50)
      - Then: every 2 seconds, check for new events and send them as a batch.

    Frontend can subscribe to:
        ws://localhost:8000/ws/events
    """

    await websocket.accept()
    print("[WS] Client connected")

    # On first connect, send last 50 events
    recent_rows = (
        db.query(EventModel)
        .order_by(EventModel.id.desc())
        .limit(50)
        .all()
    )
    recent_rows = list(reversed(recent_rows))
    events = [EventSchema.model_validate(r) for r in recent_rows]
    last_id = events[-1].id if events else 0

    await websocket.send_json({
        "type": "snapshot",
        "events": [e.model_dump() for e in events]
    })

    try:
        while True:
            await asyncio.sleep(2.0)

            new_events = await fetch_new_events(db, last_id)
            if not new_events:
                continue

            last_id = new_events[-1].id

            await websocket.send_json({
                "type": "append",
                "events": [e.model_dump() for e in new_events]
            })

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print("[WS] Error:", e)
        try:
            await websocket.close()
        except Exception:
            pass
