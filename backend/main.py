"""FastAPI event API and browser-accessible webcam stream."""

import csv
import logging
from collections import deque
from pathlib import Path
from typing import List

import cv2
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

LOG_PATH = Path(CONFIG["paths"]["logs_csv"])
if not LOG_PATH.is_absolute():
    LOG_PATH = PROJECT_ROOT / LOG_PATH
MODEL_PATH = Path(CONFIG["paths"]["emotion_model"])
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH
CAMERA_INDEX = int(CONFIG.get("video", {}).get("camera_index", 0))
LOGGER = logging.getLogger(__name__)


class Event(BaseModel):
    timestamp: str
    person: str
    emotion: str


class EventsResponse(BaseModel):
    events: List[Event]


app = FastAPI(
    title="Face Emotion System",
    description=(
        "Local facial-expression demo API with recent detection events and an "
        "MJPEG webcam stream."
    ),
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_events_from_csv(limit=100):
    """Read the newest CSV events without retaining the full file in memory."""
    if not LOG_PATH.exists():
        return []

    recent_rows = deque(maxlen=limit)
    with LOG_PATH.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            timestamp = row.get("timestamp", "")
            emotion = row.get("emotion", "")
            if not timestamp and not emotion:
                continue
            recent_rows.append(
                Event(timestamp=timestamp, person="Face", emotion=emotion)
            )
    return list(recent_rows)


def generate_video_frames(camera, detector):
    """Yield browser-compatible JPEG frames and release the camera on disconnect."""
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                LOGGER.warning("Camera stopped returning frames.")
                return
            processed = detector.process_frame(frame)
            encoded, jpeg = cv2.imencode(
                ".jpg", processed, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if not encoded:
                raise RuntimeError("OpenCV failed to encode a camera frame as JPEG.")
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-store\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
    finally:
        camera.release()


@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "message": "Face Emotion System Backend is running.",
        "emotion_model_ready": MODEL_PATH.is_file(),
        "video_feed": "/video_feed",
        "events": "/events",
    }


@app.get("/events", response_model=EventsResponse, tags=["events"])
def get_events(limit: int = Query(100, ge=1, le=1000)):
    return EventsResponse(events=read_events_from_csv(limit=limit))


@app.get("/video_feed", tags=["video"])
def video_feed():
    """Stream camera frames annotated by the active trained emotion model."""
    from src.app.camera_handler import CameraHandler
    from src.app.detector import RealTimeDetector

    try:
        detector = RealTimeDetector()
        camera = CameraHandler(camera_index=CAMERA_INDEX)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.exception("Unable to start video detection.")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return StreamingResponse(
        generate_video_frames(camera, detector),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
