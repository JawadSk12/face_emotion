"""Real-time face detection and facial-expression inference."""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml
from tensorflow.keras.models import load_model

from src.models.emotion_cnn import apply_class_logit_adjustments
from src.models.face_detector import FaceDetector
from src.utils.logger import EventLogger
from src.utils.visualizer import draw_face_box, put_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

EMOTION_LABELS = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)
UNCERTAIN_LABEL = "Uncertain"
MODEL_PATH = Path(CONFIG["paths"]["emotion_model"])
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH


class RealTimeDetector:
    def __init__(self, event_logger=None, event_cooldown_seconds=None):
        thresholds = CONFIG.get("thresholds", {})
        if event_cooldown_seconds is None:
            event_cooldown_seconds = float(
                thresholds.get("event_cooldown_seconds", 5.0)
            )
        if event_cooldown_seconds < 0:
            raise ValueError("event_cooldown_seconds cannot be negative.")

        self.emotion_min_confidence = float(
            thresholds.get("emotion_min_confidence", 0.60)
        )
        self.emotion_logit_adjustments = CONFIG.get("inference", {}).get(
            "emotion_logit_adjustments", {}
        )
        if not 0 <= self.emotion_min_confidence <= 1:
            raise ValueError("emotion_min_confidence must be between 0 and 1.")
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Emotion model not found: {MODEL_PATH}. "
                "Place a trained model at the configured path before inference."
            )

        self.face_detector = FaceDetector()
        self.emotion_model = load_model(MODEL_PATH, compile=False)
        if (
            self.emotion_model.input_shape[1:] != (48, 48, 1)
            or self.emotion_model.output_shape[-1] != len(EMOTION_LABELS)
        ):
            raise ValueError(
                f"Emotion model at {MODEL_PATH} must accept 48x48 grayscale "
                f"faces and produce {len(EMOTION_LABELS)} class scores."
            )

        self.logger = event_logger or EventLogger()
        self.event_cooldown_seconds = event_cooldown_seconds
        self.last_logged_at = {}
        self.last_detections = []
        self.frame_index = 0
        self.face_tracks = []
        self.next_track_id = 0

    def _prep_emotion(self, gray_face):
        face = cv2.resize(gray_face, (48, 48))
        face = face.astype("float32") / 255.0
        return face[np.newaxis, ..., np.newaxis]

    def _log_if_due(self, track_id, emotion, now):
        key = (track_id, emotion)
        last_time = self.last_logged_at.get(key)
        if last_time is None or now - last_time >= self.event_cooldown_seconds:
            self.logger.log("Face", emotion)
            self.last_logged_at[key] = now

    @staticmethod
    def _box_iou(first, second):
        first_x, first_y, first_width, first_height = first
        second_x, second_y, second_width, second_height = second
        left = max(first_x, second_x)
        top = max(first_y, second_y)
        right = min(first_x + first_width, second_x + second_width)
        bottom = min(first_y + first_height, second_y + second_height)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = first_width * first_height + second_width * second_height - intersection
        return intersection / union if union else 0.0

    def _smooth_probabilities(self, box, probabilities):
        """Smooth predictions over nearby frames for the same face location."""
        best_track = None
        best_iou = 0.15
        for track in self.face_tracks:
            if track["last_seen"] == self.frame_index:
                continue
            if self.frame_index - track["last_seen"] > 5:
                continue
            overlap = self._box_iou(box, track["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_track = track

        if best_track is None:
            best_track = {
                "id": self.next_track_id,
                "box": box,
                "scores": deque(maxlen=5),
                "last_seen": self.frame_index,
            }
            self.next_track_id += 1
            self.face_tracks.append(best_track)

        best_track["box"] = box
        best_track["last_seen"] = self.frame_index
        best_track["scores"].append(probabilities)
        self.face_tracks = [
            track
            for track in self.face_tracks
            if self.frame_index - track["last_seen"] <= 5
        ]
        return np.mean(best_track["scores"], axis=0), best_track["id"]

    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Cannot process an empty video frame.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detect(gray)
        self.frame_index += 1
        detections = []
        now = time.monotonic()

        for x, y, width, height in faces:
            x, y, width, height = map(int, (x, y, width, height))
            face = gray[y : y + height, x : x + width]
            probabilities = self.emotion_model(
                self._prep_emotion(face), training=False
            ).numpy()[0]
            probabilities = apply_class_logit_adjustments(
                probabilities[np.newaxis, :],
                EMOTION_LABELS,
                self.emotion_logit_adjustments,
            )[0]
            box = (x, y, width, height)
            probabilities, track_id = self._smooth_probabilities(
                box, probabilities
            )

            emotion_index = int(np.argmax(probabilities))
            top_emotion = EMOTION_LABELS[emotion_index]
            emotion_confidence = float(probabilities[emotion_index])
            emotion = (
                top_emotion
                if emotion_confidence >= self.emotion_min_confidence
                else UNCERTAIN_LABEL
            )
            detection = {
                "emotion": emotion,
                "top_emotion": top_emotion,
                "emotion_confidence": emotion_confidence,
                "emotion_scores": probabilities.tolist(),
                "track_id": track_id,
                "box": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                },
            }
            detections.append(detection)

            draw_face_box(frame, x, y, width, height)
            label = (
                f"Uncertain | top estimate: {top_emotion} "
                f"({emotion_confidence * 100:.0f}%)"
                if emotion == UNCERTAIN_LABEL
                else f"{emotion} ({emotion_confidence * 100:.0f}%)"
            )
            put_label(frame, label, x, max(20, y - 10))
            self._log_if_due(track_id, emotion, now)

        active_track_ids = {track["id"] for track in self.face_tracks}
        self.last_logged_at = {
            key: timestamp
            for key, timestamp in self.last_logged_at.items()
            if key[0] in active_track_ids
        }
        self.last_detections = detections
        return frame
