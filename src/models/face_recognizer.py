# src/models/face_recognizer.py

import pickle
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.metrics import pairwise_distances

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

REC_PATH = Path(CONFIG["paths"]["face_recognizer"])
LABEL_PATH = Path(CONFIG["paths"]["face_labels"])
if not REC_PATH.is_absolute():
    REC_PATH = PROJECT_ROOT / REC_PATH
if not LABEL_PATH.is_absolute():
    LABEL_PATH = PROJECT_ROOT / LABEL_PATH


class FaceRecognizerLBPH:
    """
    KNN-based face recognizer (replaces LBPH).
    Kept class name 'FaceRecognizerLBPH' so the rest of the project works unchanged.

    Expects:
      - Pickled scikit-learn classifier at REC_PATH
      - Label map (id -> name) at LABEL_PATH
    """

    def __init__(
        self,
        target_size=(100, 100),
        threshold=0.45,
        max_distance=5.5,
        min_identity_margin=1.0,
    ):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        if max_distance <= 0 or min_identity_margin < 0:
            raise ValueError("Identity distance thresholds must be non-negative.")
        if not REC_PATH.exists():
            raise FileNotFoundError(f"Recognizer not found: {REC_PATH}")

        if not LABEL_PATH.exists():
            raise FileNotFoundError(f"Label map missing: {LABEL_PATH}")

        with open(REC_PATH, "rb") as f:
            self.model = pickle.load(f)

        with open(LABEL_PATH, "rb") as f:
            self.label_map = pickle.load(f)  # id -> name

        self.target_size = target_size
        self.threshold = threshold
        self.max_distance = max_distance
        self.min_identity_margin = min_identity_margin

    def _prep_face_vector(self, gray_face):
        if gray_face.ndim == 3:
            gray_face = cv2.cvtColor(gray_face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(gray_face, self.target_size)
        vec = face_resized.astype("float32").flatten() / 255.0
        return vec

    def recognize(self, face_gray):
        if face_gray is None or face_gray.size == 0:
            return "Unknown", None

        vec = np.expand_dims(self._prep_face_vector(face_gray), axis=0)
        probabilities = self.model.predict_proba(vec)[0]
        probability_by_label = {
            int(label): float(probability)
            for label, probability in zip(self.model.classes_, probabilities)
        }

        training_labels = np.asarray(self.model._y).reshape(-1)
        training_vectors = np.asarray(self.model._fit_X)
        if training_vectors.shape[1] != vec.shape[1]:
            raise ValueError(
                "Saved face recognizer image size does not match its input vectors. "
                "Retrain it with scripts/train_lbph.py."
            )
        distances = pairwise_distances(vec, training_vectors, metric="euclidean")[0]
        class_distances = {
            int(label): float(np.min(distances[training_labels == label]))
            for label in self.model.classes_
            if np.any(training_labels == label)
        }
        nearest = sorted(class_distances.items(), key=lambda item: item[1])
        if not nearest:
            return "Unknown", None

        label_id, nearest_distance = nearest[0]
        confidence = probability_by_label.get(label_id, 0.0)
        if len(nearest) > 1:
            identity_margin = nearest[1][1] - nearest_distance
        else:
            identity_margin = float("inf")

        if (
            nearest_distance > self.max_distance
            or identity_margin < self.min_identity_margin
            or confidence < self.threshold
        ):
            return "Unknown", confidence * 100.0

        return self.label_map.get(label_id, "Unknown"), confidence * 100.0
