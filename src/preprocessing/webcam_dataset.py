"""Load manually labeled webcam face crops organized by recording session."""

from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.data_loader import EMOTION_LABELS

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class WebcamEmotionDataLoader:
    """Read ``split/session/emotion/image`` crops with session isolation."""

    def __init__(self, data_dir, image_size=48):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_absolute():
            self.data_dir = Path(__file__).resolve().parents[2] / self.data_dir
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        self.image_size = image_size

    def _load_split(self, split):
        split_dir = self.data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {split} webcam dataset directory: {split_dir}"
            )

        images = []
        labels = []
        session_ids = set()
        counts = {emotion: 0 for emotion in EMOTION_LABELS}

        for session_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            session_ids.add(session_dir.name)
            for emotion_id, emotion in enumerate(EMOTION_LABELS):
                emotion_dir = session_dir / emotion
                if not emotion_dir.is_dir():
                    continue
                for image_path in sorted(emotion_dir.iterdir()):
                    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        raise ValueError(f"Unable to read webcam face crop: {image_path}")
                    resized = cv2.resize(
                        image, (self.image_size, self.image_size)
                    )
                    images.append(resized[..., np.newaxis])
                    labels.append(emotion_id)
                    counts[emotion] += 1

        if not images:
            raise ValueError(f"No labeled webcam images found under {split_dir}")
        return (
            np.asarray(images, dtype=np.uint8),
            np.asarray(labels, dtype=np.int64),
            session_ids,
            counts,
        )

    def load(self, minimum_train_per_class=10, minimum_validation_per_class=5):
        """Return webcam train/validation crops and reject session overlap."""
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Webcam dataset not found: {self.data_dir}. "
                "Capture labeled crops before using --webcam-dir."
            )

        train_images, train_labels, train_sessions, train_counts = self._load_split(
            "train"
        )
        validation_images, validation_labels, validation_sessions, validation_counts = (
            self._load_split("validation")
        )

        overlap = train_sessions & validation_sessions
        if overlap:
            raise ValueError(
                "A recording session cannot appear in both train and validation: "
                f"{', '.join(sorted(overlap))}. Capture validation expressions in "
                "a separate camera session."
            )

        missing_train = [
            emotion
            for emotion, count in train_counts.items()
            if count < minimum_train_per_class
        ]
        missing_validation = [
            emotion
            for emotion, count in validation_counts.items()
            if count < minimum_validation_per_class
        ]
        if missing_train:
            raise ValueError(
                f"Need at least {minimum_train_per_class} training crops per "
                f"emotion; insufficient: {', '.join(missing_train)}."
            )
        if missing_validation:
            raise ValueError(
                f"Need at least {minimum_validation_per_class} validation crops "
                f"per emotion; insufficient: {', '.join(missing_validation)}."
            )

        return {
            "train_images": train_images,
            "train_labels": train_labels,
            "validation_images": validation_images,
            "validation_labels": validation_labels,
            "train_sessions": train_sessions,
            "validation_sessions": validation_sessions,
            "train_counts": train_counts,
            "validation_counts": validation_counts,
        }
