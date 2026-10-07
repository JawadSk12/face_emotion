"""Loading and validating the FER-2013 CSV dataset."""

import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMOTION_LABELS = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)
_SPLIT_ALIASES = {
    "training": "train",
    "train": "train",
    "publictest": "validation",
    "validation": "validation",
    "val": "validation",
    "privatetest": "test",
    "test": "test",
}


class FERDataLoader:
    """Read FER-2013 CSV images as uint8 and return integer emotion labels."""

    def __init__(self, csv_path=None, img_size=48):
        if img_size <= 0:
            raise ValueError("img_size must be a positive integer.")
        self.csv_path = Path(csv_path) if csv_path else (
            PROJECT_ROOT / "data" / "fer2013" / "fer2013.csv"
        )
        if not self.csv_path.is_absolute():
            self.csv_path = PROJECT_ROOT / self.csv_path
        self.img_size = img_size

    def load(self):
        """Return ``(images, labels, splits)`` from an FER CSV file.

        Images have shape ``(N, img_size, img_size, 1)`` and dtype ``uint8``.
        Labels are integer IDs in ``[0, 6]``; split names are canonicalized to
        ``train``, ``validation``, and ``test``.
        """
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"FER-2013 CSV not found: {self.csv_path}. "
                "Download the FER-2013 CSV and set paths.fer_csv in config.yaml."
            )

        expected_pixels = self.img_size * self.img_size
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"FER-2013 CSV is empty: {self.csv_path}")
            columns = {name.strip().lower(): index for index, name in enumerate(header)}
            required = {"emotion", "pixels", "usage"}
            missing = required - columns.keys()
            if missing:
                raise ValueError(
                    f"FER-2013 CSV is missing columns: {', '.join(sorted(missing))}."
                )

            row_count = sum(1 for _ in reader)
        if row_count == 0:
            raise ValueError(f"FER-2013 CSV contains no data rows: {self.csv_path}")

        images = np.empty((row_count, expected_pixels), dtype=np.uint8)
        labels = np.empty(row_count, dtype=np.int64)
        splits = np.empty(row_count, dtype="<U10")

        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            header = next(reader)
            columns = {name.strip().lower(): index for index, name in enumerate(header)}
            emotion_col = columns["emotion"]
            pixels_col = columns["pixels"]
            usage_col = columns["usage"]

            for row_index, row in enumerate(reader):
                if len(row) <= max(emotion_col, pixels_col, usage_col):
                    raise ValueError(
                        f"Malformed FER-2013 row {row_index + 2}: expected "
                        "emotion, pixels, and Usage fields."
                    )
                try:
                    label = int(row[emotion_col])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid emotion label at CSV row {row_index + 2}: "
                        f"{row[emotion_col]!r}."
                    ) from exc
                if label < 0 or label >= len(EMOTION_LABELS):
                    raise ValueError(
                        f"Emotion label at CSV row {row_index + 2} must be "
                        f"between 0 and {len(EMOTION_LABELS) - 1}; got {label}."
                    )

                pixel_values = np.fromstring(
                    row[pixels_col], dtype=np.int16, sep=" "
                )
                if pixel_values.size != expected_pixels:
                    raise ValueError(
                        f"CSV row {row_index + 2} has {pixel_values.size} pixels; "
                        f"expected {expected_pixels} for {self.img_size}x{self.img_size}."
                    )
                if np.any((pixel_values < 0) | (pixel_values > 255)):
                    raise ValueError(
                        f"Pixel values at CSV row {row_index + 2} must be in [0, 255]."
                    )

                usage = row[usage_col].strip().replace("_", "").replace(" ", "").lower()
                split = _SPLIT_ALIASES.get(usage)
                if split is None:
                    raise ValueError(
                        f"Unknown dataset split at CSV row {row_index + 2}: "
                        f"{row[usage_col]!r}. Expected Training, PublicTest, "
                        "or PrivateTest."
                    )

                images[row_index] = pixel_values
                labels[row_index] = label
                splits[row_index] = split

        images = images.reshape(-1, self.img_size, self.img_size, 1)
        return images, labels, splits
