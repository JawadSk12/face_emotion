import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.capture_emotion_samples import crop_face
from src.preprocessing.data_loader import EMOTION_LABELS
from src.preprocessing.webcam_dataset import WebcamEmotionDataLoader


class WebcamEmotionDataLoaderTests(unittest.TestCase):
    def test_capture_crop_matches_live_detector_face_box(self):
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)

        crop = crop_face(image, (2, 3, 4, 5))

        np.testing.assert_array_equal(crop, image[3:8, 2:6])

    def write_split(self, root, split, session, counts=None, invalid=False):
        counts = counts or {emotion: 1 for emotion in EMOTION_LABELS}
        for emotion, count in counts.items():
            emotion_dir = root / split / session / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image_path = emotion_dir / f"{index}.png"
                if invalid:
                    image_path.write_bytes(b"not an image")
                else:
                    self.assertTrue(
                        cv2.imwrite(
                            str(image_path),
                            np.full((24, 24), index, dtype=np.uint8),
                        )
                    )

    def test_loads_images_and_labels_for_disjoint_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_split(root, "train", "morning")
            self.write_split(root, "validation", "evening")

            data = WebcamEmotionDataLoader(root).load(
                minimum_train_per_class=1,
                minimum_validation_per_class=1,
            )

        self.assertEqual(data["train_images"].shape, (7, 48, 48, 1))
        self.assertEqual(data["validation_images"].shape, (7, 48, 48, 1))
        self.assertEqual(data["train_images"].dtype, np.uint8)
        np.testing.assert_array_equal(data["train_labels"], np.arange(7))
        self.assertEqual(data["train_sessions"], {"morning"})
        self.assertEqual(data["validation_sessions"], {"evening"})

    def test_rejects_session_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_split(root, "train", "same-session")
            self.write_split(root, "validation", "same-session")

            with self.assertRaisesRegex(ValueError, "cannot appear in both"):
                WebcamEmotionDataLoader(root).load(
                    minimum_train_per_class=1,
                    minimum_validation_per_class=1,
                )

    def test_rejects_insufficient_per_class_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_split(root, "train", "train-session")
            self.write_split(root, "validation", "validation-session")

            with self.assertRaisesRegex(ValueError, "at least 2 training crops"):
                WebcamEmotionDataLoader(root).load(
                    minimum_train_per_class=2,
                    minimum_validation_per_class=1,
                )

    def test_rejects_unreadable_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_split(root, "train", "train-session", invalid=True)

            with self.assertRaisesRegex(ValueError, "Unable to read"):
                WebcamEmotionDataLoader(root)._load_split("train")

    def test_rejects_missing_dataset_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_root = Path(directory) / "missing"

            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                WebcamEmotionDataLoader(missing_root).load()


if __name__ == "__main__":
    unittest.main()
