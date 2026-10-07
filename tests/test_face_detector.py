import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.models.face_detector import FaceDetector


class FaceDetectorTests(unittest.TestCase):
    def test_requires_multiple_cascades_to_agree_on_a_face(self):
        frame = np.arange(64, dtype=np.uint8).reshape(8, 8)
        face_boxes = (
            np.array([[10, 10, 60, 60], [1, 1, 48, 48]]),
            np.array([[11, 10, 61, 60]]),
            np.array([[10, 11, 59, 59]]),
        )

        class FakeCascade:
            def __init__(self, faces):
                self.faces = faces
                self.frames = []
                self.options = []

            def empty(self):
                return False

            def detectMultiScale(self, image, **options):
                self.frames.append(image.copy())
                self.options.append(options)
                return self.faces

        cascades = [FakeCascade(boxes) for boxes in face_boxes]
        with patch(
            "src.models.face_detector.cv2.CascadeClassifier",
            side_effect=cascades,
        ):
            detector = FaceDetector("fake-cascade.xml")

        faces = detector.detect(frame)

        np.testing.assert_array_equal(faces, [[10, 10, 60, 60]])
        for cascade in cascades:
            np.testing.assert_array_equal(cascade.frames[0], frame)
            self.assertEqual(cascade.options[0]["minNeighbors"], 6)
            self.assertEqual(cascade.options[0]["minSize"], (48, 48))

    def test_rejects_a_candidate_reported_by_only_one_cascade(self):
        class FakeCascade:
            def __init__(self, candidates):
                self.candidates = candidates

            def detectMultiScale(self, _frame, **_options):
                return self.candidates

        detector = FaceDetector.__new__(FaceDetector)
        detector.detectors = [
            FakeCascade(np.array([[10, 10, 90, 100]])),
            FakeCascade(np.empty((0, 4), dtype=np.int32)),
            FakeCascade(np.empty((0, 4), dtype=np.int32)),
        ]

        result = detector.detect(np.zeros((240, 320), dtype=np.uint8))

        self.assertEqual(result.shape, (0, 4))

    def test_keeps_only_largest_confirmed_face_in_single_subject_mode(self):
        small_face = np.array([[150, 20, 70, 80]])
        large_face = np.array([[10, 10, 90, 100]])
        cascades = [
            np.concatenate((large_face, small_face)),
            np.concatenate((large_face, small_face)),
            np.concatenate((large_face, small_face)),
        ]

        class FakeCascade:
            def __init__(self, candidates):
                self.candidates = candidates

            def detectMultiScale(self, _frame, **_options):
                return self.candidates

        detector = FaceDetector.__new__(FaceDetector)
        detector.detectors = [FakeCascade(boxes) for boxes in cascades]

        result = detector.detect(np.zeros((240, 320), dtype=np.uint8))

        self.assertEqual(result.shape, (1, 4))
        np.testing.assert_array_equal(result[0], [10, 10, 90, 100])

    def test_empty_frame_rejected_before_cascade_detection(self):
        class FakeCascade:
            def empty(self):
                return False

        with patch(
            "src.models.face_detector.cv2.CascadeClassifier",
            return_value=FakeCascade(),
        ):
            detector = FaceDetector("fake-cascade.xml")

        with self.assertRaisesRegex(ValueError, "empty image"):
            detector.detect(np.empty((0, 0), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
