import unittest

import numpy as np

from src.models.face_recognizer import FaceRecognizerLBPH


class FakeKNN:
    classes_ = np.array([0, 1])
    _y = np.array([0, 0, 1, 1])
    _fit_X = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.95, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    def predict_proba(self, _vector):
        return np.array([[0.9, 0.1]], dtype=np.float64)


class FaceRecognizerTests(unittest.TestCase):
    def make_recognizer(self):
        recognizer = FaceRecognizerLBPH.__new__(FaceRecognizerLBPH)
        recognizer.model = FakeKNN()
        recognizer.label_map = {0: "Enrolled", 1: "Other"}
        recognizer.target_size = (2, 2)
        recognizer.threshold = 0.45
        recognizer.max_distance = 0.5
        recognizer.min_identity_margin = 0.1
        return recognizer

    def test_recognizes_face_with_close_unambiguous_enrollment_match(self):
        recognizer = self.make_recognizer()
        image = np.zeros((2, 2), dtype=np.uint8)

        name, confidence = recognizer.recognize(image)

        self.assertEqual(name, "Enrolled")
        self.assertAlmostEqual(confidence, 90.0)

    def test_rejects_face_too_far_from_every_enrolled_identity(self):
        recognizer = self.make_recognizer()
        recognizer.model.predict_proba = lambda _vector: np.array([[0.9, 0.1]])
        image = np.full((2, 2), 255, dtype=np.uint8)

        name, _ = recognizer.recognize(image)

        self.assertEqual(name, "Unknown")

    def test_rejects_ambiguous_face_between_identities(self):
        recognizer = self.make_recognizer()
        recognizer.max_distance = 2.0
        image = np.full((2, 2), 127, dtype=np.uint8)

        name, _ = recognizer.recognize(image)

        self.assertEqual(name, "Unknown")


if __name__ == "__main__":
    unittest.main()
