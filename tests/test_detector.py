import unittest

import numpy as np

from src.app.detector import RealTimeDetector


class RecordingLogger:
    def __init__(self):
        self.events = []

    def log(self, person, emotion):
        self.events.append((person, emotion))


class DetectorTests(unittest.TestCase):
    def test_event_logging_is_throttled_by_face_track_and_expression(self):
        detector = RealTimeDetector.__new__(RealTimeDetector)
        detector.event_cooldown_seconds = 5.0
        detector.last_logged_at = {}
        detector.logger = RecordingLogger()

        detector._log_if_due(0, "Happy", 10.0)
        detector._log_if_due(0, "Sad", 11.0)
        detector._log_if_due(0, "Sad", 15.0)

        self.assertEqual(
            detector.logger.events,
            [("Face", "Happy"), ("Face", "Sad")],
        )

    def test_low_confidence_prediction_is_reported_as_uncertain(self):
        class ArrayResult:
            def numpy(self):
                return np.array([[0.10, 0.05, 0.05, 0.55, 0.10, 0.05, 0.10]])

        class FakeModel:
            def __call__(self, _face, training=False):
                return ArrayResult()

        class FakeFaceDetector:
            def detect(self, _gray):
                return [(20, 20, 80, 80)]

        detector = RealTimeDetector.__new__(RealTimeDetector)
        detector.face_detector = FakeFaceDetector()
        detector.emotion_model = FakeModel()
        detector.logger = RecordingLogger()
        detector.last_logged_at = {}
        detector.event_cooldown_seconds = 5.0
        detector.emotion_min_confidence = 0.60
        detector.emotion_logit_adjustments = {}
        detector.last_detections = []
        detector.frame_index = 0
        detector.face_tracks = []
        detector.next_track_id = 0

        detector.process_frame(np.zeros((120, 120, 3), dtype=np.uint8))

        self.assertEqual(detector.last_detections[0]["emotion"], "Uncertain")
        self.assertEqual(detector.last_detections[0]["top_emotion"], "Happy")
        self.assertNotIn("person", detector.last_detections[0])
        self.assertEqual(detector.logger.events, [("Face", "Uncertain")])

    def test_predictions_are_smoothed_for_the_same_face_location(self):
        detector = RealTimeDetector.__new__(RealTimeDetector)
        detector.frame_index = 1
        detector.face_tracks = []
        detector.next_track_id = 0
        first = np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025])
        second = np.array([0.1, 0.1, 0.05, 0.6, 0.05, 0.05, 0.05])

        initial, track_id = detector._smooth_probabilities((10, 10, 50, 50), first)
        detector.frame_index += 1
        smoothed, next_track_id = detector._smooth_probabilities(
            (12, 10, 50, 50), second
        )

        self.assertEqual(track_id, next_track_id)
        np.testing.assert_allclose(initial, first)
        np.testing.assert_allclose(smoothed, (first + second) / 2)


if __name__ == "__main__":
    unittest.main()
