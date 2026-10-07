import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

import backend.main as backend


class FakeCamera:
    released = False

    def __init__(self, camera_index=0):
        self.frames = [np.zeros((48, 48, 3), dtype=np.uint8)]
        type(self).released = False

    def get_frame(self):
        return self.frames.pop(0) if self.frames else None

    def release(self):
        type(self).released = True


class FakeDetector:
    def process_frame(self, frame):
        return frame


class BackendTests(unittest.TestCase):
    def test_events_endpoint_returns_latest_rows_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.csv"
            log_path.write_text(
                "timestamp,person,emotion\n"
                "2026-10-07T10:00:00+00:00,Alice,Happy\n"
                "2026-10-07T10:00:05+00:00,Bob,Neutral\n",
                encoding="utf-8",
            )

            with patch.object(backend, "LOG_PATH", log_path):
                response = backend.get_events(limit=1)

        self.assertEqual(
            [event.model_dump() for event in response.events],
            [
                {
                    "timestamp": "2026-10-07T10:00:05+00:00",
                    "person": "Face",
                    "emotion": "Neutral",
                }
            ],
        )

    def test_video_feed_streams_jpeg_and_releases_camera(self):
        FakeCamera.released = False
        with (
            patch("src.app.detector.RealTimeDetector", FakeDetector),
            patch("src.app.camera_handler.CameraHandler", FakeCamera),
        ):
            response = backend.video_feed()
            async def consume():
                async for _ in response.body_iterator:
                    pass

            asyncio.run(consume())

        self.assertEqual(response.media_type, "multipart/x-mixed-replace; boundary=frame")
        self.assertTrue(FakeCamera.released)

    def test_frame_generator_yields_jpeg_and_releases_camera(self):
        FakeCamera.released = False
        generator = backend.generate_video_frames(FakeCamera(), FakeDetector())
        chunk = next(generator)
        self.assertIn(b"Content-Type: image/jpeg", chunk)
        with self.assertRaises(StopIteration):
            next(generator)
        self.assertTrue(FakeCamera.released)

    def test_video_feed_reports_camera_unavailable(self):
        def unavailable_camera(camera_index=0):
            raise RuntimeError("Test camera unavailable")

        with (
            patch("src.app.detector.RealTimeDetector", FakeDetector),
            patch("src.app.camera_handler.CameraHandler", unavailable_camera),
        ):
            with self.assertRaises(HTTPException) as raised:
                backend.video_feed()

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("Test camera unavailable", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
