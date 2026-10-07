"""Run the trained model in a local OpenCV webcam window."""

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.app.camera_handler import CameraHandler
from src.app.detector import RealTimeDetector


def main():
    detector = RealTimeDetector()
    camera = CameraHandler()
    print("Press 'q' to quit.")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                raise RuntimeError("Camera stopped returning frames.")

            cv2.imshow("Face Emotion System", detector.process_frame(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
