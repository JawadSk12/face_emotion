"""OpenCV camera capture helper."""

import cv2


class CameraHandler:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                f"Cannot access camera index {camera_index}. "
                "Check camera permissions and that no other application is using it."
            )

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
