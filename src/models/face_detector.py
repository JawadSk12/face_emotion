"""Haar-cascade face detection."""

from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

CASCADE_PATH = Path(CONFIG["paths"]["haar_cascade"])
if not CASCADE_PATH.is_absolute():
    CASCADE_PATH = PROJECT_ROOT / CASCADE_PATH


class FaceDetector:
    """Single-subject Haar detector that requires independent cascade agreement."""

    def __init__(self, cascade_path=None):
        self.cascade_path = Path(cascade_path) if cascade_path else CASCADE_PATH
        cascade_paths = [
            self.cascade_path,
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt.xml",
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml",
        ]
        cascade_paths = list(dict.fromkeys(cascade_paths))

        self.detectors = []
        for path in cascade_paths:
            detector = cv2.CascadeClassifier(str(path))
            if not detector.empty():
                self.detectors.append(detector)
        if not self.detectors:
            raise RuntimeError(
                "Could not load any Haar face detector. Checked: "
                f"{', '.join(map(str, cascade_paths))}"
            )

    def detect(self, gray_frame):
        if gray_frame is None or gray_frame.size == 0:
            raise ValueError("Cannot detect faces in an empty image.")

        image_height, image_width = gray_frame.shape[:2]
        minimum_face_size = max(48, round(min(image_height, image_width) * 0.23))
        detection_options = {
            "scaleFactor": 1.05,
            "minNeighbors": 6,
            "minSize": (minimum_face_size, minimum_face_size),
        }
        clusters = []
        for detector_index, detector in enumerate(self.detectors):
            faces = detector.detectMultiScale(gray_frame, **detection_options)
            for face in faces:
                candidate = tuple(map(int, face))
                matching_clusters = [
                    cluster
                    for cluster in clusters
                    if self._boxes_overlap(candidate, cluster["box"])
                ]
                if not matching_clusters:
                    clusters.append(
                        {
                            "box": candidate,
                            "boxes": [candidate],
                            "detectors": {detector_index},
                        }
                    )
                    continue

                cluster = max(
                    matching_clusters,
                    key=lambda item: len(item["detectors"]),
                )
                if detector_index not in cluster["detectors"]:
                    cluster["boxes"].append(candidate)
                    cluster["detectors"].add(detector_index)
                    cluster["box"] = tuple(
                        int(
                            round(
                                sum(box[axis] for box in cluster["boxes"])
                                / len(cluster["boxes"])
                            )
                        )
                        for axis in range(4)
                    )

        confirmed = [
            cluster
            for cluster in clusters
            if len(cluster["detectors"]) >= 2
        ]
        if not confirmed:
            return np.empty((0, 4), dtype=np.int32)

        primary = max(
            confirmed,
            key=lambda cluster: (
                len(cluster["detectors"]),
                cluster["box"][2] * cluster["box"][3],
            ),
        )
        return np.asarray([primary["box"]], dtype=np.int32)

    @staticmethod
    def _boxes_overlap(first, second):
        first_x, first_y, first_width, first_height = first
        second_x, second_y, second_width, second_height = second
        intersection_width = max(
            0,
            min(first_x + first_width, second_x + second_width)
            - max(first_x, second_x),
        )
        intersection_height = max(
            0,
            min(first_y + first_height, second_y + second_height)
            - max(first_y, second_y),
        )
        intersection = intersection_width * intersection_height
        union = (
            first_width * first_height
            + second_width * second_height
            - intersection
        )
        return bool(union and intersection / union > 0.25)
