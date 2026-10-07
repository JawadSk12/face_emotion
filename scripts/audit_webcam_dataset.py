"""Audit captured webcam crops for missing faces without modifying any files."""

import argparse
import os
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.face_detector import FaceDetector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def audit_dataset(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Webcam dataset directory not found: {data_dir}")

    detector = FaceDetector()
    audited = 0
    unreadable = 0
    no_face = 0
    multiple_faces = 0

    for split in ("train", "validation"):
        for image_path in sorted((data_dir / split).rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            audited += 1
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                unreadable += 1
                print(f"UNREADABLE: {image_path.relative_to(data_dir)}")
                continue

            boxes = detector.detect(image)
            if len(boxes) == 0:
                no_face += 1
                print(f"NO FACE IN CROP: {image_path.relative_to(data_dir)}")
            elif len(boxes) > 1:
                multiple_faces += 1
                print(
                    f"MULTIPLE FACES IN CROP ({len(boxes)}): "
                    f"{image_path.relative_to(data_dir)}"
                )

    print(
        "\nAudit summary: "
        f"{audited} images, {no_face} with no detectable face, "
        f"{multiple_faces} with multiple detections, {unreadable} unreadable."
    )
    print("This audit does not delete or alter any files.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check captured webcam crops for face-detection problems."
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/webcam_emotions",
        help="Dataset root containing train/ and validation/ directories.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    arguments = parse_args()
    audit_dataset(resolve_path(arguments.dataset_dir))
