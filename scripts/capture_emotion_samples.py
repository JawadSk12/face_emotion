"""Capture intentionally posed, manually labeled webcam facial expressions."""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.face_detector import FaceDetector
from src.preprocessing.data_loader import EMOTION_LABELS


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def crop_face(gray, box):
    x, y, width, height = map(int, box)
    return gray[y : y + height, x : x + width]


def open_camera(camera_index):
    backends = []
    if os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        camera = cv2.VideoCapture(camera_index, backend)
        if camera.isOpened():
            return camera
        camera.release()
    return None


def find_available_cameras(max_index=5):
    available = []
    for camera_index in range(max_index + 1):
        camera = open_camera(camera_index)
        if camera is not None:
            ok, frame = camera.read()
            if ok and frame is not None:
                available.append(camera_index)
            camera.release()
    return available


def capture(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.session):
        raise ValueError(
            "Session name must be 1-64 letters, numbers, underscores, or hyphens."
        )

    data_dir = resolve_path(args.output_dir)
    session_dir = data_dir / args.split / args.session
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError(
            f"Session directory already contains data: {session_dir}. "
            "Choose a new session name; do not reuse a session for validation."
        )

    cascade_path = resolve_path(args.cascade)
    face_detector = FaceDetector(cascade_path)

    camera = open_camera(args.camera_index)
    if camera is None:
        available = find_available_cameras()
        available_text = (
            ", ".join(map(str, available)) if available else "none detected (0-5)"
        )
        raise RuntimeError(
            f"Could not open camera index {args.camera_index}. "
            f"Available camera indexes: {available_text}. "
            "Run again with --camera-index followed by one of those indexes. "
            "Close other apps using the camera and check Windows camera privacy "
            "permissions if no camera indexes are available."
        )

    counts = {emotion: 0 for emotion in EMOTION_LABELS}
    for emotion in EMOTION_LABELS:
        emotion_dir = session_dir / emotion
        if emotion_dir.is_dir():
            counts[emotion] = len(
                [
                    path
                    for path in emotion_dir.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )

    selected_emotion = 0
    print(f"Split: {args.split}; session: {args.session}")
    print("Pose the selected expression; it is a label for appearance, not inner emotion.")
    print("Press 1-7 to select expression, SPACE to save one face crop, q to quit.")

    try:
        while any(count < args.samples_per_class for count in counts.values()):
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped returning frames.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detect(gray)
            face_box = (
                max(faces, key=lambda rect: rect[2] * rect[3])
                if len(faces)
                else None
            )
            display = frame.copy()
            if face_box is not None:
                x, y, width, height = map(int, face_box)
                cv2.rectangle(
                    display, (x, y), (x + width, y + height), (0, 220, 80), 2
                )

            emotion = EMOTION_LABELS[selected_emotion]
            status = (
                f"{args.split.upper()} | {emotion} "
                f"{counts[emotion]}/{args.samples_per_class}"
            )
            instructions = [
                status,
                "1 Angry  2 Disgust  3 Fear  4 Happy  5 Sad  6 Surprise  7 Neutral",
                (
                    "SPACE saves face | q quits"
                    if face_box is not None
                    else "NO FACE: face camera, add light, move closer | q quits"
                ),
            ]
            for line_number, text in enumerate(instructions):
                y = 28 + line_number * 28
                cv2.putText(
                    display,
                    text,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58 if line_number == 0 else 0.46,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    text,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58 if line_number == 0 else 0.46,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow("Capture labeled facial expressions", display)
            key = cv2.waitKey(1) & 0xFF
            if ord("1") <= key <= ord("7"):
                selected_emotion = key - ord("1")
            elif key == ord("q"):
                break
            elif key == ord(" ") and face_box is not None:
                if counts[emotion] >= args.samples_per_class:
                    continue
                face_image = crop_face(gray, face_box)
                if face_image.size == 0:
                    continue
                emotion_dir = session_dir / emotion
                emotion_dir.mkdir(parents=True, exist_ok=True)
                filename = (
                    f"{args.session}_{emotion.lower()}_"
                    f"{counts[emotion] + 1:04d}.png"
                )
                output_path = emotion_dir / filename
                if not cv2.imwrite(str(output_path), face_image):
                    raise OSError(f"Failed to write face crop: {output_path}")
                counts[emotion] += 1
                print(f"Saved {output_path} ({counts[emotion]}/{args.samples_per_class})")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    session_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "session": args.session,
        "split": args.split,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "labeling": "manually posed facial-expression appearance",
        "counts": counts,
    }
    (session_dir / "session.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Session saved to: {session_dir}")
    for emotion, count in counts.items():
        print(f"{emotion:10} {count:4} / {args.samples_per_class}")


def parse_args():
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    parser = argparse.ArgumentParser(
        description="Capture manually labeled face crops for webcam calibration."
    )
    parser.add_argument(
        "--split", choices=("train", "validation"), default="train"
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Unique recording-session name; use a different session for validation.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/webcam_emotions",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--cascade",
        default=config["paths"]["haar_cascade"],
        help="OpenCV cascade path.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=int(config.get("video", {}).get("camera_index", 0)),
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe camera indexes 0-5 and exit.",
    )
    parser.add_argument("--samples-per-class", type=int, default=50)
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be positive.")
    if args.list_cameras:
        return args
    if not args.session:
        parser.error("--session is required unless --list-cameras is used.")
    return args


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    parsed_args = parse_args()
    if parsed_args.list_cameras:
        available = find_available_cameras()
        if available:
            print(f"Available camera indexes: {', '.join(map(str, available))}")
        else:
            print(
                "No working camera was found at indexes 0-5. Check Windows "
                "camera privacy settings and close other apps using the camera."
            )
    else:
        capture(parsed_args)
