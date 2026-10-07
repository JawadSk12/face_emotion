"""Manually capture distinct, consented enrollment images for one person."""

import os
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

KNOWN_DIR = Path(CONFIG["paths"]["known_faces_dir"])
CASCADE_PATH = Path(CONFIG["paths"]["haar_cascade"])
if not KNOWN_DIR.is_absolute():
    KNOWN_DIR = PROJECT_ROOT / KNOWN_DIR
if not CASCADE_PATH.is_absolute():
    CASCADE_PATH = PROJECT_ROOT / CASCADE_PATH


def main():
    name = input("Enter the enrolled person's name: ").strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("Enter a valid, non-empty folder name.")

    person_dir = KNOWN_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar Cascade: {CASCADE_PATH}")

    camera = cv2.VideoCapture(int(CONFIG.get("video", {}).get("camera_index", 0)))
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    existing = [
        path for path in person_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    existing_indices = []
    for path in existing:
        try:
            existing_indices.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    next_index = max(existing_indices, default=-1) + 1
    captured = 0
    print("Look toward the camera, vary your expression and head angle.")
    print("Press SPACE to capture one face; press q to quit.")

    try:
        while captured < 50:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped returning frames.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=7, minSize=(80, 80)
            )
            largest_face = None
            if len(faces):
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, width, height = largest_face
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Captured {captured}/50 | SPACE capture | q quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Capture enrollment faces", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and largest_face is not None:
                x, y, width, height = largest_face
                face = gray[y : y + height, x : x + width]
                output = person_dir / f"{name}_{next_index:03d}.jpg"
                if not cv2.imwrite(str(output), face):
                    raise OSError(f"Could not save captured face: {output}")
                print(f"Saved {output}")
                next_index += 1
                captured += 1
                cv2.waitKey(300)
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Captured {captured} images in {person_dir}.")
    print("Rebuild the identity classifier with: python scripts\\train_lbph.py")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
