"""Train the optional known-person nearest-neighbor recognizer."""

import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_face_images_for_knn(base_dir, target_size=(100, 100)):
    """Load images sorted by identity and discard exact duplicate face vectors."""
    base_dir = Path(base_dir)
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Known faces directory not found: {base_dir}")

    face_vectors = []
    labels = []
    label_map = {}
    duplicates_removed = 0
    people = sorted(path for path in base_dir.iterdir() if path.is_dir())

    for label_id, person_folder in enumerate(people):
        label_map[label_id] = person_folder.name
        seen_vectors = set()
        for image_path in sorted(person_folder.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[WARN] Unable to read image: {image_path}")
                continue

            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            resized = cv2.resize(gray, target_size)
            vector = resized.astype(np.float32).flatten() / 255.0
            fingerprint = vector.tobytes()
            if fingerprint in seen_vectors:
                duplicates_removed += 1
                continue
            seen_vectors.add(fingerprint)
            face_vectors.append(vector)
            labels.append(label_id)

        if not any(label == label_id for label in labels):
            label_map.pop(label_id)
            print(f"[WARN] No readable unique face images for {person_folder.name}.")

    if not face_vectors:
        raise RuntimeError(
            f"No unique face images found in {base_dir}. "
            "Capture consented face images before training recognition."
        )

    X = np.asarray(face_vectors, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    counts = {label_id: int(np.count_nonzero(y == label_id)) for label_id in np.unique(y)}
    if any(count < 3 for count in counts.values()):
        raise ValueError(
            "Each enrolled person needs at least three distinct readable images. "
            f"Unique image counts by identity: {counts}"
        )
    print(f"[INFO] Loaded {len(X)} unique face images from {len(counts)} people.")
    print(f"[INFO] Removed {duplicates_removed} exact duplicate images.")
    return X, y, label_map


def train():
    paths = CONFIG["paths"]
    known_dir = resolve_path(paths["known_faces_dir"])
    recognizer_path = resolve_path(paths["face_recognizer"])
    labels_path = resolve_path(paths["face_labels"])

    X, y, label_map = load_face_images_for_knn(known_dir)
    neighbors = min(3, min(int(np.count_nonzero(y == label)) for label in np.unique(y)))
    model = KNeighborsClassifier(
        n_neighbors=neighbors,
        weights="distance",
        metric="euclidean",
        n_jobs=-1,
    )
    model.fit(X, y)

    recognizer_path.parent.mkdir(parents=True, exist_ok=True)
    with recognizer_path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    with labels_path.open("wb") as file:
        pickle.dump(label_map, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved recognizer to {recognizer_path}")
    print(f"[INFO] Saved identity labels to {labels_path}")
    print(
        "[INFO] Live identity predictions reject faces outside the configured "
        "distance/margin thresholds as Unknown."
    )


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    train()
