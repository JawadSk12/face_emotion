"""Check the local FER-2013 data or download it through the Kaggle CLI."""

import argparse
import csv
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "fer2013" / "fer2013.csv"
COMPETITION = "challenges-in-representation-learning-facial-expression-recognition-challenge"
sys.path.insert(0, str(PROJECT_ROOT))


def _copy_fer_csv_stream(stream, destination):
    try:
        first_line = stream.readline()
        header = next(csv.reader([first_line.decode("utf-8-sig")]))
    except (StopIteration, UnicodeDecodeError):
        return False
    if not {"emotion", "pixels", "usage"} <= {
        column.strip().lower() for column in header
    }:
        return False
    with destination.open("wb") as output:
        output.write(first_line)
        shutil.copyfileobj(stream, output)
    return True


def _copy_tar_csv(tar_stream, destination):
    with tarfile.open(fileobj=tar_stream, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".csv"):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            if _copy_fer_csv_stream(source, destination):
                return True
    return False


def copy_fer_csv(archive_path, destination):
    """Find and copy the FER CSV from a Kaggle CSV, ZIP, or tar.gz download."""
    archive_path = Path(archive_path)
    if archive_path.suffix.lower() == ".csv":
        with archive_path.open("rb") as source:
            return _copy_fer_csv_stream(source, destination)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            csv_names = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]
            for name in csv_names:
                with archive.open(name) as source:
                    if _copy_fer_csv_stream(source, destination):
                        return True
            for name in archive.namelist():
                if not name.lower().endswith((".tar.gz", ".tgz")):
                    continue
                with archive.open(name) as source:
                    if _copy_tar_csv(source, destination):
                        return True
        return False

    if tarfile.is_tarfile(archive_path):
        with archive_path.open("rb") as source:
            return _copy_tar_csv(source, destination)
    return False


def download_dataset(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fer2013-") as temp_dir:
        download_dir = Path(temp_dir)
        try:
            subprocess.run(
                [
                    "kaggle",
                    "competitions",
                    "download",
                    "-c",
                    COMPETITION,
                    "-p",
                    str(download_dir),
                ],
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Kaggle CLI was not found. Install it with 'pip install kaggle', "
                "configure your Kaggle API credentials, and accept the FER-2013 "
                "competition rules before downloading."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Kaggle download failed. Check your API credentials and confirm "
                "you accepted the FER-2013 competition rules."
            ) from exc

        staged_csv = download_dir / "fer2013.csv"
        found = any(
            copy_fer_csv(archive, staged_csv)
            for archive in download_dir.iterdir()
            if archive.is_file()
        )
        if not found:
            raise RuntimeError(
                "The downloaded Kaggle archive did not contain a FER-2013 CSV "
                "with emotion, pixels, and Usage columns."
            )

        from src.preprocessing.data_loader import FERDataLoader

        FERDataLoader(staged_csv).load()
        staged_csv.replace(destination)


def main():
    parser = argparse.ArgumentParser(
        description="Validate the local FER-2013 CSV or fetch it from Kaggle."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the FER-2013 competition archive using the Kaggle CLI.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV path (default: data/fer2013/fer2013.csv).",
    )
    args = parser.parse_args()
    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    if args.download:
        download_dataset(dataset_path)

    from src.preprocessing.data_loader import EMOTION_LABELS, FERDataLoader

    images, labels, splits = FERDataLoader(dataset_path).load()
    print(f"Validated FER-2013 CSV: {dataset_path}")
    print(f"Total images: {len(images)}")
    for split in ("train", "validation", "test"):
        print(f"{split.title():10} {int((splits == split).sum()):6} images")
    print("Emotion counts:")
    for label_id, label_name in enumerate(EMOTION_LABELS):
        print(f"  {label_name:8} {int((labels == label_id).sum()):6}")


if __name__ == "__main__":
    main()
