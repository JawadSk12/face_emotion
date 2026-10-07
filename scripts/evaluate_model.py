"""Evaluate the active emotion checkpoint without starting or changing training."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluate(args):
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    from src.preprocessing.data_loader import EMOTION_LABELS, FERDataLoader
    from src.models.emotion_cnn import apply_class_logit_adjustments

    dataset_path = resolve_path(
        args.dataset or config["paths"].get(
            "fer_csv", "data/fer2013/fer2013.csv"
        )
    )
    model_path = resolve_path(
        args.model or config["paths"].get(
            "emotion_model", "models/emotion_model.h5"
        )
    )
    output_dir = resolve_path(args.output_dir or "models/evaluation")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained emotion model not found: {model_path}")

    images, labels, splits = FERDataLoader(dataset_path).load()
    model = tf.keras.models.load_model(model_path, compile=False)
    if (
        model.input_shape[1:] != (48, 48, 1)
        or model.output_shape[-1] != len(EMOTION_LABELS)
    ):
        raise ValueError(
            f"Model must accept 48x48 grayscale faces and return "
            f"{len(EMOTION_LABELS)} emotion scores."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": str(model_path),
        "dataset": str(dataset_path),
        "emotion_labels": list(EMOTION_LABELS),
        "emotion_logit_adjustments": config.get("inference", {}).get(
            "emotion_logit_adjustments", {}
        ),
        "splits": {},
    }
    label_ids = np.arange(len(EMOTION_LABELS))
    for split in ("validation", "test"):
        mask = splits == split
        if not np.any(mask):
            raise ValueError(f"FER-2013 dataset has no '{split}' split.")
        split_images = images[mask].astype(np.float32) / 255.0
        split_labels = labels[mask]
        probabilities = model.predict(
            split_images, batch_size=args.batch_size, verbose=1
        )
        probabilities = apply_class_logit_adjustments(
            probabilities,
            EMOTION_LABELS,
            config.get("inference", {}).get("emotion_logit_adjustments", {}),
        )
        predictions = np.argmax(probabilities, axis=1)
        report = classification_report(
            split_labels,
            predictions,
            labels=label_ids,
            target_names=list(EMOTION_LABELS),
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(
            split_labels, predictions, labels=label_ids
        )
        summary["splits"][split] = {
            "samples": int(np.count_nonzero(mask)),
            "accuracy": float(accuracy_score(split_labels, predictions)),
            "classification_report": report,
        }
        np.savetxt(
            output_dir / f"{split}_confusion_matrix.csv",
            matrix,
            delimiter=",",
            fmt="%d",
            header=",".join(EMOTION_LABELS),
            comments="",
        )
        print(
            f"{split.title():10} accuracy: "
            f"{summary['splits'][split]['accuracy']:.4f}"
        )
        print(
            f"{split.title():10} macro F1: "
            f"{report['macro avg']['f1-score']:.4f}"
        )

    summary_path = output_dir / "evaluation_results.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Evaluation results saved to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the configured trained model; this does not train."
    )
    parser.add_argument("--dataset", help="FER-2013 CSV (defaults to config.yaml).")
    parser.add_argument("--model", help="Model path (defaults to config.yaml).")
    parser.add_argument("--output-dir", help="Evaluation output directory.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    return args


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    evaluate(parse_args())
