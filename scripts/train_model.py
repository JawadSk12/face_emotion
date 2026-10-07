"""Fine-tune, train, and evaluate the FER-2013 facial-expression CNN."""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Input,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from tensorflow.keras.models import Model, load_model
from src.models.emotion_cnn import apply_class_logit_adjustments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_dataset(images, labels, batch_size, training, seed):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(labels), seed=seed, reshuffle_each_iteration=True
        )

    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    return (
        dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def make_finetuning_model(base_model):
    """Add training-only augmentation around a compatible pre-trained model."""
    inputs = Input(shape=(48, 48, 1), name="face_image")
    augmented = RandomFlip("horizontal", name="random_flip")(inputs)
    augmented = RandomRotation(0.04, name="random_rotation")(augmented)
    augmented = RandomTranslation(
        0.04, 0.04, name="random_translation"
    )(augmented)
    augmented = RandomZoom(0.08, name="random_zoom")(augmented)
    outputs = base_model(augmented)
    return Model(inputs=inputs, outputs=outputs, name="fer2013_finetuned_cnn")


def read_last_epoch(history_path):
    if not history_path.exists():
        return 0
    with history_path.open("r", newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("epoch")]
    return int(float(rows[-1]["epoch"])) + 1 if rows else 0


def plot_training_history(history, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric in zip(axes, ("accuracy", "loss")):
        axis.plot(history.history[metric], label=f"train {metric}")
        validation_metric = f"val_{metric}"
        if validation_metric in history.history:
            axis.plot(history.history[validation_metric], label=validation_metric)
        axis.set_xlabel("Epoch in this run")
        axis.set_ylabel(metric.title())
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def evaluate_metrics(
    model,
    dataset,
    labels,
    label_ids,
    label_names,
    logit_adjustments=None,
):
    loss, _ = model.evaluate(dataset, verbose=1)
    probabilities = model.predict(dataset, verbose=1)
    probabilities = apply_class_logit_adjustments(
        probabilities, label_names, logit_adjustments or {}
    )
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        labels=label_ids,
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=label_ids)
    return float(loss), float(accuracy), report, matrix


class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """Save by balanced class performance rather than majority-heavy accuracy."""

    def __init__(
        self,
        validation_dataset,
        validation_labels,
        label_ids,
        checkpoint_path,
        logit_adjustments=None,
    ):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.validation_labels = validation_labels
        self.label_ids = label_ids
        self.checkpoint_path = Path(checkpoint_path)
        self.logit_adjustments = logit_adjustments or {}
        self.best_macro_f1 = -1.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probabilities = self.model.predict(
            self.validation_dataset, verbose=0
        )
        probabilities = apply_class_logit_adjustments(
            probabilities, EMOTION_LABELS, self.logit_adjustments
        )
        predictions = np.argmax(probabilities, axis=1)
        macro_f1 = float(
            f1_score(
                self.validation_labels,
                predictions,
                labels=self.label_ids,
                average="macro",
                zero_division=0,
            )
        )
        logs["val_macro_f1"] = macro_f1
        print(f"\nEpoch {epoch + 1}: val_macro_f1: {macro_f1:.4f}")
        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            self.model.save(self.checkpoint_path)
            print(
                f"Saved new best balanced-class checkpoint "
                f"(macro F1 {macro_f1:.4f}) to {self.checkpoint_path}"
            )


def webcam_candidate_is_acceptable(
    accuracy,
    report,
    label_names,
    minimum_accuracy,
    minimum_class_recall,
):
    """Require adequate overall and per-class webcam validation performance."""
    class_recalls = [float(report[label]["recall"]) for label in label_names]
    return (
        accuracy >= minimum_accuracy
        and all(recall >= minimum_class_recall for recall in class_recalls)
    ), class_recalls


def fer_candidate_is_acceptable(
    candidate_accuracy,
    baseline_accuracy,
    candidate_report,
    baseline_report,
    label_names,
    max_accuracy_degradation,
    min_macro_f1_improvement,
    min_class_recall,
):
    candidate_macro_f1 = float(candidate_report["macro avg"]["f1-score"])
    baseline_macro_f1 = (
        float(baseline_report["macro avg"]["f1-score"])
        if baseline_report is not None
        else None
    )
    accuracy_preserved = (
        baseline_accuracy is None
        or candidate_accuracy >= baseline_accuracy - max_accuracy_degradation
    )
    macro_f1_improved = (
        baseline_macro_f1 is None
        or candidate_macro_f1 >= baseline_macro_f1 + min_macro_f1_improvement
    )
    class_recalls = {
        label: float(candidate_report[label]["recall"])
        for label in label_names
    }
    minimum_recall_passed = all(
        recall >= min_class_recall for recall in class_recalls.values()
    )
    return (
        accuracy_preserved and macro_f1_improved and minimum_recall_passed,
        {
            "baseline_macro_f1": baseline_macro_f1,
            "candidate_macro_f1": candidate_macro_f1,
            "accuracy_preserved": accuracy_preserved,
            "macro_f1_improved": macro_f1_improved,
            "minimum_recall_passed": minimum_recall_passed,
            "class_recalls": class_recalls,
        },
    )


def train(args):
    config = load_config()
    emotion_logit_adjustments = config.get("inference", {}).get(
        "emotion_logit_adjustments", {}
    )
    paths = config.get("paths", {})
    dataset_path = resolve_path(
        args.dataset or paths.get("fer_csv", "data/fer2013/fer2013.csv")
    )
    model_path = resolve_path(
        args.output or paths.get("emotion_model", "models/emotion_model.h5")
    )
    base_artifacts_dir = resolve_path(
        args.artifacts_dir
        or paths.get("training_artifacts_dir", "models/training")
    )
    webcam_dir = resolve_path(args.webcam_dir) if args.webcam_dir else None
    webcam_mode = webcam_dir is not None
    artifacts_dir = base_artifacts_dir / "webcam" if webcam_mode else base_artifacts_dir
    candidate_path = artifacts_dir / "emotion_model_candidate.keras"
    history_path = artifacts_dir / "training_history.csv"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and not candidate_path.is_file():
        raise FileNotFoundError(
            f"No resumable checkpoint exists at {candidate_path}. "
            "Start a new run without --resume."
        )
    if not args.resume and candidate_path.exists():
        raise FileExistsError(
            f"A candidate checkpoint already exists: {candidate_path}. "
            "Use --resume to continue it, or move it aside before starting a new run."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    from src.models.emotion_cnn import build_emotion_model
    from src.preprocessing.data_loader import EMOTION_LABELS, FERDataLoader

    images, labels, splits = FERDataLoader(dataset_path).load()
    masks = {split: splits == split for split in ("train", "validation", "test")}
    for split, mask in masks.items():
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise ValueError(
                f"The FER dataset has no '{split}' samples. "
                "Use the standard Training/PublicTest/PrivateTest CSV splits."
            )
        print(f"{split.title():10} samples: {count}")

    train_labels = labels[masks["train"]]
    label_ids = np.arange(len(EMOTION_LABELS))
    present_classes = np.unique(train_labels)
    if len(present_classes) != len(EMOTION_LABELS):
        missing = sorted(set(label_ids) - set(present_classes))
        missing_names = ", ".join(EMOTION_LABELS[index] for index in missing)
        raise ValueError(f"Training split is missing emotion classes: {missing_names}.")

    train_ds = make_dataset(
        images[masks["train"]], train_labels, args.batch_size, True, args.seed
    )
    validation_labels = labels[masks["validation"]]
    validation_ds = make_dataset(
        images[masks["validation"]],
        validation_labels,
        args.batch_size,
        False,
        args.seed,
    )
    test_labels = labels[masks["test"]]
    test_ds = make_dataset(
        images[masks["test"]], test_labels, args.batch_size, False, args.seed
    )
    webcam_data = None
    if webcam_mode:
        from src.preprocessing.webcam_dataset import WebcamEmotionDataLoader

        webcam_data = WebcamEmotionDataLoader(webcam_dir).load()
        fit_train_ds = make_dataset(
            webcam_data["train_images"],
            webcam_data["train_labels"],
            args.batch_size,
            True,
            args.seed,
        )
        fit_validation_labels = webcam_data["validation_labels"]
        fit_validation_ds = make_dataset(
            webcam_data["validation_images"],
            fit_validation_labels,
            args.batch_size,
            False,
            args.seed,
        )
        print(f"Webcam train sessions: {sorted(webcam_data['train_sessions'])}")
        print(
            "Webcam validation sessions: "
            f"{sorted(webcam_data['validation_sessions'])}"
        )
        print(f"Webcam train samples: {webcam_data['train_counts']}")
        print(
            "Webcam validation samples: "
            f"{webcam_data['validation_counts']}"
        )
    else:
        fit_train_ds = train_ds
        fit_validation_labels = validation_labels
        fit_validation_ds = validation_ds

    if args.resume:
        model = load_model(candidate_path)
        initial_epoch = read_last_epoch(history_path)
        print(f"Resuming candidate at epoch {initial_epoch + 1}: {candidate_path}")
    else:
        initial_epoch = 0
        if not args.from_scratch and model_path.is_file():
            base_model = load_model(model_path, compile=False)
            if base_model.input_shape[1:] != (48, 48, 1) or base_model.output_shape[-1] != 7:
                raise ValueError(
                    f"Existing model at {model_path} must accept 48x48 grayscale "
                    "images and return seven emotion scores. Use --from-scratch "
                    "to train a replacement architecture."
                )
            model = make_finetuning_model(base_model)
            print(f"Fine-tuning compatible model from: {model_path}")
        else:
            model = build_emotion_model(num_classes=len(EMOTION_LABELS))
            print("Training the FER CNN from random initialization.")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    baseline_accuracy = None
    baseline_report = None
    baseline_fer_validation_accuracy = None
    if model_path.is_file():
        baseline_model = load_model(model_path, compile=False)
        baseline_model.compile(
            loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        (
            _,
            baseline_accuracy,
            baseline_report,
            _,
        ) = evaluate_metrics(
            baseline_model,
            fit_validation_ds,
            fit_validation_labels,
            label_ids,
            EMOTION_LABELS,
            emotion_logit_adjustments,
        )
        validation_name = "webcam" if webcam_mode else "FER"
        print(
            f"Active model {validation_name} validation accuracy: "
            f"{baseline_accuracy:.4f}"
        )
        if webcam_mode:
            (
                _,
                baseline_fer_validation_accuracy,
                _,
                _,
            ) = evaluate_metrics(
                baseline_model,
                validation_ds,
                validation_labels,
                label_ids,
                EMOTION_LABELS,
                emotion_logit_adjustments,
            )
            print(
                "Active model FER validation accuracy: "
                f"{baseline_fer_validation_accuracy:.4f}"
            )

    starting_accuracy = float(model.evaluate(fit_validation_ds, verbose=1)[1])
    print(f"Training checkpoint validation accuracy: {starting_accuracy:.4f}")
    model.summary()

    class_weights = None
    if args.class_weighting:
        fit_labels = (
            webcam_data["train_labels"] if webcam_mode else train_labels
        )
        weights = compute_class_weight(
            class_weight="balanced", classes=label_ids, y=fit_labels
        )
        class_weights = {
            int(class_id): float(weight)
            for class_id, weight in zip(label_ids, weights)
        }

    callbacks = [
        MacroF1Checkpoint(
            fit_validation_ds,
            fit_validation_labels,
            label_ids,
            candidate_path,
            emotion_logit_adjustments,
        ),
        EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 3),
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(str(history_path), append=args.resume),
    ]

    try:
        history = model.fit(
            fit_train_ds,
            validation_data=fit_validation_ds,
            initial_epoch=initial_epoch,
            epochs=initial_epoch + args.epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. The active inference model was not changed. "
            f"Resume from the best epoch checkpoint with: "
            f"python scripts\\train_model.py --resume"
        )
        raise SystemExit(130) from None

    plot_training_history(history, artifacts_dir / "training_history.png")
    candidate_model = load_model(candidate_path, compile=False)
    candidate_model.compile(
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    (
        _,
        candidate_validation_accuracy,
        webcam_report,
        webcam_matrix,
    ) = evaluate_metrics(
        candidate_model,
        fit_validation_ds,
        fit_validation_labels,
        label_ids,
        EMOTION_LABELS,
        emotion_logit_adjustments,
    )
    candidate_fer_validation_accuracy = candidate_validation_accuracy
    if webcam_mode:
        fer_val_loss, candidate_fer_validation_accuracy, fer_val_report, fer_val_matrix = (
            evaluate_metrics(
                candidate_model,
                validation_ds,
                validation_labels,
                label_ids,
                EMOTION_LABELS,
                emotion_logit_adjustments,
            )
        )
        np.savetxt(
            artifacts_dir / "fer_validation_confusion_matrix.csv",
            fer_val_matrix,
            delimiter=",",
            fmt="%d",
            header=",".join(EMOTION_LABELS),
            comments="",
        )
        with (artifacts_dir / "fer_validation_report.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(fer_val_report, file, indent=2)

        np.savetxt(
            artifacts_dir / "webcam_validation_confusion_matrix.csv",
            webcam_matrix,
            delimiter=",",
            fmt="%d",
            header=",".join(EMOTION_LABELS),
            comments="",
        )
        with (artifacts_dir / "webcam_validation_report.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(webcam_report, file, indent=2)
    else:
        fer_val_loss = None

    test_loss, test_accuracy, report, matrix = evaluate_metrics(
        candidate_model,
        test_ds,
        test_labels,
        label_ids,
        EMOTION_LABELS,
        emotion_logit_adjustments,
    )
    np.savetxt(
        artifacts_dir / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
        header=",".join(EMOTION_LABELS),
        comments="",
    )
    with (artifacts_dir / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    improved_target_validation = baseline_accuracy is None or (
        candidate_validation_accuracy >= baseline_accuracy + args.min_improvement
    )
    candidate_macro_f1 = float(webcam_report["macro avg"]["f1-score"])
    baseline_macro_f1 = (
        float(baseline_report["macro avg"]["f1-score"])
        if baseline_report is not None
        else None
    )
    webcam_quality_passed = True
    webcam_class_recalls = None
    if webcam_mode:
        webcam_quality_passed, webcam_class_recalls = (
            webcam_candidate_is_acceptable(
                candidate_validation_accuracy,
                webcam_report,
                EMOTION_LABELS,
                args.min_webcam_accuracy,
                args.min_webcam_class_recall,
            )
        )
    preserved_fer_validation = (
        not webcam_mode
        or baseline_fer_validation_accuracy is None
        or candidate_fer_validation_accuracy
        >= baseline_fer_validation_accuracy - args.max_fer_degradation
    )
    if webcam_mode:
        promoted = (
            improved_target_validation
            and preserved_fer_validation
            and webcam_quality_passed
        )
        fer_quality_metrics = None
    else:
        fer_quality_passed, fer_quality_metrics = fer_candidate_is_acceptable(
            candidate_validation_accuracy,
            baseline_accuracy,
            webcam_report,
            baseline_report,
            EMOTION_LABELS,
            args.max_accuracy_degradation,
            args.min_macro_f1_improvement,
            args.min_class_recall,
        )
        promoted = fer_quality_passed
    if promoted:
        candidate_model.save(model_path)
        print(f"Promoted candidate to active model: {model_path}")
    else:
        reasons = []
        if not improved_target_validation:
            reasons.append(
                "it did not exceed the active model's validation accuracy "
                "by the required margin"
            )
        if not preserved_fer_validation:
            reasons.append(
                "FER-2013 validation accuracy regressed beyond the allowed limit"
            )
        if not webcam_quality_passed:
            reasons.append(
                "webcam validation did not meet the minimum overall accuracy "
                f"({args.min_webcam_accuracy:.1%}) and per-class recall "
                f"({args.min_webcam_class_recall:.1%}) requirements"
            )
        if not webcam_mode:
            if not fer_quality_metrics["accuracy_preserved"]:
                reasons.append(
                    "FER validation accuracy regressed beyond the allowed limit"
                )
            if not fer_quality_metrics["macro_f1_improved"]:
                reasons.append(
                    "FER validation macro F1 did not improve by the required margin"
                )
            if not fer_quality_metrics["minimum_recall_passed"]:
                reasons.append(
                    "one or more FER validation classes fell below the "
                    f"minimum recall of {args.min_class_recall:.1%}"
                )
        print("Candidate was not promoted: " + "; ".join(reasons) + ".")

    results = {
        "dataset": str(dataset_path),
        "active_model": str(model_path),
        "candidate_model": str(candidate_path),
        "candidate_promoted": promoted,
        "active_model_validation_accuracy_before_run": baseline_accuracy,
        "candidate_validation_accuracy": candidate_validation_accuracy,
        "active_model_validation_macro_f1_before_run": baseline_macro_f1,
        "candidate_validation_macro_f1": candidate_macro_f1,
        "fer_quality_metrics": fer_quality_metrics,
        "validation_domain": "webcam" if webcam_mode else "FER-2013",
        "active_model_fer_validation_accuracy_before_run": baseline_fer_validation_accuracy,
        "candidate_fer_validation_accuracy": candidate_fer_validation_accuracy,
        "maximum_allowed_fer_validation_degradation": args.max_fer_degradation,
        "maximum_allowed_accuracy_degradation": args.max_accuracy_degradation,
        "minimum_macro_f1_improvement": args.min_macro_f1_improvement,
        "minimum_class_recall": args.min_class_recall,
        "minimum_webcam_validation_accuracy": (
            args.min_webcam_accuracy if webcam_mode else None
        ),
        "minimum_webcam_class_recall": (
            args.min_webcam_class_recall if webcam_mode else None
        ),
        "webcam_class_recalls": webcam_class_recalls,
        "webcam_quality_gate_passed": webcam_quality_passed,
        "webcam_train_sessions": (
            sorted(webcam_data["train_sessions"]) if webcam_mode else []
        ),
        "webcam_validation_sessions": (
            sorted(webcam_data["validation_sessions"]) if webcam_mode else []
        ),
        "candidate_test_loss": test_loss,
        "candidate_test_accuracy": test_accuracy,
        "emotion_labels": list(EMOTION_LABELS),
        "emotion_logit_adjustments": emotion_logit_adjustments,
        "class_weighting_enabled": args.class_weighting,
        "selection_policy": (
            (
                "Webcam run: require target accuracy improvement, webcam quality "
                "gates, and preserved FER validation accuracy; PrivateTest is "
                "evaluation-only."
                if webcam_mode
                else "FER run: select checkpoints by validation macro F1; promote "
                "only if macro F1 improves by the required margin, overall "
                "accuracy regression is bounded, and every class meets minimum "
                "recall; PrivateTest is evaluation-only."
            )
        ),
    }
    with (artifacts_dir / "training_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    print("\nTraining complete.")
    print(
        f"Candidate {('webcam' if webcam_mode else 'FER-2013')} validation "
        f"accuracy: {candidate_validation_accuracy:.4f}; "
        f"macro F1: {candidate_macro_f1:.4f}"
    )
    if webcam_mode:
        print(
            f"Candidate FER validation accuracy: "
            f"{candidate_fer_validation_accuracy:.4f} "
            f"(loss {fer_val_loss:.4f})"
        )
    print(f"Candidate PrivateTest accuracy: {test_accuracy:.4f}")
    print(f"Evaluation artifacts: {artifacts_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune, train, and evaluate the FER-2013 emotion CNN."
    )
    parser.add_argument("--dataset", help="FER-2013 CSV path (defaults to config.yaml).")
    parser.add_argument("--output", help="Active model output path (defaults to config.yaml).")
    parser.add_argument(
        "--artifacts-dir",
        help=(
            "Directory for this run's candidate, history, and reports "
            "(defaults to paths.training_artifacts_dir)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum epochs for this run; additional epochs when resuming (default: 20).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Fine-tuning learning rate (default: 0.00001).",
    )
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.001,
        help="Minimum validation accuracy gain required to replace the active model.",
    )
    parser.add_argument(
        "--min-macro-f1-improvement",
        type=float,
        default=0.002,
        help="Minimum FER validation macro-F1 gain needed for promotion.",
    )
    parser.add_argument(
        "--max-accuracy-degradation",
        type=float,
        default=0.02,
        help=(
            "Maximum allowed FER validation accuracy loss when promoting a "
            "class-balanced model (default: 0.02)."
        ),
    )
    parser.add_argument(
        "--min-class-recall",
        type=float,
        default=0.30,
        help=(
            "Minimum validation recall required for every FER emotion class "
            "(default: 0.30)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--webcam-dir",
        help=(
            "Fine-tune on captured face crops under "
            "DIR/{train,validation}/SESSION/EMOTION/."
        ),
    )
    parser.add_argument(
        "--max-fer-degradation",
        type=float,
        default=0.02,
        help=(
            "Maximum allowed absolute FER validation accuracy decrease when "
            "promoting a webcam-tuned model (default: 0.02)."
        ),
    )
    parser.add_argument(
        "--min-webcam-accuracy",
        type=float,
        default=0.60,
        help=(
            "Minimum absolute webcam validation accuracy required before "
            "promoting a webcam-tuned model (default: 0.60)."
        ),
    )
    parser.add_argument(
        "--min-webcam-class-recall",
        type=float,
        default=0.30,
        help=(
            "Minimum validation recall required for every emotion before "
            "promoting a webcam-tuned model (default: 0.30)."
        ),
    )
    parser.add_argument(
        "--class-weighting",
        action="store_true",
        help="Use inverse-frequency class weights; disabled by default for fine-tuning.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore the active checkpoint and train the CNN from random weights.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue models/training/emotion_model_candidate.keras.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size, and --patience must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if not 0 <= args.min_improvement < 1:
        parser.error("--min-improvement must be in [0, 1).")
    if not 0 <= args.max_fer_degradation < 1:
        parser.error("--max-fer-degradation must be in [0, 1).")
    if not 0 <= args.min_macro_f1_improvement < 1:
        parser.error("--min-macro-f1-improvement must be in [0, 1).")
    if not 0 <= args.max_accuracy_degradation < 1:
        parser.error("--max-accuracy-degradation must be in [0, 1).")
    if not 0 <= args.min_class_recall <= 1:
        parser.error("--min-class-recall must be in [0, 1].")
    if not 0 <= args.min_webcam_accuracy <= 1:
        parser.error("--min-webcam-accuracy must be in [0, 1].")
    if not 0 <= args.min_webcam_class_recall <= 1:
        parser.error("--min-webcam-class-recall must be in [0, 1].")
    if args.webcam_dir and args.from_scratch:
        parser.error(
            "--webcam-dir requires the existing pretrained checkpoint; "
            "do not combine it with --from-scratch."
        )
    return args


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    train(parse_args())
