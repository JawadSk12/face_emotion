import unittest
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model, load_model

from scripts.train_model import (
    fer_candidate_is_acceptable,
    make_finetuning_model,
    webcam_candidate_is_acceptable,
)
from src.models.emotion_cnn import (
    apply_class_logit_adjustments,
    build_emotion_model,
)
from src.preprocessing.data_loader import EMOTION_LABELS


class EmotionCNNTests(unittest.TestCase):
    def test_class_logit_adjustment_reduces_neutral_overprediction(self):
        probabilities = np.array(
            [[0.19, 0.02, 0.16, 0.15, 0.15, 0.13, 0.20]],
            dtype=np.float32,
        )

        adjusted = apply_class_logit_adjustments(
            probabilities, EMOTION_LABELS, {"Neutral": -0.75}
        )

        self.assertEqual(int(np.argmax(probabilities[0])), 6)
        self.assertEqual(int(np.argmax(adjusted[0])), 0)
        np.testing.assert_allclose(adjusted.sum(axis=1), 1.0, atol=1e-6)

    def test_class_logit_adjustment_rejects_unknown_labels(self):
        with self.assertRaisesRegex(ValueError, "unknown classes"):
            apply_class_logit_adjustments(
                np.ones((1, 7)), EMOTION_LABELS, {"Calm": -0.5}
            )

    def test_fer_candidate_requires_macro_f1_gain_accuracy_and_all_class_recall(self):
        baseline = {
            "macro avg": {"f1-score": 0.80},
            **{label: {"recall": 0.50} for label in EMOTION_LABELS},
        }
        candidate = {
            "macro avg": {"f1-score": 0.82},
            **{label: {"recall": 0.35} for label in EMOTION_LABELS},
        }

        accepted, metrics = fer_candidate_is_acceptable(
            0.80,
            0.81,
            candidate,
            baseline,
            EMOTION_LABELS,
            0.02,
            0.01,
            0.30,
        )

        self.assertTrue(accepted)
        self.assertEqual(metrics["candidate_macro_f1"], 0.82)

        candidate["Fear"]["recall"] = 0.10
        accepted, _ = fer_candidate_is_acceptable(
            0.80,
            0.81,
            candidate,
            baseline,
            EMOTION_LABELS,
            0.02,
            0.01,
            0.30,
        )
        self.assertFalse(accepted)

    def test_webcam_candidate_requires_overall_and_every_class_recall(self):
        report = {
            label: {"recall": 0.5}
            for label in EMOTION_LABELS
        }

        accepted, recalls = webcam_candidate_is_acceptable(
            0.70, report, EMOTION_LABELS, 0.60, 0.30
        )

        self.assertTrue(accepted)
        self.assertEqual(recalls, [0.5] * len(EMOTION_LABELS))

        report["Fear"]["recall"] = 0.0
        accepted, _ = webcam_candidate_is_acceptable(
            0.95, report, EMOTION_LABELS, 0.60, 0.30
        )
        self.assertFalse(accepted)

    def test_webcam_candidate_rejected_when_accuracy_below_minimum(self):
        report = {
            label: {"recall": 0.8}
            for label in EMOTION_LABELS
        }

        accepted, _ = webcam_candidate_is_acceptable(
            0.3714, report, EMOTION_LABELS, 0.60, 0.30
        )

        self.assertFalse(accepted)

    def test_cnn_returns_seven_class_probabilities(self):
        model = build_emotion_model()
        predictions = model(
            np.zeros((2, 48, 48, 1), dtype=np.float32), training=False
        ).numpy()

        self.assertEqual(predictions.shape, (2, 7))
        np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-5)

    def test_finetuning_wrapper_preserves_inference_predictions(self):
        inputs = Input(shape=(48, 48, 1))
        outputs = Dense(7, activation="softmax")(
            Flatten()(inputs)
        )
        base_model = Model(inputs, outputs)
        wrapped_model = make_finetuning_model(base_model)
        sample = tf.random.uniform((2, 48, 48, 1), seed=3)

        expected = base_model(sample, training=False).numpy()
        actual = wrapped_model(sample, training=False).numpy()

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_candidate_checkpoint_can_be_saved_and_resumed(self):
        inputs = Input(shape=(48, 48, 1))
        outputs = Dense(7, activation="softmax")(Flatten()(inputs))
        model = make_finetuning_model(Model(inputs, outputs))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        images = np.zeros((2, 48, 48, 1), dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "candidate.keras"
            model.fit(
                images,
                labels,
                validation_data=(images, labels),
                epochs=1,
                callbacks=[
                    ModelCheckpoint(
                        checkpoint_path,
                        monitor="val_accuracy",
                        mode="max",
                        save_best_only=True,
                        initial_value_threshold=-1.0,
                    )
                ],
                verbose=0,
            )
            resumed = load_model(checkpoint_path)
            predictions = resumed(images, training=False).numpy()

        self.assertEqual(predictions.shape, (2, 7))
        np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
