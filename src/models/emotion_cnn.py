"""CNN architecture for 48x48 grayscale FER-2013 images."""

from collections.abc import Mapping, Sequence

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)


def apply_class_logit_adjustments(
    probabilities,
    class_labels: Sequence[str],
    adjustments: Mapping[str, float],
):
    """Apply validated per-class logit offsets and return normalized scores."""
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(class_labels):
        raise ValueError(
            "Probabilities must be a 2D array with one column per class label."
        )

    unknown_labels = set(adjustments) - set(class_labels)
    if unknown_labels:
        raise ValueError(
            "Logit adjustments contain unknown classes: "
            + ", ".join(sorted(unknown_labels))
        )

    logits = np.log(np.maximum(scores, np.finfo(np.float64).tiny))
    for label, offset in adjustments.items():
        class_index = class_labels.index(label)
        logits[:, class_index] += float(offset)
    logits -= np.max(logits, axis=1, keepdims=True)
    adjusted_scores = np.exp(logits)
    adjusted_scores /= np.sum(adjusted_scores, axis=1, keepdims=True)
    return adjusted_scores.astype(np.float32)


def build_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """Build the established FER CNN with augmentation active only during training."""
    inputs = Input(shape=input_shape, name="face_image")
    x = RandomFlip("horizontal", name="random_flip")(inputs)
    x = RandomRotation(0.04, name="random_rotation")(x)
    x = RandomTranslation(0.04, 0.04, name="random_translation")(x)
    x = RandomZoom(0.08, name="random_zoom")(x)

    for block, filters, conv_count in (
        (1, 64, 2),
        (2, 128, 2),
        (3, 256, 1),
    ):
        for conv_index in range(1, conv_count + 1):
            x = Conv2D(
                filters,
                kernel_size=3,
                activation="relu",
                padding="same",
                name=f"block{block}_conv{conv_index}",
            )(x)
            x = BatchNormalization(name=f"block{block}_bn{conv_index}")(x)
        x = MaxPooling2D(pool_size=2, name=f"block{block}_pool")(x)
        x = Dropout(0.15 + block * 0.05, name=f"block{block}_dropout")(x)

    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="relu", name="classifier_dense")(x)
    x = BatchNormalization(name="classifier_bn")(x)
    x = Dropout(0.5, name="classifier_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="emotion")(x)

    return Model(inputs=inputs, outputs=outputs, name="fer2013_emotion_cnn")
