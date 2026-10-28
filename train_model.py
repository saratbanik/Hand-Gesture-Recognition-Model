"""
Hand Gesture Recognition using a Convolutional Neural Network (CNN).

Dataset:
    LEAPGestRecog

Features:
    - Automatic dataset-root detection
    - Subject-based train/validation/test split
    - Image preprocessing and normalization
    - CNN model training
    - Early stopping and learning-rate reduction
    - Test accuracy, macro F1 and classification report
    - Confusion matrix and training-history plots
    - Saved Keras model and class-name metadata
"""

from pathlib import Path
import argparse
import json
import os

# Reduce TensorFlow informational messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from dataset_utils import (
    CLASS_NAMES,
    collect_records,
    find_dataset_root,
    split_by_subject,
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "hand_gesture_model.keras"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_validation_model.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
HISTORY_PLOT_PATH = ARTIFACTS_DIR / "training_history.png"


# --------------------------------------------------
# Create TensorFlow datasets
# --------------------------------------------------

def make_dataset(
    records,
    image_size,
    batch_size,
    training=False,
    seed=42,
):
    """
    Build a TensorFlow dataset that loads images from disk in batches.

    Each record contains:
        (image_path, class_id, subject_id)

    Images are converted to grayscale, resized, and normalized.
    """

    image_paths = np.asarray(
        [record[0] for record in records],
        dtype=str,
    )

    labels = np.asarray(
        [record[1] for record in records],
        dtype=np.int32,
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(records), 4000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load_and_preprocess(image_path, label):
        image_bytes = tf.io.read_file(image_path)

        # LEAPGestRecog images are PNG files.
        image = tf.io.decode_png(
            image_bytes,
            channels=1,
        )

        image = tf.image.resize(
            image,
            [image_size, image_size],
        )

        image = tf.cast(image, tf.float32) / 255.0

        image = tf.ensure_shape(
            image,
            [image_size, image_size, 1],
        )

        # Apply mild brightness augmentation only during training.
        if training:
            image = tf.image.random_brightness(
                image,
                max_delta=0.04,
            )
            image = tf.clip_by_value(
                image,
                0.0,
                1.0,
            )

        return image, label

    dataset = dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# --------------------------------------------------
# Build CNN architecture
# --------------------------------------------------

def build_model(image_size, number_of_classes):
    """
    Create a CNN for ten-class hand gesture classification.
    """

    inputs = tf.keras.Input(
        shape=(image_size, image_size, 1),
        name="hand_image",
    )

    # Block 1
    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 2
    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 3
    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 4
    x = tf.keras.layers.Conv2D(
        192,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Classification head
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(0.35)(x)

    outputs = tf.keras.layers.Dense(
        number_of_classes,
        activation="softmax",
        name="gesture_probabilities",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="hand_gesture_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# --------------------------------------------------
# Plot training history
# --------------------------------------------------

def save_training_history(history, output_path):
    """Save training and validation accuracy/loss plots."""

    history_data = history.history

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
    )

    # Accuracy
    axes[0].plot(
        history_data["accuracy"],
        label="Training",
    )

    axes[0].plot(
        history_data["val_accuracy"],
        label="Validation",
    )

    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # Loss
    axes[1].plot(
        history_data["loss"],
        label="Training",
    )

    axes[1].plot(
        history_data["val_loss"],
        label="Validation",
    )

    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# --------------------------------------------------
# Plot confusion matrix
# --------------------------------------------------

def save_confusion_matrix(y_true, y_pred, output_path):
    """Save the confusion matrix for the held-out test subject."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    figure, axis = plt.subplots(
        figsize=(11, 9),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False,
    )

    axis.set_title(
        "Hand Gesture Recognition — Test Set"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    return matrix


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a CNN for LEAPGestRecog hand gesture recognition."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images per training batch.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="Square image input dimension in pixels.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    if args.image_size < 32:
        parser.error("--image-size must be at least 32.")

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tf.keras.utils.set_random_seed(42)

    print("=" * 60)
    print("HAND GESTURE RECOGNITION - CNN TRAINING")
    print("=" * 60)

    # Step 1: Find the actual dataset root.
    print("\n[1/7] Discovering dataset...")

    dataset_root = find_dataset_root(
        DATA_DIR
    )

    # Step 2: Load image paths and class labels.
    print("\n[2/7] Collecting image paths and labels...")

    records = collect_records(
        dataset_root
    )

    print(f"Total images discovered: {len(records):,}")

    if len(CLASS_NAMES) != 10:
        raise ValueError(
            f"Expected 10 gesture classes, found {len(CLASS_NAMES)}."
        )

    # Step 3: Create a subject-based split.
    print("\n[3/7] Splitting data by subject...")

    train_records, val_records, test_records, split_info = (
        split_by_subject(records)
    )

    def summarize_split(name, subset):
        counts = np.bincount(
            [record[1] for record in subset],
            minlength=len(CLASS_NAMES),
        )

        print(f"\n{name}: {len(subset):,} images")

        for class_name, count in zip(CLASS_NAMES, counts):
            print(
                f"  {class_name:<15} {int(count):>6,}"
            )

        if np.any(counts == 0):
            raise ValueError(
                f"{name} is missing one or more gesture classes. "
                f"Class counts: {counts.tolist()}"
            )

    summarize_split(
        "Training set",
        train_records,
    )

    summarize_split(
        "Validation set",
        val_records,
    )

    summarize_split(
        "Test set",
        test_records,
    )

    print("\nSubject split:")
    print(json.dumps(split_info, indent=2))

    # Build disk-backed, batched data pipelines.
    train_dataset = make_dataset(
        train_records,
        args.image_size,
        args.batch_size,
        training=True,
    )

    validation_dataset = make_dataset(
        val_records,
        args.image_size,
        args.batch_size,
    )

    test_dataset = make_dataset(
        test_records,
        args.image_size,
        args.batch_size,
    )

    # Step 4: Build and train the model.
    print("\n[4/7] Building CNN...")

    model = build_model(
        args.image_size,
        len(CLASS_NAMES),
    )

    model.summary()

    # Remove an earlier checkpoint so it cannot be mistaken for
    # the best checkpoint from this training run.
    if BEST_MODEL_PATH.exists():
        BEST_MODEL_PATH.unlink()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n[5/7] Training the CNN...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Load the best checkpoint from this run, if available.
    if BEST_MODEL_PATH.exists():
        model = tf.keras.models.load_model(
            BEST_MODEL_PATH
        )

    save_training_history(
        history,
        HISTORY_PLOT_PATH,
    )

    # Step 5: Final evaluation on the held-out subject.
    print("\n[6/7] Evaluating the held-out test subject...")

    test_results = model.evaluate(
        test_dataset,
        verbose=1,
        return_dict=True,
    )

    # Keep record order consistent with the unshuffled test dataset.
    y_true = np.asarray(
        [record[1] for record in test_records],
        dtype=np.int32,
    )

    probabilities = model.predict(
        test_dataset,
        verbose=1,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int32)

    test_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = save_confusion_matrix(
        y_true,
        y_pred,
        CONFUSION_MATRIX_PATH,
    )

    # Step 6: Save all model artifacts.
    print("\n[7/7] Saving model and evaluation results...")

    model.save(
        MODEL_PATH
    )

    CLASS_NAMES_PATH.write_text(
        json.dumps(CLASS_NAMES, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "dataset_root": str(dataset_root),
        "total_images": int(len(records)),
        "train_images": int(len(train_records)),
        "validation_images": int(len(val_records)),
        "test_images": int(len(test_records)),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(
            len(history.history.get("loss", []))
        ),
        "subject_split": split_info,
        "test_loss": float(test_results["loss"]),
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(macro_f1),
        "class_names": CLASS_NAMES,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Test accuracy : {test_accuracy:.4f}")
    print(f"Macro F1 score: {macro_f1:.4f}")
    print(f"Test loss     : {float(test_results['loss']):.4f}")

    print("\nSaved files:")

    for path in [
        MODEL_PATH,
        CLASS_NAMES_PATH,
        METRICS_PATH,
        CONFUSION_MATRIX_PATH,
        HISTORY_PLOT_PATH,
    ]:
        print(f"  {path}")

    print("\nYou can now start the app using:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()