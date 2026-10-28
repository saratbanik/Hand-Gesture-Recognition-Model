
"""
Hand Gesture Recognition - Single Image Prediction

Loads the trained CNN and predicts one of the ten LEAPGestRecog
hand gesture classes from a PNG, JPG, or JPEG image.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from dataset_utils import preprocess_image


# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

MODEL_PATH = ARTIFACTS / "hand_gesture_model.keras"
CLASS_NAMES_PATH = ARTIFACTS / "class_names.json"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# --------------------------------------------------
# Load trained model and class names
# --------------------------------------------------

def load_prediction_resources():
    """Load the trained CNN and its class-name metadata."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}\n\n"
            "Train the model first using:\n"
            "python train_model.py"
        )

    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}\n\n"
            "Run python train_model.py to generate it."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list) or not class_names:
        raise ValueError(
            "class_names.json must contain a non-empty list of class names."
        )

    output_shape = model.output_shape

    if len(output_shape) != 2:
        raise ValueError(
            f"Unexpected model output shape: {output_shape}"
        )

    if output_shape[-1] != len(class_names):
        raise ValueError(
            "The number of model output classes does not match "
            "the number of names in class_names.json."
        )

    return model, class_names


# --------------------------------------------------
# Preprocess an input image
# --------------------------------------------------

def prepare_image(image_path, model):
    """
    Load an image and preprocess it using the same utility used
    by the Streamlit application.
    """

    image_path = Path(image_path).expanduser()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found:\n{image_path}"
        )

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported image format. Use PNG, JPG, or JPEG."
        )

    # Read the expected input dimensions from the trained model.
    input_shape = model.input_shape

    if len(input_shape) != 4:
        raise ValueError(
            f"Unexpected model input shape: {input_shape}"
        )

    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    if height is None or width is None:
        raise ValueError(
            "The model must have fixed image input dimensions."
        )

    if channels != 1:
        raise ValueError(
            f"Expected a grayscale model with 1 input channel, "
            f"but received {channels}."
        )

    try:
        with Image.open(image_path) as image:
            processed = preprocess_image(
                image,
                size=(int(height), int(width)),
            )

    except UnidentifiedImageError as exc:
        raise ValueError(
            f"The file is not a valid image: {image_path.name}"
        ) from exc

    except OSError as exc:
        raise ValueError(
            f"Could not read image: {image_path}\n{exc}"
        ) from exc

    # Add the batch dimension: (height, width, 1)
    # becomes (1, height, width, 1).
    return processed[np.newaxis, ...]


# --------------------------------------------------
# Predict the gesture
# --------------------------------------------------

def predict_gesture(image_path, model, class_names):
    """Return the predicted gesture and its probability estimates."""

    image_batch = prepare_image(
        image_path,
        model,
    )

    probabilities = model.predict(
        image_batch,
        verbose=0,
    )[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            "Prediction output does not match the class-name list."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_gesture = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return (
        predicted_gesture,
        confidence,
        probabilities,
    )


# --------------------------------------------------
# Display prediction results
# --------------------------------------------------

def display_results(image_path, predicted_gesture, confidence,
                    probabilities, class_names):

    print("\n" + "=" * 52)
    print("       HAND GESTURE RECOGNITION RESULT")
    print("=" * 52)

    print(f"Image          : {Path(image_path).name}")
    print(f"Predicted class: {predicted_gesture}")
    print(f"Confidence     : {confidence:.2%}")

    print("\nTop predictions:")

    ranked_indices = np.argsort(probabilities)[::-1]
    top_count = min(3, len(class_names))

    for rank, index in enumerate(ranked_indices[:top_count], start=1):
        gesture = class_names[int(index)]
        score = float(probabilities[index])

        print(f"{rank}. {gesture:<18} {score:.2%}")

    print("=" * 52)

    print(
        "\nNote: Confidence is a model estimate, not a guarantee. "
        "Results may vary with image quality and sensor differences."
    )


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify a hand gesture from an image using "
            "the trained LEAPGestRecog CNN."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to a PNG, JPG, or JPEG image.",
    )

    args = parser.parse_args()

    try:
        print("Loading trained gesture recognition model...")

        model, class_names = load_prediction_resources()

        print("Loading image and preparing input...")

        predicted_gesture, confidence, probabilities = predict_gesture(
            args.image,
            model,
            class_names,
        )

        display_results(
            args.image,
            predicted_gesture,
            confidence,
            probabilities,
            class_names,
        )

    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    except Exception as exc:
        print(
            f"\nPrediction failed: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()