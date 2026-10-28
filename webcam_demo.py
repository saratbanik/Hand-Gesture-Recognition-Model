
import argparse
import json
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from dataset_utils import preprocess_image


# Project paths
ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "hand_gesture_model.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"


def load_resources():
    """Load the trained model and gesture class names."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run 'python train_model.py' before starting the webcam demo."
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}\n"
            "Run 'python train_model.py' first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    # Support either a list or a dictionary of class names.
    if isinstance(class_names, dict):
        class_names = [
            class_names[key]
            for key in sorted(class_names, key=lambda value: int(value))
        ]

    if len(class_names) != model.output_shape[-1]:
        raise ValueError(
            "The number of class names does not match the model output."
        )

    return model, class_names


def format_label(class_name):
    """Convert a dataset label such as 01_palm into Palm."""
    label = str(class_name)
    label = label.split("_", 1)[-1] if "_" in label else label
    return label.replace("_", " ").title()


def predict_frame(model, frame):
    """Prepare a webcam frame and return the predicted probabilities."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    input_height = model.input_shape[1] or 96
    input_width = model.input_shape[2] or 96

    processed_image = preprocess_image(
        pil_image,
        size=(input_height, input_width),
    )

    input_batch = np.expand_dims(processed_image, axis=0)
    probabilities = model.predict(input_batch, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    return predicted_index, confidence


def draw_overlay(frame, label, confidence, threshold, fps):
    """Draw the prediction information over the webcam frame."""
    height, width = frame.shape[:2]

    # Top information panel
    cv2.rectangle(frame, (0, 0), (width, 100), (35, 35, 35), -1)

    if label is None:
        display_text = "Waiting for prediction..."
    elif confidence < threshold:
        display_text = "Gesture uncertain"
    else:
        display_text = f"Gesture: {label}"

    cv2.putText(
        frame,
        display_text,
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if confidence is not None:
        confidence_text = f"Confidence: {confidence:.1%}"
        cv2.putText(
            frame,
            confidence_text,
            (15, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (max(10, width - 130), 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (15, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Real-time hand gesture recognition using a webcam."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.55,
        help="Minimum confidence required to display a gesture (default: 0.55).",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=3,
        help="Run predictions every N frames (default: 3).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1.")

    if args.every < 1:
        parser.error("--every must be at least 1.")

    print("Loading trained model...")
    model, class_names = load_resources()
    print("Model loaded successfully.")
    print("Opening webcam...")

    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open webcam {args.camera}. "
            "Check your camera connection or try --camera 1."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    recent_predictions = deque(maxlen=7)
    frame_count = 0
    predicted_index = None
    confidence = None
    previous_time = cv2.getTickCount()
    fps = 0.0

    print("Webcam running. Press Q in the video window to exit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a frame from the webcam.")
                break

            frame_count += 1

            # Predict periodically to reduce processing load.
            if frame_count % args.every == 0 or predicted_index is None:
                predicted_index, confidence = predict_frame(model, frame)
                recent_predictions.append(predicted_index)

            # Smooth predictions to reduce rapid label changes.
            if recent_predictions:
                predicted_index = Counter(recent_predictions).most_common(1)[0][0]

                # The confidence shown is from the most recent prediction.
                label = format_label(class_names[predicted_index])
            else:
                label = None

            # Calculate approximate FPS.
            current_time = cv2.getTickCount()
            elapsed = (current_time - previous_time) / cv2.getTickFrequency()

            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = current_fps if fps == 0 else (0.8 * fps + 0.2 * current_fps)

            previous_time = current_time

            frame = draw_overlay(
                frame,
                label,
                confidence,
                args.confidence,
                fps,
            )

            cv2.imshow("Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nWebcam demo stopped.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()