
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from dataset_utils import preprocess_image


# --------------------------------------------------
# Project paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "hand_gesture_model.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Hand Gesture Studio",
    page_icon="✋",
    layout="wide",
)


# --------------------------------------------------
# Custom styling and fixed HTML title
# IMPORTANT: unsafe_allow_html=True renders HTML tags.
# Do not put this HTML inside Markdown code fences.
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #0b1326 0%,
            #14213d 55%,
            #0b3038 100%
        );
        color: #f1f5f9;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #101b30;
    }

    .hero {
        padding: 28px 32px;
        margin: 5px 0 28px 0;
        border-radius: 22px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: linear-gradient(
            115deg,
            #1c2b57 0%,
            #123d49 100%
        );
    }

    .eyebrow {
        color: #5ef0df;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero h1 {
        color: white;
        font-size: 2.7rem;
        font-weight: 800;
        line-height: 1.2;
        margin: 0 0 12px 0;
    }

    .hero p {
        color: #d9e5f5;
        font-size: 1.02rem;
        line-height: 1.7;
        max-width: 780px;
        margin: 0;
    }

    .result-card {
        background: rgba(17, 35, 54, 0.85);
        border: 1px solid rgba(94, 240, 223, 0.35);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
    }

    .result-label {
        color: #5ef0df;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .result-value {
        color: white;
        font-size: 1.8rem;
        font-weight: 800;
    }

    div[data-testid="stMetric"] {
        background: rgba(17, 35, 54, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 12px;
    }
    </style>

    <div class="hero">
        <div class="eyebrow">
            Computer Vision · Deep Learning
        </div>
        <h1>Hand Gesture Studio</h1>
        <p>
            Discover hand gestures using a trained Convolutional
            Neural Network. Upload an image to identify one of the
            ten gesture classes from the LEAPGestRecog dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load trained model and labels
# --------------------------------------------------
@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "class_names.json not found in the artifacts folder."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if isinstance(class_names, dict):
        try:
            class_names = [
                class_names[key]
                for key in sorted(class_names, key=lambda item: int(item))
            ]
        except (TypeError, ValueError):
            class_names = [
                class_names[key] for key in sorted(class_names)
            ]

    class_names = [str(name) for name in class_names]

    if len(class_names) != int(model.output_shape[-1]):
        raise ValueError(
            "The number of class names does not match the model outputs."
        )

    return model, class_names


def format_label(label):
    """Convert labels such as 01_palm into Palm."""
    label = str(label)

    if "_" in label:
        label = label.split("_", 1)[1]

    return label.replace("_", " ").title()


def predict_gesture(image, model):
    """Preprocess an image and predict its gesture."""
    input_shape = model.input_shape

    height = int(input_shape[1] or 96)
    width = int(input_shape[2] or 96)

    processed = preprocess_image(
        image,
        size=(height, width),
    )

    processed = np.asarray(processed, dtype=np.float32)

    if processed.ndim == 2:
        processed = processed[..., np.newaxis]

    expected_channels = input_shape[-1]

    if (
        expected_channels is not None
        and processed.shape[-1] != int(expected_channels)
    ):
        raise ValueError(
            "Image preprocessing does not match the trained model input."
        )

    batch = np.expand_dims(processed, axis=0)

    predictions = model.predict(batch, verbose=0)[0]

    return predictions


# --------------------------------------------------
# Load model
# --------------------------------------------------
try:
    model, class_names = load_model_and_classes()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.info(
        "Check the artifacts folder and make sure training completed successfully."
    )
    st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("Model Information")
    st.write("**Architecture:** CNN")
    st.write("**Gesture classes:** 10")
    st.write("**Dataset:** LEAPGestRecog")

    st.divider()

    st.markdown("**Supported gestures**")

    st.write(
        "Palm, L, Fist, Moving Fist, Thumb, Index, "
        "OK, Moving Palm, C, Down"
    )


# --------------------------------------------------
# Image upload and prediction interface
# --------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Upload a Gesture Image")

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
    )

    image = None

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded hand gesture",
                use_container_width=True,
            )

        except Exception as error:
            st.error(f"Unable to open the image: {error}")


with right:
    st.subheader("Prediction Results")

    if image is None:
        st.info(
            "Upload a hand gesture image to see the prediction."
        )

    else:
        try:
            with st.spinner("Analyzing the image..."):
                probabilities = predict_gesture(image, model)

            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            predicted_label = format_label(class_names[best_index])

            # Display prediction using rendered HTML.
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted Gesture
                    </div>
                    <div class="result-value">
                        {predicted_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.metric(
                "Prediction Score",
                f"{confidence:.1%}",
            )

            st.subheader("Top Predictions")

            top_indices = np.argsort(probabilities)[::-1][:3]

            for index in top_indices:
                label = format_label(class_names[int(index)])
                score = float(probabilities[int(index)])

                st.write(f"**{label}** — {score:.1%}")

                st.progress(
                    min(max(score, 0.0), 1.0)
                )

            st.caption(
                "Prediction scores are not a guarantee of correctness."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# Training evaluation
# --------------------------------------------------
st.divider()
st.subheader("Training Evaluation")

if METRICS_PATH.exists():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        scalar_metrics = [
            (key, value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        ]

        if scalar_metrics:
            columns = st.columns(min(4, len(scalar_metrics)))

            for index, (key, value) in enumerate(scalar_metrics):
                columns[index % len(columns)].metric(
                    key.replace("_", " ").title(),
                    f"{value:.4f}"
                    if isinstance(value, float)
                    else str(value),
                )

        with st.expander("View complete evaluation metrics"):
            st.json(metrics)

    except (OSError, json.JSONDecodeError):
        st.warning("The metrics.json file could not be read.")

else:
    st.info("Training metrics are not available yet.")


st.caption(
    "Hand Gesture Studio | CNN-based hand gesture recognition"
)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <div class="footer" style="text-align: center;">
        Developed by <a href="https://www.linkedin.com/in/saratbanik" target="_blank" rel="noopener noreferrer">SARAT BANIK</a>
    </div>
    """,
    unsafe_allow_html=True,
)