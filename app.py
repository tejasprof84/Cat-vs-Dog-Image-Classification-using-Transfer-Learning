"""
Streamlit web application for Cat vs Dog image classification.

Loads three pre-trained transfer-learning models (MobileNetV2, EfficientNetB0,
ResNet50) and lets the user upload an image to get per-model predictions plus
an ensemble majority-vote result.
"""

import os
from collections import Counter

import numpy as np
import streamlit as st
from PIL import Image

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
MODEL_DIR = "models"
CLASS_NAMES = ["Cat", "Dog"]  # index 0 → Cat, index 1 → Dog

MODEL_INFO = {
    "MobileNetV2": {
        "file": "mobilenetv2.keras",
        "description": "Lightweight & fast – ideal for quick predictions",
    },
    "EfficientNetB0": {
        "file": "efficientnetb0.keras",
        "description": "Highest accuracy (97.4%) – best overall performance",
    },
    "ResNet50": {
        "file": "resnet50.keras",
        "description": "Deep & robust – uses residual connections for strong feature learning",
    },
}


# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """Load all three models once and cache them."""
    import tensorflow as tf  # import here so Streamlit's reloader is happy

    loaded = {}
    for name, info in MODEL_INFO.items():
        path = os.path.join(MODEL_DIR, info["file"])
        if os.path.exists(path):
            loaded[name] = tf.keras.models.load_model(path)
        else:
            loaded[name] = None
    return loaded


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalise an image for model inference."""
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # shape (1, 224, 224, 3)


def predict(model, img_array: np.ndarray) -> tuple[str, float]:
    """Return (class_name, confidence) for a single image array."""
    prob = float(model.predict(img_array, verbose=0)[0][0])
    class_idx = 1 if prob >= 0.5 else 0
    confidence = prob if class_idx == 1 else 1.0 - prob
    return CLASS_NAMES[class_idx], confidence


def majority_vote(predictions: list[str]) -> str:
    """Return the class with the most votes."""
    counts = Counter(predictions)
    return counts.most_common(1)[0][0]


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐾 Cat vs Dog Image Classifier")
st.markdown(
    "Upload an image of a **cat** or a **dog** and click **Predict**. "
    "All three models will analyse the image, and a majority vote will give "
    "the final answer."
)

models = load_models()
missing = [n for n, m in models.items() if m is None]
if missing:
    st.warning(
        f"⚠️ The following model files were not found in `{MODEL_DIR}/`: "
        + ", ".join(missing)
        + ". Run `python train.py` to train the models first."
    )

uploaded_file = st.file_uploader(
    "Choose an image…", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary"):
        img_array = preprocess_image(image)

        st.subheader("Model Predictions")

        available_models = {n: m for n, m in models.items() if m is not None}
        if not available_models:
            st.error("No models are available. Please train the models first.")
            st.stop()

        votes = []
        cols = st.columns(len(available_models))

        for col, (name, model) in zip(cols, available_models.items()):
            pred_class, confidence = predict(model, img_array)
            votes.append(pred_class)
            emoji = "🐱" if pred_class == "Cat" else "🐶"
            with col:
                st.metric(
                    label=name,
                    value=f"{emoji} {pred_class}",
                    delta=f"{confidence * 100:.1f}% confidence",
                )
                st.caption(MODEL_INFO[name]["description"])

        # ── Majority vote ─────────────────────────────────────────────────────
        final = majority_vote(votes)
        final_emoji = "🐱" if final == "Cat" else "🐶"

        st.divider()
        st.subheader("Final Prediction (Majority Vote)")
        st.markdown(
            f"<h2 style='text-align:center'>{final_emoji} {final}</h2>",
            unsafe_allow_html=True,
        )

        # Show vote breakdown
        vote_counts = Counter(votes)
        st.markdown(
            "**Vote breakdown:** "
            + " | ".join(
                f"{cls}: {cnt}/{len(votes)}"
                for cls, cnt in vote_counts.most_common()
            )
        )
