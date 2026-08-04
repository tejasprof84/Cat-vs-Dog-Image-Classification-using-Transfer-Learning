import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="wide")

MODEL_FILES = {
    "MobileNetV2": "best_mobilenet_model.h5",
    "EfficientNetB0": "best_efficientnet_model.h5",
    "ResNet50": "best_resnet50_model.h5",
}

MODEL_INFO = {
    "MobileNetV2": {
        "icon": "⚡",
        "tagline": "Lightweight & Fast",
        "description": (
            "MobileNetV2 uses depthwise separable convolutions to extract image features very efficiently. "
            "It was designed for mobile and edge devices where speed matters. The model freezes ImageNet "
            "weights and adds a small classification head — reaching ~96.3% validation accuracy here."
        ),
        "val_acc": "96.3%",
    },
    "EfficientNetB0": {
        "icon": "🎯",
        "tagline": "Best Accuracy",
        "description": (
            "EfficientNetB0 scales the network depth, width, and image resolution together using a compound "
            "coefficient. This balanced scaling lets it capture fine-grained visual details with fewer "
            "parameters than bigger networks — achieving the highest accuracy of ~97.4% in this project."
        ),
        "val_acc": "97.4%",
    },
    "ResNet50": {
        "icon": "🏗️",
        "tagline": "Deep & Robust",
        "description": (
            "ResNet50 is a 50-layer network that solves the vanishing gradient problem with skip connections "
            "(residual blocks). These shortcut paths allow gradients to flow through very deep layers, making "
            "training stable and robust — reaching ~96.6% validation accuracy on the cat vs dog dataset."
        ),
        "val_acc": "96.6%",
    },
}

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d0d1a; }
.model-card {
    background: #13132b;
    border: 1px solid #2a2a4a;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.model-header { font-size: 1.05rem; font-weight: 700; color: #e0e0ff; margin-bottom: 3px; }
.model-tagline { font-size: 0.8rem; color: #8888bb; margin-bottom: 10px; }
.model-desc { font-size: 0.84rem; color: #bbbbcc; line-height: 1.55; margin-bottom: 12px; }
.result-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.95rem;
}
.badge-dog { background: #0d2540; color: #4fc3f7; border: 1px solid #1a4a7a; }
.badge-cat { background: #2d0d2d; color: #f48fb1; border: 1px solid #6a1a5a; }
.val-acc { font-size: 0.76rem; color: #666688; margin-top: 8px; }
.summary-box {
    background: #1a1040;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 16px;
    border-left: 4px solid #7c4dff;
    color: #ccccff;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🐾 Cat vs Dog Image Classifier")
st.markdown(
    "Upload a **cat or dog** image, then click **Predict** to classify it through "
    "three independent deep-learning models and compare their results side by side."
)
st.divider()


@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for name, weights_file in MODEL_FILES.items():
        model = _build_model(name)
        if os.path.exists(weights_file):
            model.load_weights(weights_file)
        else:
            st.warning(f"Weights file not found for {name}: {weights_file}")
        models[name] = model
    return models


def _build_model(model_name):
    input_shape = (224, 224, 3)
    inputs = Input(shape=input_shape, name="image_input")
    x = Lambda(lambda image: tf.cast(image, tf.float32))(inputs)
    if model_name == "MobileNetV2":
        x = Lambda(tf.keras.applications.mobilenet_v2.preprocess_input)(x)
        base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    elif model_name == "EfficientNetB0":
        x = Lambda(tf.keras.applications.efficientnet.preprocess_input)(x)
        base = EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
    else:
        x = Lambda(tf.keras.applications.resnet50.preprocess_input)(x)
        base = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name=f"{model_name}_Classifier")


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image.resize((224, 224)), dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def run_predictions(image_array):
    models = load_models()
    results = {}
    for name, model in models.items():
        prob = float(model.predict(image_array, verbose=0)[0][0])
        results[name] = {"probability": prob, "label": "Dog" if prob >= 0.5 else "Cat"}
    return results


uploaded_file = st.file_uploader("📁 Choose an image (JPG / PNG / WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("#### 🖼️ Uploaded Image")
        st.image(uploaded_file, width=260)
        st.caption(f"`{uploaded_file.name}`  ·  {uploaded_file.size / 1024:.1f} KB")
        st.markdown("")
        predict_clicked = st.button("🔍 Predict Across All Models", type="primary", use_container_width=True)

    with right:
        st.markdown("#### 📊 Prediction Results")

        if predict_clicked:
            with st.spinner("Running image through all three models…"):
                image_array = preprocess_image(uploaded_file)
                results = run_predictions(image_array)

            labels = [r["label"] for r in results.values()]
            majority = "Dog 🐶" if labels.count("Dog") >= 2 else "Cat 🐱"
            agree = "All three models agree" if len(set(labels)) == 1 else "Models have a split vote"
            st.markdown(
                f"<div class='summary-box'>🗳️ <strong>{agree}.</strong> "
                f"Majority prediction: <strong>{majority}</strong></div>",
                unsafe_allow_html=True,
            )

            for model_name, result in results.items():
                info = MODEL_INFO[model_name]
                prob = result["probability"]
                label = result["label"]
                confidence = prob if label == "Dog" else 1 - prob
                badge = "badge-dog" if label == "Dog" else "badge-cat"
                emoji = "🐶" if label == "Dog" else "🐱"
                st.markdown(f"""
<div class="model-card">
  <div class="model-header">{info['icon']} {model_name}
    <span style="font-weight:400;font-size:0.82rem;color:#777799;"> — {info['tagline']}</span>
  </div>
  <div class="model-desc">{info['description']}</div>
  <span class="result-badge {badge}">{emoji} {label}</span>
  &nbsp;&nbsp;<span style="font-size:0.88rem;color:#aaa;">Confidence:
    <strong style="color:#eee;">{confidence:.1%}</strong></span>
""", unsafe_allow_html=True)
                st.progress(float(prob), text=f"🐶 Dog {prob:.1%}  ←  →  Cat {1-prob:.1%} 🐱")
                st.markdown(f"<div class='val-acc'>📈 Validation accuracy on training set: {info['val_acc']}</div></div>",
                            unsafe_allow_html=True)

        else:
            st.info("👈 Upload an image and click **Predict** to run all three models.")
            for model_name, info in MODEL_INFO.items():
                st.markdown(f"""
<div class="model-card">
  <div class="model-header">{info['icon']} {model_name}
    <span style="font-weight:400;font-size:0.82rem;color:#777799;"> — {info['tagline']}</span>
  </div>
  <div class="model-desc">{info['description']}</div>
  <div class='val-acc'>📈 Validation accuracy: {info['val_acc']}</div>
</div>""", unsafe_allow_html=True)

else:
    st.info("📂 Upload an image above to get started.")
    st.markdown("#### 🧠 About the Three Models")
    cols = st.columns(3)
    for col, (model_name, info) in zip(cols, MODEL_INFO.items()):
        with col:
            st.markdown(f"""
<div class="model-card">
  <div class="model-header">{info['icon']} {model_name}</div>
  <div class="model-tagline">{info['tagline']}</div>
  <div class="model-desc">{info['description']}</div>
  <div class='val-acc'>📈 Val accuracy: {info['val_acc']}</div>
</div>""", unsafe_allow_html=True)
