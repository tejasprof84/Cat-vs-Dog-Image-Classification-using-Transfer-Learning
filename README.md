# 🐾 Cat vs Dog Image Classifier

## 🌐 Live Project

https://cat-vs-dog-image-classification-using-transfer-learning-3v5de8.streamlit.app/

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?logo=tensorflow) ![Streamlit](https://img.shields.io/badge/Streamlit-%E2%89%A51.35-red?logo=streamlit)

---

## 📌 Project Description

**Cat vs Dog Image Classifier** is a deep learning project that automatically identifies whether an uploaded image contains a **cat** or a **dog**. It uses the power of **transfer learning** — taking three well-known neural network architectures pretrained on ImageNet and fine-tuning them on a binary cat vs dog dataset to build highly accurate classifiers.

The project is wrapped in an interactive **Streamlit web app** where users upload any image, click **Predict**, and instantly receive results from all three models simultaneously — with predicted label, confidence score, probability bar, and a majority vote summary.

---

## 🧠 Models Used

| Model | Strength | Val Accuracy |
|---|---|---|
| ⚡ MobileNetV2 | Lightweight & Fast — ideal for real-time inference | 96.3% |
| 🎯 EfficientNetB0 | Best Accuracy — balanced scaling of depth, width & resolution | 97.4% |
| 🏗️ ResNet50 | Deep & Robust — skip connections prevent vanishing gradients | 96.6% |

All three models use:
- Frozen ImageNet pretrained base weights
- `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(1, sigmoid)` classification head
- Input size: **224 × 224 × 3**

---

## 🗂️ Project Structure

```
cat vs dog/
├── app.py                        # Streamlit UI and prediction logic
├── best_mobilenet_model.h5       # Saved weights — MobileNetV2
├── best_efficientnet_model.h5    # Saved weights — EfficientNetB0
├── best_resnet50_model.h5        # Saved weights — ResNet50
├── Cats_vs_Dogs_Classification_v1_final.ipynb  # Training notebook
├── README.md                      # Project documentation
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Deployment runtime (Python 3.11)
├── .python-version               # Local Python version pin
├── .gitignore
└── .streamlit/
    └── config.toml               # Streamlit server config
```

---

## ⚙️ Setup & Installation

### 1. Create a virtual environment (Python 3.11)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the app
```powershell
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🖥️ How to Use

1. Open the app in your browser.
2. Click **Browse files** and upload a cat or dog image (JPG / PNG / WEBP).
3. Click the **🔍 Predict Across All Models** button.
4. View results for each model:
   - 🐶 **Dog** or 🐱 **Cat** prediction badge
   - Confidence percentage
   - Probability bar (Dog ← → Cat)
   - Majority vote summary across all three models

---

## 📦 Dependencies

| Package | Version |
|---|---|
| TensorFlow | 2.16.2 (Python < 3.12), tf-nightly (Python >= 3.12) |
| Streamlit | ≥ 1.35.0 |
| NumPy | ≥ 1.23.5 |
| Pillow | ≥ 10.0.0 |

---

## 👤 Author

Built as part of the **Datamites AI Projects** series.
