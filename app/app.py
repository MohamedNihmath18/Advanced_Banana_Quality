import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import requests

import keras
from keras.models import load_model

from ultralytics import YOLO
from utils import (
    load_image, classify_image, detect_bananas,
    CLASS_INFO, CLASS_NAMES
)

st.set_page_config(
    page_title="🍌 Banana Quality Detector",
    page_icon="🍌",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        padding: 10px;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-box {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #FFD700;
    }
</style>
""", unsafe_allow_html=True)

APP_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(APP_DIR)
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

MODEL_URLS = {
    "Custom CNN":     "https://github.com/MohamedNihmath18/advanced_banana_quality/releases/download/v1.0.0/custom_cnn_model.keras",
    "MobileNetV2":    "https://github.com/MohamedNihmath18/advanced_banana_quality/releases/download/v1.0.0/mobilenetv2_model.keras",
    "EfficientNetB0": "https://github.com/MohamedNihmath18/advanced_banana_quality/releases/download/v1.0.0/efficientnetb0_model.keras",
    "ResNet50":       "https://github.com/MohamedNihmath18/advanced_banana_quality/releases/download/v1.0.0/resnet50_model.keras",
    "YOLO":           "https://github.com/MohamedNihmath18/advanced_banana_quality/releases/download/v1.0.0/yolo_detect_best.pt"
}

MODEL_FILES = {
    "Custom CNN":     "custom_cnn_model.keras",
    "MobileNetV2":    "mobilenetv2_model.keras",
    "EfficientNetB0": "efficientnetb0_model.keras",
    "ResNet50":       "resnet50_model.keras",
    "YOLO":           "yolo_detect_best.pt"
}

def download_if_missing(model_name, filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path) and os.path.getsize(path) < 100000:
        os.remove(path)
    if not os.path.exists(path):
        url = MODEL_URLS.get(model_name)
        if url:
            with st.spinner(f"📥 Downloading {model_name}... Please wait."):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    with requests.get(url, headers=headers,
                                      stream=True, allow_redirects=True) as r:
                        r.raise_for_status()
                        with open(path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    if os.path.exists(path) and os.path.getsize(path) > 1000000:
                        st.sidebar.success(f"✅ {model_name} ready!")
                    else:
                        if os.path.exists(path):
                            os.remove(path)
                        st.sidebar.error(f"❌ Download failed for {model_name}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {e}")
    return path

@st.cache_resource
def load_classification_model(name):
    filename = MODEL_FILES.get(name)
    if not filename:
        return None
    path = download_if_missing(name, filename)
    if os.path.exists(path) and os.path.getsize(path) > 100000:
        try:
            return load_model(path, compile=False)
        except Exception as e:
            st.error(f"❌ Failed to load {name}: {e}")
            if os.path.exists(path):
                os.remove(path)
    return None

@st.cache_resource
def load_yolo():
    path = download_if_missing("YOLO", "yolo_detect_best.pt")
    if os.path.exists(path) and os.path.getsize(path) > 100000:
        try:
            return YOLO(path)
        except Exception as e:
            st.error(f"❌ YOLO error: {e}")
    return None

# ── Header ──
st.markdown("<div class='main-title'>🍌 Advanced Banana Quality Detector</div>",
            unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered banana ripeness classification & detection</div>",
            unsafe_allow_html=True)
st.info("📦 Dataset: Banana Ripeness Classification — Roboflow Universe (CC BY 4.0)")

# ── Sidebar ──
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    input_method = st.radio("📷 Input Method", ["Upload Image", "Camera Capture"])

    st.markdown("### 🤖 Classification Model")
    selected_model = st.selectbox(
        "Select Model",
        ["Custom CNN", "MobileNetV2", "EfficientNetB0", "ResNet50"]
    )

    model_info = {
        "Custom CNN":     {"params": "26.4M", "accuracy": "97.51%", "speed": "16.54ms"},
        "MobileNetV2":    {"params": "2.6M",  "accuracy": "97.51%", "speed": "49.01ms"},
        "EfficientNetB0": {"params": "4.4M",  "accuracy": "98.75%", "speed": "41.86ms"},
        "ResNet50":       {"params": "24.2M", "accuracy": "98.75%", "speed": "25.46ms"},
    }
    info = model_info[selected_model]
    st.markdown(f"""
    **Model Info:**
    - ✅ Accuracy: `{info["accuracy"]}`
    - ⚡ Speed: `{info["speed"]}`
    - 📦 Params: `{info["params"]}`
    """)

    st.markdown("### 🎯 Detection Settings")
    run_detection  = st.checkbox("Enable YOLO Detection", value=True)
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use **clear lighting**
    - Show banana **clearly**
    - Avoid **cluttered backgrounds**
    """)

# ── Load Models ──
with st.spinner("⏳ Loading AI models..."):
    cls_model  = load_classification_model(selected_model)
    yolo_model = load_yolo() if run_detection else None

if cls_model is None:
    st.error(f"❌ Could not load {selected_model}. Check model URLs in GitHub releases.")
    st.stop()

# ── Tabs ──
tab1, tab2, tab3 = st.tabs([
    "🔍 Detection & Classification",
    "📊 Model Comparison",
    "ℹ️ About"
])

# ── TAB 1 ──
with tab1:
    st.markdown("## 📸 Upload or Capture a Banana Image")
    image_input = None

    if input_method == "Upload Image":
        uploaded = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )
        if uploaded:
            image_input = uploaded
    else:
        camera_img = st.camera_input("📷 Take a photo")
        if camera_img:
            image_input = camera_img

    if image_input is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🖼️ Input Image")
            img_original, img_resized = load_image(image_input)
            st.image(img_original, use_column_width=True)

        with col2:
            st.markdown("### 🤖 Classification Result")
            with st.spinner(f"Running {selected_model}..."):
                result = classify_image(cls_model, img_resized, selected_model)

            cls_name = result["class"]
            cls_info = CLASS_INFO.get(cls_name, {})

            st.markdown(f"""
            <div class="result-box">
                <h2>{cls_info.get("emoji","🍌")} {cls_name.upper()}</h2>
                <p>{cls_info.get("description","")}</p>
                <h3>Confidence: {result["confidence"]:.1f}%</h3>
                <p>⚡ Inference: {result["inference_time"]:.1f} ms</p>
                <p>🤖 Model: {selected_model}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Confidence per Class:")
            for cls, conf in sorted(
                result["all_confidences"].items(),
                key=lambda x: x[1], reverse=True
            ):
                emoji = CLASS_INFO.get(cls, {}).get("emoji", "🍌")
                st.progress(int(conf), text=f"{emoji} {cls}: {conf:.1f}%")

        if run_detection and yolo_model is not None:
            st.markdown("---")
            st.markdown("### 🎯 YOLO Object Detection & Counting")
            image_input.seek(0)
            with st.spinner("Running YOLO detection..."):
                det_result = detect_bananas(yolo_model, image_input, conf_threshold)

            col3, col4 = st.columns([1, 1])
            with col3:
                st.image(det_result["annotated_image"],
                         use_column_width=True,
                         caption="YOLO Detection Result")
            with col4:
                st.markdown(f"""
                <div class="result-box">
                    <h2>🍌 Bananas Detected: {det_result["count"]}</h2>
                    <p>⚡ Detection Time: {det_result["inference_time"]:.1f} ms</p>
                </div>
                """, unsafe_allow_html=True)

                if det_result["count"] > 0:
                    det_df = pd.DataFrame({
                        "Class": det_result["detected_classes"],
                        "Confidence (%)": [
                            f"{c:.1f}%" for c in det_result["confidences"]
                        ]
                    })
                    st.dataframe(det_df, use_container_width=True)
                else:
                    st.warning("No bananas detected. Try a clearer image!")

        elif run_detection and yolo_model is None:
            st.warning("⚠️ YOLO model not available!")

    else:
        st.markdown("""
        <div style="text-align:center; padding:50px; color:#666;">
            <h2>👆 Upload or capture a banana image!</h2>
            <p>Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2 ──
with tab2:
    st.markdown("## 📊 Model Comparison Dashboard")

    comparison_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(comparison_path):
        df = pd.read_csv(comparison_path)

        st.markdown("### 🏆 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        best_acc = df.loc[df["Test Accuracy (%)"].idxmax()]
        fastest  = df.loc[df["Inference Time (ms)"].idxmin()]

        col1.metric("🥇 Best Accuracy",
                    f"{best_acc['Test Accuracy (%)']:.2f}%",
                    best_acc["Model"])
        col2.metric("⚡ Fastest",
                    f"{fastest['Inference Time (ms)']:.1f}ms",
                    fastest["Model"])
        col3.metric("📦 Models", "4 + YOLO")
        col4.metric("🎯 YOLO mAP50", "94.92%")

        st.dataframe(df, use_container_width=True)

        col5, col6 = st.columns(2)
        colors = ["gold", "steelblue", "orange", "green"]

        with col5:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(df["Model"], df["Test Accuracy (%)"], color=colors)
            ax.set_ylim(90, 100)
            ax.set_title("Test Accuracy (%)")
            for bar, val in zip(bars, df["Test Accuracy (%)"]):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.1,
                        f"{val}%", ha="center", fontsize=9, fontweight="bold")
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)

        with col6:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            bars2 = ax2.bar(df["Model"], df["Inference Time (ms)"], color=colors)
            ax2.set_title("Inference Time (ms/image)")
            for bar, val in zip(bars2, df["Inference Time (ms)"]):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.2,
                         f"{val}ms", ha="center", fontsize=9, fontweight="bold")
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig2)

        yolo_path = os.path.join(REPORTS_DIR, "yolo_detection_results.json")
        if os.path.exists(yolo_path):
            with open(yolo_path, "r") as f:
                yolo_data = json.load(f)
            st.markdown("### 🎯 YOLO Detection Results")
            yc1, yc2, yc3, yc4 = st.columns(4)
            yc1.metric("mAP50",     f"{yolo_data['mAP50']:.4f}")
            yc2.metric("mAP50-95",  f"{yolo_data['mAP50_95']:.4f}")
            yc3.metric("Precision", f"{yolo_data['precision']:.4f}")
            yc4.metric("Recall",    f"{yolo_data['recall']:.4f}")

        st.success("""
        **🏆 Best Overall: EfficientNetB0** — 98.75% accuracy, lowest loss
        **⚡ Best Speed: ResNet50** — 98.75% accuracy, 25.46ms inference
        **🎯 Detection: YOLOv8n** — mAP50: 94.92%
        """)
    else:
        st.warning("⚠️ model_comparison.csv not found in reports/ folder!")

# ── TAB 3 ──
with tab3:
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
    ### 🍌 Advanced Banana Quality Detection System
    End-to-end computer vision pipeline for banana ripeness classification and detection.

    ### 🔬 Models:
    | Model | Type | Accuracy |
    |---|---|---|
    | Custom CNN | Scratch | 97.51% |
    | MobileNetV2 | Transfer Learning | 97.51% |
    | EfficientNetB0 | Transfer Learning | 98.75% |
    | ResNet50 | Transfer Learning | 98.75% |
    | YOLOv8n | Object Detection | mAP50: 94.92% |

    ### 📊 Classes:
    - 🟢 **Unripe** — Not ready to eat
    - 🟡 **Ripe** — Perfect to eat
    - 🟤 **Overripe** — Best for baking
    - ⚫ **Rotten** — Not suitable

    ### 📦 Dataset:
    Roboflow Universe — CC BY 4.0

    ### 🛠️ Tech Stack:
    Python | TensorFlow/Keras | PyTorch | Streamlit | YOLOv8
    """)