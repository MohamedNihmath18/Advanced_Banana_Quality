import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import json
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import gdown

# Import custom processing utilities from your local utils.py file
from utils import (
    load_image, classify_image, detect_bananas,
    CLASS_INFO, CLASS_NAMES
)

st.set_page_config(
    page_title="🍌 Banana Quality Detector",
    page_icon="🍌",
    layout="wide"
)

# ── Custom CSS UI Styling ──
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
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths Configuration ──
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Ensure the models directory exists locally/on the server container
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# ── Google Drive File ID Mapping ──
# CRITICAL: Replace these string placeholders with your actual Google Drive shareable file IDs!
DRIVE_IDS = {
    "Custom CNN":     "14q_FPZdQ8sC1Hfnm-meSzN8eMPcLlZeq", 
    "MobileNetV2":    "1wP5bbqtB5j5PZfgHK0GiK0XNhIP_lWuQ", 
    "EfficientNetB0": "1C4EgCue5GOxke2fAiLkjC3MZpEYO1pIX",
    "ResNet50":       "1Z7N9PLeb_aDrmoxg3gyon4tmdVho-9Sf",  
    "YOLO":           "1LatSoXDZme8FlhQKD-dQcBWZF5kRFIiD"
}

MODEL_FILES = {
    "Custom CNN":     "custom_cnn_model.keras",
    "MobileNetV2":    "mobilenetv2_model.keras",
    "EfficientNetB0": "efficientnetb0_model.keras",
    "ResNet50":       "resnet50_model.keras",
}

# ── Automated On-Demand Downloader Helper ──
def download_if_missing(model_name, filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        file_id = DRIVE_IDS.get(model_name)
        if file_id and "YOUR_" not in file_id:
            download_url = f"https://drive.google.com/uc?id={file_id}"
            with st.spinner(f"📥 Downloading {model_name} architecture from Google Drive... Please wait."):
                try:
                    gdown.download(download_url, path, quiet=False)
                    st.sidebar.success(f"✅ {model_name} binary downloaded successfully!")
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to fetch weights for {model_name}: {e}")
        else:
            st.sidebar.error(f"⚠️ Missing Google Drive File ID config for: {model_name}")
    return path

# ── Optimized Lazy-Loading Cache Hooks ──
@st.cache_resource
def load_selected_classification_model(name):
    filename = MODEL_FILES[name]
    path = download_if_missing(name, filename)
    if os.path.exists(path):
        return load_model(path)
    return None

@st.cache_resource
def load_yolo_model():
    path = download_if_missing("YOLO", "yolo_detect_best.pt")
    if os.path.exists(path):
        return YOLO(path)
    return None

# ── Layout Header Application Title ──
st.markdown("<div class='main-title'>🍌 Advanced Banana Quality Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered banana ripeness classification & detection system</div>", unsafe_allow_html=True)
st.info("📦 Dataset: Banana Ripeness Classification — Roboflow Universe (CC BY 4.0)")

# ── Sidebar Settings UI Control Panel ──
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    input_method = st.radio(
        "📷 Input Method",
        ["Upload Image", "Camera Capture"]
    )

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
    run_detection   = st.checkbox("Enable YOLO Detection", value=True)
    conf_threshold  = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use **clear lighting**
    - Show **one banana clearly**
    - Avoid **cluttered backgrounds**
    """)

# ── Dynamic Model Invocations ──
# This only downloads/loads the ONE chosen classification model to optimize RAM runtime safety
with st.spinner("⏳ Configuring runtime environment..."):
    active_classification_model = load_selected_classification_model(selected_model)
    
    if run_detection:
        yolo_model = load_yolo_model()
    else:
        yolo_model = None

# Fallback error guard layer
if active_classification_model is None:
    st.error(f"❌ Could not load {selected_model}. Ensure your Google Drive File ID is mapped correctly.")
    st.stop()

# ── Navigational App Tabs Layout ──
tab1, tab2, tab3 = st.tabs([
    "🔍 Detection & Classification",
    "📊 Model Comparison",
    "ℹ️ About"
])

# ── TAB 1: Core Computer Vision Processing Engine ──
with tab1:
    st.markdown("## 📸 Upload or Capture a Banana Image")
    image_input = None

    if input_method == "Upload Image":
        uploaded = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"]
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
            st.image(img_original, use_container_width=True)

        with col2:
            st.markdown("### 🤖 Classification Result")
            with st.spinner(f"Running {selected_model}..."):
                result = classify_image(active_classification_model, img_resized, selected_model)

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
                det_result = detect_bananas(
                    yolo_model, image_input, conf_threshold
                )

            col3, col4 = st.columns([1, 1])
            with col3:
                st.image(
                    det_result["annotated_image"],
                    use_container_width=True,
                    caption="YOLO Detection Result"
                )
            with col4:
                st.markdown(f"""
                <div class="result-box">
                    <h2>🍌 Bananas Detected: {det_result["count"]}</h2>
                    <p>⚡ Detection Time: {det_result["inference_time"]:.1f} ms</p>
                </div>
                """, unsafe_allow_html=True)

                if det_result["count"] > 0:
                    st.markdown("**Detected Objects:**")
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
            st.warning("⚠️ YOLO model file not available!")

    else:
        st.markdown("""
        <div style="text-align:center; padding:50px; color:#666;">
            <h2>👆 Upload or capture a banana image to get started!</h2>
            <p>Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2: Model Performance Analytics ──
with tab2:
    st.markdown("## 📊 Model Comparison Dashboard")

    comparison_path = os.path.join(REPORTS_DIR, "model_comparison.csv")

    if os.path.exists(comparison_path):
        df = pd.read_csv(comparison_path)

        st.markdown("### 🏆 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        best_acc_row = df.loc[df["Test Accuracy (%)"].idxmax()]
        fastest_row  = df.loc[df["Inference Time (ms)"].idxmin()]

        col1.metric("🥇 Best Accuracy",
                    f"{best_acc_row['Test Accuracy (%)']:.2f}%",
                    best_acc_row["Model"])
        col2.metric("⚡ Fastest",
                    f"{fastest_row['Inference Time (ms)']:.1f}ms",
                    fastest_row["Model"])
        col3.metric("📦 Models", "4 + YOLO", "Classification + Detection")
        col4.metric("🎯 YOLO mAP50", "94.92%", "Object Detection")

        st.markdown("### 📋 Comparison Table")
        st.dataframe(df, use_container_width=True)

        col5, col6 = st.columns(2)
        colors = ["gold", "steelblue", "orange", "green"]

        with col5:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(df["Model"], df["Test Accuracy (%)"], color=colors)
            ax.set_ylim(90, 100)
            ax.set_title("Test Accuracy (%)")
            for bar, val in zip(bars, df["Test Accuracy (%)"]):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    f"{val}%", ha="center", fontsize=9, fontweight="bold"
                )
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)

        with col6:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            bars2 = ax2.bar(df["Model"], df["Inference Time (ms)"], color=colors)
            ax2.set_title("Inference Time (ms/image)")
            for bar, val in zip(bars2, df["Inference Time (ms)"]):
                ax2.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.2,
                    f"{val}ms", ha="center", fontsize=9, fontweight="bold"
                )
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
        **🏆 Best Overall: EfficientNetB0**
        - Highest accuracy (98.75%) + lowest loss (0.037)

        **⚡ Best Speed: ResNet50**
        - Same accuracy (98.75%) — faster inference (25.46ms)

        **🎯 Detection: YOLOv8n**
        - mAP50: 94.92% — detects & counts bananas
        """)
    else:
        st.warning("⚠️ model_comparison.csv not found!")

# ── TAB 3: Metadata / Documentation Documentation ──
with tab3:
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
    ### 🍌 Advanced Banana Quality Detection System
    End-to-end computer vision pipeline for banana ripeness classification and object detection.

    ### 🔬 Models Summary Metrics:
    | Model | Type | Accuracy |
    |---|---|---|
    | Custom CNN | Scratch | 97.51% |
    | MobileNetV2 | Transfer Learning | 97.51% |
    | EfficientNetB0 | Transfer Learning | 98.75% |
    | ResNet50 | Transfer Learning | 98.75% |
    | YOLOv8n | Object Detection | mAP50: 94.92% |

    ### 📊 Managed Target Classes:
    - 🟢 **Unripe** — Not ready to eat
    - 🟡 **Ripe** — Perfect to eat
    - 🟤 **Overripe** — Best for baking
    - ⚫ **Rotten** — Not suitable

    ### 📦 Dataset Attribution:
    - Roboflow Universe — CC BY 4.0

    ### 🛠️ Tech Stack:
    Python | TensorFlow | PyTorch | Streamlit | YOLOv8
    """)
    st.markdown("""
    <div class="footer">
        🍌 Banana Quality Detection | GUVI Final Project |
        Dataset: Roboflow Universe (CC BY 4.0)
    </div>
    """, unsafe_allow_html=True)