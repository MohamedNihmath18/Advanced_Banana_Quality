import os
import sys

# ── CRITICAL: Environment Overrides to Prevent Segmentation Faults ──
# Force TensorFlow to use safe, low-RAM configurations inside Streamlit Cloud
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["XLA_FLAGS"] = "--xla_cpu_compilation_parallelism=1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import requests
from tensorflow.keras.models import load_model
from ultralytics import YOLO

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
</style>
""", unsafe_allow_html=True)

# ── Paths Configuration ──
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# ── GitHub Release Asset URL Mapping ──
GITHUB_USER = "mohamednihmath18"
REPO_NAME   = "advanced_banana_quality"
TAG_VERSION = "v1.0.0"

BASE_URL = f"https://github.com{GITHUB_USER}/{REPO_NAME}/releases/download/{TAG_VERSION}"

MODEL_URLS = {
    "Custom CNN":     f"{BASE_URL}/custom_cnn_model.keras",
    "MobileNetV2":    f"{BASE_URL}/mobilenetv2_model.keras",
    "EfficientNetB0": f"{BASE_URL}/efficientnetb0_model.keras",
    "ResNet50":       f"{BASE_URL}/resnet50_model.keras",
    "YOLO":           f"{BASE_URL}/yolo_detect_best.pt"
}

MODEL_FILES = {
    "Custom CNN":     "custom_cnn_model.keras",
    "MobileNetV2":    "mobilenetv2_model.keras",
    "EfficientNetB0": "efficientnetb0_model.keras",
    "ResNet50":       "resnet50_model.keras",
    "YOLO":           "yolo_detect_best.pt"
}

# ── Robust Streaming Asset Downloader via Requests ──
def download_if_missing(model_name, filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        download_url = MODEL_URLS.get(model_name)
        if download_url:
            with st.spinner(f"📥 Downloading {model_name} from GitHub Assets... Please wait."):
                try:
                    # stream=True handles files cleanly without running out of RAM
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    with requests.get(download_url, headers=headers, stream=True, allow_redirects=True) as r:
                        r.raise_for_status()
                        with open(path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                                
                    if os.path.exists(path) and os.path.getsize(path) > 5000:
                        st.sidebar.success(f"✅ {model_name} ready!")
                    else:
                        if os.path.exists(path): os.remove(path)
                        st.sidebar.error(f"❌ Download failed: File size validation error for {model_name}.")
                except Exception as e:
                    st.sidebar.error(f"❌ Network error downloading {model_name}: {e}")
        else:
            st.sidebar.error(f"⚠️ Link configuration mapping missing for: {model_name}")
    return path

# ── Optimized Lazy-Loading Cache Hooks ──
@st.cache_resource
def load_selected_classification_model(name):
    filename = MODEL_FILES.get(name)
    if not filename:
        return None
    path = download_if_missing(name, filename)
    if os.path.exists(path):
        try:
            # compile=False bypasses Keras version structural mismatch errors
            return load_model(path, compile=False)
        except Exception as e:
            st.error(f"Failed to load the model file binary: {e}")
            if os.path.exists(path): os.remove(path) # Wipe broken file to force clean re-download
    return None

@st.cache_resource
def load_yolo_model():
    path = download_if_missing("YOLO", "yolo_detect_best.pt")
    if os.path.exists(path):
        try:
            return YOLO(path)
        except Exception as e:
            st.error(f"YOLO backend runtime init exception: {e}")
            if os.path.exists(path): os.remove(path)
    return None

# ── Layout Header Application Title ──
st.markdown("<div class='main-title'>🍌 Advanced Banana Quality Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered banana ripeness classification & detection system</div>", unsafe_allow_html=True)
st.info("📦 Dataset: Banana Ripeness Classification — Roboflow Universe (CC BY 4.0)")

# ── Sidebar Settings UI Control Panel ──
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
    run_detection   = st.checkbox("Enable YOLO Detection", value=True)
    conf_threshold  = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

# ── Dynamic Model Invocations ──
with st.spinner("⏳ Configuring cloud runtime environment..."):
    active_classification_model = load_selected_classification_model(selected_model)
    if run_detection:
        yolo_model = load_yolo_model()
    else:
        yolo_model = None

if active_classification_model is None:
    st.error(f"❌ Cloud deployment failed to load {selected_model} from GitHub Release Assets.")
    st.stop()

# ── Navigational App Tabs Layout ──
tab1, tab2, tab3 = st.tabs(["🔍 Detection & Classification", "📊 Model Comparison", "ℹ️ About"])

with tab1:
    st.markdown("## 📸 Upload or Capture a Banana Image")
    image_input = None

    if input_method == "Upload Image":
        uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded: image_input = uploaded
    else:
        camera_img = st.camera_input("📷 Take a photo")
        if camera_img: image_input = camera_img

    if image_input is not None:
        col1, col2 = st.columns()

        with col1:
            st.markdown("### 🖼️ Input Image")
            img_original, img_resized = load_image(image_input)
            st.image(img_original, width='stretch')

        with col2:
            st.markdown("### 🤖 Classification Result")
            with st.spinner(f"Running Inference via {selected_model}..."):
                result = classify_image(active_classification_model, img_resized, selected_model)

            cls_name = result["class"]
            cls_info = CLASS_INFO.get(cls_name, {})

            st.markdown(f"""
            <div class="result-box">
                <h2>{cls_info.get("emoji","🍌")} {cls_name.upper()}</h2>
                <p>{cls_info.get("description","")}</p>
                <h3>Confidence: {result["confidence"]:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

        if run_detection and yolo_model is not None:
            st.markdown("---")
            st.markdown("### 🎯 YOLO Object Detection & Counting")
            image_input.seek(0)
            with st.spinner("Running YOLO architecture parsing..."):
                det_result = detect_bananas(yolo_model, image_input, conf_threshold)
                if det_result is not None:
                    st.image(det_result, width='stretch')
