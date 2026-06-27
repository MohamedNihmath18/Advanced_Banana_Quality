# 🍌 Advanced Banana Quality Detector

An end-to-end computer vision and deep learning pipeline designed to automate banana quality control by integrating multi-architecture image classification with real-time object detection and counting. 

Live Demo: [advancedbananaquality-nihmath.streamlit.app](https://advancedbananaquality-nihmath.streamlit.app/)

---

## 📌 Project Overview
In industrial food processing and smart agricultural supply chains, tracking fruit ripeness accurately is essential to minimize waste. This project addresses the challenge by creating a two-fold vision intelligence system:
1. **Multi-Model Image Classification:** Compares a custom-built Convolutional Neural Network (CNN) against state-of-the-art Transfer Learning architectures to classify banana ripeness into 4 distinct phases.
2. **Object Detection & Counting:** Utilizes a custom-trained YOLOv8 model to detect, locate, and count individual bananas within cluttered frames or clusters.

---

## 📊 Model Performance & Comparison
The models were trained using the **Banana Ripeness Classification Dataset** from Roboflow Universe (CC BY 4.0). Evaluation metrics demonstrate high precision across deep learning frameworks:

| Model Architecture | Type | Parameters | Test Accuracy | Inference Speed |
| :--- | :--- | :--- | :--- | :--- |
| **EfficientNetB0** | Transfer Learning | 4.4M | **98.75%** | 41.86 ms |
| **ResNet50** | Transfer Learning | 24.2M | **98.75%** | **25.46 ms** |
| **Custom CNN** | Trained from Scratch | 26.4M | 97.51% | 16.54 ms |
| **MobileNetV2** | Transfer Learning | 2.6M | 97.51% | 49.01 ms |
| **YOLOv8n** | Object Detection | 3.2M | **94.92% (mAP50)** | — |

### 🏆 Analytical Key Takeaways:
* **Best Overall Accuracy:** `EfficientNetB0` achieved the top validation accuracy profile with the lowest recorded categorical cross-entropy loss (0.037).
* **Best Production Performance:** `ResNet50` matches the top accuracy tier while significantly reducing inference latency down to **25.46ms**, making it ideal for edge deployment.
* **Localization Capabilities:** `YOLOv8n` successfully localizes multi-object boundaries with a **94.92% mAP50**, enabling automated batch processing.

---

## 🗂️ Project Architecture
```text
advanced_banana_quality/
│
├── app/
│   ├── app.py          # Streamlit UI dashboard and lazy-loading logic
│   └── utils.py        # Image pre-processing and model inference functions
│
├── notebooks/          # Jupyter Notebooks for data prep, training, and metrics
│   ├── 01_data_preparation.ipynb
│   ├── 02_custom_cnn_training.ipynb
│   ├── 03_transfer_learning_comparison.ipynb
│   ├── 04_yolo_detection_count.ipynb
│   └── 05_model_evaluation.ipynb
│
├── reports/            # Exported metrics (CSV/JSON) used to render charts dynamically
│   ├── model_comparison.csv
│   └── yolo_detection_results.json
│
├── screenshots/        # Saved confusion matrices, ROC curves, and sample outputs
├── .gitignore          # Prevents massive image datasets and heavy model weights from being tracked
├── packages.txt        # Native headless Linux OS multimedia drivers 
└── requirements.txt    # Python library dependencies


# 1. Clone the repository
git clone [https://github.com/MohamedNihmath18/Advanced_Banana_Quality.git](https://github.com/MohamedNihmath18/Advanced_Banana_Quality.git)
cd Advanced_Banana_Quality

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit Dashboard
streamlit run app/app.py


# 🧠 Targeted Classification Metadata
# The vision system tracks and handles banana specimens based on the following classification metadata mapping:

# 🟢 Unripe: Firm green structural profile; starch layers intact. Needs more time to ripen.

# 🟡 Ripe: Ideal yellow coloration scale. Perfect for immediate consumption.

# 🟤 Overripe: Increasing sugar spots and soft texture composition. Excellent for baking or blending.

# ⚫ Rotten: Advanced enzymatic breakdown; structural decay. Unsafe for food consumption metrics.

# 📜 License & Acknowledgments
# Dataset Source: Roboflow Universe Platform (CC BY 4.0).


***

This markdown is comprehensive, clean, properly utilizes technical structural references, and showcases your software engineering approach alongside the data science metrics!