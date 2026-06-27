import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import time

CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]

CLASS_INFO = {
    "ripe": {
        "emoji": "🟡",
        "description": "Perfectly ripe banana — ready to eat!",
        "color": "#FFD700"
    },
    "overripe": {
        "emoji": "🟤",
        "description": "Overripe banana — best for smoothies or baking.",
        "color": "#8B4513"
    },
    "unripe": {
        "emoji": "🟢",
        "description": "Unripe banana — needs more time to ripen.",
        "color": "#228B22"
    },
    "rotten": {
        "emoji": "⚫",
        "description": "Rotten banana — not suitable for consumption.",
        "color": "#2F2F2F"
    }
}

def load_image(image_input, target_size=(224, 224)):
    img = Image.open(image_input).convert("RGB")
    img_resized = img.resize(target_size)
    return img, img_resized

def preprocess_for_model(img_resized, model_name):
    img_array = np.array(img_resized).astype(np.float32)
    img_expanded = np.expand_dims(img_array, axis=0)
    if model_name == "Custom CNN":
        return img_expanded / 255.0
    elif model_name == "MobileNetV2":
        return mobilenet_preprocess(img_expanded)
    elif model_name == "EfficientNetB0":
        return efficientnet_preprocess(img_expanded)
    elif model_name == "ResNet50":
        return resnet_preprocess(img_expanded)
    else:
        return img_expanded / 255.0

def classify_image(model, img_resized, model_name):
    processed = preprocess_for_model(img_resized, model_name)
    start = time.time()
    predictions = model.predict(processed, verbose=0)
    end = time.time()
    inference_time = (end - start) * 1000
    predicted_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[0][predicted_idx]) * 100
    all_confidences = {
        CLASS_NAMES[i]: float(predictions[0][i]) * 100
        for i in range(len(CLASS_NAMES))
    }
    return {
        "class": predicted_class,
        "confidence": confidence,
        "all_confidences": all_confidences,
        "inference_time": inference_time
    }

def detect_bananas(yolo_model, image_input, conf_threshold=0.25):
    img = Image.open(image_input).convert("RGB")
    start = time.time()
    results = yolo_model.predict(
        np.array(img), conf=conf_threshold, verbose=False
    )
    end = time.time()
    inference_time = (end - start) * 1000
    result = results[0]
    banana_count = len(result.boxes)
    detected_classes = []
    confidences = []
    if banana_count > 0:
        yolo_class_names = ["freshripe", "freshunripe", "overripe",
                            "ripe", "rotten", "unripe"]
        for box in result.boxes:
            cls_idx = int(box.cls.item())
            conf = float(box.conf.item()) * 100
            detected_classes.append(yolo_class_names[cls_idx])
            confidences.append(conf)
    annotated = result.plot()
    annotated_pil = Image.fromarray(annotated[:, :, ::-1])
    return {
        "count": banana_count,
        "detected_classes": detected_classes,
        "confidences": confidences,
        "annotated_image": annotated_pil,
        "inference_time": inference_time
    }
