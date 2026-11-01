#!/usr/bin/env python3
"""
Cats vs. Dogs Image Classifier — Production CLI Inference Interface & Core Engine
Repo: AIML_1_ImageClassification_byte
Project: Cats vs. Dogs Image Classification Pipeline (MobileNetV2 Deep Transfer Learning)

Usage:
    python src/predict.py outputs/sample_inferences/cat_01.jpg
    python src/predict.py path/to/any_real_photo.jpg
"""

import sys
import os
import io
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'cats_dogs_mobilenetv2.keras')
IMG_SIZE = (160, 160)

# Global cached model singleton
_CACHED_MODEL = None


def get_model(model_path: str = MODEL_PATH):
    """
    Load and cache the trained Keras model singleton once.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Target model not found at: {model_path}\n"
            "Please ensure 'cats_dogs_mobilenetv2.keras' is present in the models/ directory."
        )

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for model inference. Install via: pip install tensorflow"
        ) from exc

    # Load serialized Keras model
    _CACHED_MODEL = tf.keras.models.load_model(model_path)
    return _CACHED_MODEL


def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL image for MobileNetV2 inference:
    1. Convert to RGB
    2. Resize to (160, 160) using bilinear interpolation
    3. Scale pixel values to [-1, 1] range: (pixel / 127.5) - 1.0
    4. Expand batch dimension to (1, 160, 160, 3)
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_resized = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    
    # MobileNetV2 normalization: maps [0, 255] to [-1.0, 1.0]
    img_preprocessed = (img_array / 127.5) - 1.0
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    return img_batch


def run_inference_on_pil(img: Image.Image, model=None) -> Dict[str, Any]:
    """
    Run MobileNetV2 inference on an in-memory PIL image.
    Returns structured prediction dictionary with calibrated probabilities.
    """
    if model is None:
        model = get_model()

    start_t = time.perf_counter()
    original_size = list(img.size)
    img_batch = preprocess_pil_image(img)

    # Raw linear logit output from Dense(1) layer
    logits = model.predict(img_batch, verbose=0)[0][0]
    logit_val = float(logits)

    # Sigmoid activation: P(dog) = 1 / (1 + exp(-logits))
    prob_dog = float(1.0 / (1.0 + np.exp(-logit_val)))
    prob_cat = float(1.0 - prob_dog)

    # Class threshold: >= 0.50 -> DOG, else CAT
    label = "DOG" if prob_dog >= 0.50 else "CAT"
    confidence = prob_dog if label == "DOG" else prob_cat
    latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

    return {
        "label": label,
        "confidence": confidence,
        "prob_dog": prob_dog,
        "prob_cat": prob_cat,
        "logit": logit_val,
        "threshold": 0.50,
        "latency_ms": latency_ms,
        "original_size": original_size,
        "processed_size": list(IMG_SIZE),
    }


def predict(image_path: str) -> Tuple[str, float, float, float]:
    """
    Public CLI API preserving original signature:
    Returns (label, confidence, prob_dog, prob_cat).
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        model = get_model()
    except Exception as exc:
        print("=" * 65, file=sys.stderr)
        print("MODEL ARTIFACT / DEPENDENCY NOTICE:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("=" * 65, file=sys.stderr)
        sys.exit(1)

    try:
        with Image.open(image_path) as img:
            res = run_inference_on_pil(img, model=model)
    except Exception as exc:
        print(f"Error reading image {image_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    return res["label"], res["confidence"], res["prob_dog"], res["prob_cat"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    label, confidence, p_dog, p_cat = predict(image_path)

    print("-" * 55)
    print(f"INPUT IMAGE    : {os.path.basename(image_path)}")
    print(f"PREDICTED CLASS: [{label}]")
    print(f"CONFIDENCE     : {confidence * 100:.2f}%")
    print(f"P(DOG): {p_dog:.4f} | P(CAT): {p_cat:.4f}")
    print("-" * 55)


if __name__ == '__main__':
    main()
