#!/usr/bin/env python3
"""
Cats vs. Dogs Image Classifier — Production CLI Inference Interface
Repo: AIML_1_ImageClassification_byte
Project: Cats vs. Dogs Image Classification Pipeline (MobileNetV2 Deep Transfer Learning)

Usage:
    python src/predict.py outputs/sample_inferences/cat_01.jpg
    python src/predict.py path/to/any_real_photo.jpg
"""

import sys
import os
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'cats_dogs_mobilenetv2.keras')
IMG_SIZE = (160, 160)

def predict(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print("=" * 65)
        print("MODEL ARTIFACT NOTICE:")
        print(f"Target model not found at: {MODEL_PATH}")
        print("Please run notebooks/cats_vs_dogs_transfer_learning.ipynb in Google Colab")
        print("to train MobileNetV2 on GPU, download 'cats_dogs_mobilenetv2.keras',")
        print("and place it into models/.")
        print("=" * 65)
        sys.exit(1)

    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow is required for inference. Run: pip install tensorflow")
        sys.exit(1)

    # 1. Load trained Keras model
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Load & preprocess image using MobileNetV2 scaling
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    # MobileNetV2 expects [-1, 1] normalized pixel tensors
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # 3. Model inference (raw logit output)
    logits = model.predict(img_batch, verbose=0)[0][0]
    
    # 4. Sigmoid activation -> P(dog)
    prob_dog = float(1.0 / (1.0 + np.exp(-logits)))
    prob_cat = 1.0 - prob_dog

    label = "DOG" if prob_dog >= 0.50 else "CAT"
    confidence = prob_dog if label == "DOG" else prob_cat

    return label, confidence, prob_dog, prob_cat

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
