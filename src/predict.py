#!/usr/bin/env python3
"""
Cats vs. Dogs Image Classifier — CLI Inference Interface
Repo: AIML_1_ImageClassification_byte
Project: Cats vs. Dogs Image Classification Pipeline

Usage:
    python src/predict.py outputs/sample_inferences/cat_01.jpg
    python src/predict.py outputs/sample_inferences/dog_02.jpg
"""

import sys
import os

# Ensure repo root is in python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import joblib
from PIL import Image
from src.feature_extractor import extract_image_features

MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'cats_dogs_model.joblib')

def predict(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model artifact not found at {MODEL_PATH}")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    features = extract_image_features(image_path).reshape(1, -1)
    
    proba = model.predict_proba(features)[0]
    classes = list(model.classes_)
    dog_idx = classes.index('dog')
    cat_idx = classes.index('cat')
    
    dog_proba = float(proba[dog_idx])
    cat_proba = float(proba[cat_idx])
    
    label = "DOG" if dog_proba >= 0.50 else "CAT"
    confidence = dog_proba if label == "DOG" else cat_proba
    
    return label, confidence, dog_proba, cat_proba

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image.jpg>")
        sys.exit(1)

    img_path = sys.argv[1]
    label, conf, p_dog, p_cat = predict(img_path)

    print("-" * 55)
    print(f"INPUT IMAGE    : {os.path.basename(img_path)}")
    print(f"PREDICTED CLASS: [{label}]")
    print(f"CONFIDENCE     : {conf * 100:.2f}%")
    print(f"P(DOG): {p_dog:.4f} | P(CAT): {p_cat:.4f}")
    print("-" * 55)

if __name__ == '__main__':
    main()
