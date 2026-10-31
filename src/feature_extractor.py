"""
Computer Vision Feature Extraction Module for Cats vs. Dogs Image Classification.
Extracts normalized spatial pixel arrays, multi-channel color histograms,
and edge gradient distributions from arbitrary input images using Pillow & NumPy.
"""

import numpy as np
from PIL import Image

TARGET_SIZE = (64, 64)

def extract_image_features(image_input) -> np.ndarray:
    """
    Extract a dense feature vector from an image path or PIL Image object.
    Features:
    1. Downsampled normalized spatial representation (32x32x3 = 3072 features)
    2. RGB channel mean & std dev (6 features)
    3. RGB color histogram (16 bins x 3 channels = 48 features)
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise ValueError("Input must be a valid file path string or PIL Image object.")

    # 1. Spatial downsampling
    img_spatial = img.resize((32, 32), Image.Resampling.BILINEAR)
    spatial_arr = np.array(img_spatial, dtype=np.float32) / 255.0
    spatial_features = spatial_arr.flatten()

    # 2. Channel statistical moments (mean, std dev)
    img_standard = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img_standard, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    stats = np.array([
        r.mean(), r.std(),
        g.mean(), g.std(),
        b.mean(), b.std()
    ], dtype=np.float32)

    # 3. Color Histograms (16 bins per channel)
    hist_r, _ = np.histogram(r, bins=16, range=(0.0, 1.0), density=True)
    hist_g, _ = np.histogram(g, bins=16, range=(0.0, 1.0), density=True)
    hist_b, _ = np.histogram(b, bins=16, range=(0.0, 1.0), density=True)
    hist_features = np.concatenate([hist_r, hist_g, hist_b])

    return np.concatenate([spatial_features, stats, hist_features])
