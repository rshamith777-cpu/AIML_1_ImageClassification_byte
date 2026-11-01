# 🐾 Cats vs Dogs Image Classifier — Deep Transfer Learning (MobileNetV2)

<p align="center">
  <img src="outputs/project_banner.png" alt="Cats vs Dogs Image Classifier Banner" width="100%" />
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/rshamith777-cpu/AIML_1_ImageClassification_byte/blob/main/notebooks/cats_vs_dogs_transfer_learning.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" /></a>
  <a href="https://keras.io/"><img src="https://img.shields.io/badge/Architecture-MobileNetV2-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="MobileNetV2" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-gray?style=for-the-badge" alt="License" /></a>
</p>

---

## 📌 Executive Overview

| Program | Arithmatrix Virtual Internship Program (AVIP 2026) |
|---|---|
| **Domain** | Artificial Intelligence & Machine Learning (AI/ML) |
| **Track Ticket** | **Task 1 — Image Classification (Cats vs Dogs)** |
| **Repository Name** | `AIML_1_ImageClassification_byte` |
| **Primary Deliverables** | Repository, GPU Transfer Learning Notebook, Production `.keras` Model, CLI Predictor, Dataset Access Documentation, Executive Engineering Summary |

This repository implements an end-to-end deep convolutional computer vision pipeline engineered to classify real-world photographs as either feline (`cat`) or canine (`dog`). Rather than relying on fragile hand-crafted color histograms that fail to generalize across diverse real-world lighting and poses, this project employs **Deep Transfer Learning** via a pre-trained **MobileNetV2** backbone trained on ImageNet and fine-tuned on the real 25,000-image Microsoft / Kaggle Dogs vs. Cats benchmark.

---

## 📊 Dataset Reference & Access Instructions

- **Official Source:** [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) / [Microsoft Research Open Data](https://www.microsoft.com/en-us/download/details.aspx?id=54765).
- **Scale:** 25,000 raw RGB photographs (TFDS automatically filters corrupted/truncated headers yielding **23,262 verified images**: 11,631 Cats and 11,631 Dogs).
- **Deterministic Stratified Partitioning:** Stratified example-level shuffle with disjoint ranges:
  - **80% Training:** 18,608 images (9,304 Cats / 9,304 Dogs)
  - **10% Validation:** 2,326 images (1,163 Cats / 1,163 Dogs)
  - **10% Testing (Held-Out):** 2,328 images (1,164 Cats / 1,164 Dogs)
- **Detailed Manual Download Instructions:** See [`data/dataset_instructions.md`](data/dataset_instructions.md).

---

## 📈 Empirical Evaluation & Benchmark Results

The model was evaluated on the strictly held-out test split of **2,328 real photographic images** (never seen during feature extraction or fine-tuning).

### Classification Metrics (Held-Out Test Set)

| Evaluation Metric | Score | Note / Interpretation |
|---|---|---|
| **Test Accuracy** | **96.09%** | Correct classification across all 2,328 test samples |
| **Precision (Dog)** | **93.50%** | When predicting canine, 93.5% were true dogs |
| **Recall (Dog)** | **99.05%** | Successfully identified 99.05% of all canine photographs |
| **F1-Score (Dog)** | **96.19%** | Harmonic mean of precision and recall |
| **Held-Out Test Samples** | **2,328** | Balanced 50/50 test distribution |

<p align="center">
  <img src="outputs/confusion_matrix.png" alt="Confusion Matrix — Cats vs Dogs" width="480px" />
</p>

### Qualitative Held-Out Sample Predictions

Predictions and calibrated probabilities on 10 randomly selected held-out test photographs:

| Image File | Ground Truth | Predicted | $P(\text{Dog})$ | Confidence | Status |
|---|---|---|---|---|:---:|
| `cat_01.jpg` | Cat | **Cat** | 0.0023 | 99.77% | ✅ Correct |
| `cat_02.jpg` | Cat | **Cat** | 0.0220 | 97.80% | ✅ Correct |
| `cat_03.jpg` | Cat | **Cat** | 0.0431 | 95.69% | ✅ Correct |
| `cat_04.jpg` | Cat | **Cat** | 0.1114 | 88.86% | ✅ Correct |
| `cat_05.jpg` | Cat | **Cat** | 0.0040 | 99.60% | ✅ Correct |
| `dog_01.jpg` | Dog | **Dog** | 0.8987 | 89.87% | ✅ Correct |
| `dog_02.jpg` | Dog | **Dog** | 0.9362 | 93.62% | ✅ Correct |
| `dog_03.jpg` | Dog | **Dog** | 0.9286 | 92.86% | ✅ Correct |
| `dog_04.jpg` | Dog | **Dog** | 0.6855 | 68.55% | ✅ Correct |
| `dog_05.jpg` | Dog | **Dog** | 0.9008 | 90.08% | ✅ Correct |

*(Logged directly in [`outputs/sample_predictions.csv`](outputs/sample_predictions.csv) and stored in [`outputs/sample_inferences/`](outputs/sample_inferences/)).*

---

## 🧠 Deep Learning Architecture & Training Strategy

```
Input RGB Image (160x160x3)
        │
        ▼
[ Data Augmentation ] ────► RandomFlip('horizontal'), RandomRotation(0.1)
        │
        ▼
[ MobileNetV2 Backbone ] ─► Pre-trained on ImageNet (Frozen during Phase 1)
        │
        ▼
[ GlobalAveragePooling2D ]► Spatial feature pooling (reduces params & prevents overfitting)
        │
        ▼
[ Dropout(0.3) ] ─────────► Regularization
        │
        ▼
[ Dense(1) ] ─────────────► Linear Logit Output (Sigmoid: P(Dog) >= 0.50 -> DOG, else CAT)
```

### Two-Phase Training Protocol:
1. **Phase 1 (Feature Extraction):** MobileNetV2 base weights are frozen (`base_model.trainable = False`). The top classification head is trained for 3 epochs using the Adam optimizer (`lr=1e-3`) with binary cross-entropy loss.
2. **Phase 2 (Fine-Tuning):** The top 30 convolutional layers of MobileNetV2 are unfrozen and trained with a reduced learning rate (`lr=1e-5`) for 2 additional epochs, allowing deep filters to adapt specifically to pet anatomy (e.g., ear curvature, snout geometry, eye shape).

---

## 🚀 How to Train & Generate Real Artifacts (Google Colab GPU)

Because training deep convolutional neural networks on 25,000 high-resolution images requires a GPU, the notebook is designed to run on free Google Colab hardware:

1. Click the **Open In Colab** badge at the top of this README, or open [`notebooks/cats_vs_dogs_transfer_learning.ipynb`](notebooks/cats_vs_dogs_transfer_learning.ipynb).
2. Enable GPU acceleration: Go to **Runtime > Change runtime type > T4 GPU**.
3. Run all cells (**Runtime > Run all**). The notebook will:
   - Load the authentic 25,000-image dataset via TFDS.
   - Train MobileNetV2 and fine-tune on GPU (~25–35 minutes).
   - Evaluate on the 2,500 test images and compute Accuracy, Precision, Recall, F1, and Confusion Matrix.
   - Save the trained model to `models/cats_dogs_mobilenetv2.keras`.
   - Extract 10 sample test images into `outputs/sample_inferences/` and log predictions to `outputs/sample_predictions.csv`.
4. Download the artifacts and place them into your local repository clone:
   ```bash
   AIML_1_ImageClassification_byte/
   ├── models/cats_dogs_mobilenetv2.keras
   ├── outputs/confusion_matrix.png
   ├── outputs/metrics.json
   ├── outputs/sample_predictions.csv
   └── outputs/sample_inferences/ (10 test photos)
   ```

---

## 💻 CLI Inference Interface (`src/predict.py`)

Once your trained `.keras` model is placed in `models/`, run instant predictions on any real photograph:

```bash
python src/predict.py path/to/any_cat_or_dog_photo.jpg
```

**Example Output:**
```
-------------------------------------------------------
INPUT IMAGE    : my_dog.jpg
PREDICTED CLASS: [DOG]
CONFIDENCE     : 99.82%
P(DOG): 0.9982 | P(CAT): 0.0018
-------------------------------------------------------
```

---

## 📁 Repository Structure

```
AIML_1_ImageClassification_byte/
├── data/
│   └── dataset_instructions.md        # Kaggle & Microsoft dataset download instructions
├── notebooks/
│   └── cats_vs_dogs_transfer_learning.ipynb # Complete Colab GPU deep learning notebook
├── outputs/
│   ├── project_banner.png             # Repository hero banner
│   ├── confusion_matrix.png           # Confusion matrix visual (generated from Colab)
│   ├── metrics.json                   # Test evaluation metrics (Accuracy, Precision, Recall, F1)
│   ├── sample_predictions.csv         # Table with 10 test predictions
│   └── sample_inferences/             # 10 real test photos
├── models/
│   └── cats_dogs_mobilenetv2.keras    # Trained production deep learning model
├── src/
│   └── predict.py                     # Production CLI inference script (TensorFlow / Keras)
├── .gitignore                         # Clean repository exclusions
├── requirements.txt                   # Environment dependencies (TensorFlow, Pillow, NumPy)
└── README.md                          # Project documentation & engineering report
```

---

## 📝 Executive Engineering Summary (≤ 300 Words)

This project establishes an end-to-end computer vision pipeline engineered to classify real-world photographs as either feline (`cat`) or canine (`dog`) with calibrated confidence scores. Utilizing the Microsoft / Kaggle Dogs vs. Cats benchmark dataset (25,000 RGB images), the architecture demonstrates production-grade deep transfer learning.

Hand-crafted computer vision features (such as color histograms and spatial downsampling) historically struggle on natural images due to complex background clutter, fur pattern variations, and arbitrary camera poses, rarely exceeding 65–70% accuracy. To achieve industry-standard performance, this system employs **MobileNetV2** pre-trained on the ImageNet-1k dataset as a convolutional feature backbone. MobileNetV2 uses depthwise separable convolutions and inverted residual bottlenecks, delivering high parameter efficiency and near-instant inference latency.

The model is trained using a disciplined two-phase transfer learning protocol. In Phase 1, the pre-trained convolutional base is frozen while a custom classification head—consisting of a global average pooling layer, 30% dropout regularization, and a dense output layer—is trained on 20,000 augmented images using binary cross-entropy and the Adam optimizer ($10^{-3}$). In Phase 2, the top 30 layers of the base network are unfrozen for fine-tuning at a reduced learning rate ($10^{-5}$), enabling high-level feature detectors to adapt specifically to pet anatomy (e.g., pointed cat ears vs. rounded dog snouts).

On 2,328 strictly held-out test images, the model achieves **96.09% accuracy**, with **93.50% precision**, **99.05% recall**, and **96.19% F1-score**. The primary training notebook (`cats_vs_dogs_transfer_learning.ipynb`) is optimized for free Google Colab T4 GPU execution with automated dataset streaming. The final model is serialized as `cats_dogs_mobilenetv2.keras` and paired with a production CLI inference script (`src/predict.py`) for standalone deployment.
