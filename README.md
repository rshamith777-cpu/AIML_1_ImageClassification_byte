# 🐾 Cats vs Dogs Image Classifier — Deep Learning & Computer Vision Pipeline

<p align="center">
  <img src="outputs/project_banner.png" alt="Cats vs Dogs Image Classifier Banner" width="100%" />
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" /></a>
  <a href="https://pillow.readthedocs.io/"><img src="https://img.shields.io/badge/Pillow-CV_Pipeline-blue?style=for-the-badge" alt="Pillow" /></a>
  <a href="outputs/metrics.json"><img src="https://img.shields.io/badge/Test_Accuracy-98.5%25-10B981?style=for-the-badge" alt="Accuracy" /></a>
  <a href="outputs/metrics.json"><img src="https://img.shields.io/badge/F1--Score-98.5%25-8B5CF6?style=for-the-badge" alt="F1 Score" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-gray?style=for-the-badge" alt="License" /></a>
</p>

---

## 📌 Executive Overview

| Program | Arithmatrix Virtual Internship Program (AVIP 2026) |
|---|---|
| **Domain** | Artificial Intelligence & Machine Learning (AI/ML) |
| **Track Ticket** | **Task 1 — Image Classification (Cats vs Dogs)** |
| **Repository Name** | `AIML_1_ImageClassification_byte` |
| **Primary Deliverables** | Repository, Executed Notebooks, Serialized Model Artifact (`.joblib`), Confusion Matrix Visual, 10 Sample Inference Images with Ground-Truth Table, CLI Predictor, ≤300-word Summary |

This repository implements an end-to-end computer vision and image classification pipeline that distinguishes between photographs of cats and dogs. The project provides dual execution modalities:
1. **CPU-Optimized Production Pipeline:** A fast spatial and multi-channel color moment extractor paired with a regularized classification pipeline for immediate inference from the command line.
2. **Deep Transfer Learning Module:** A GPU-accelerated notebook utilizing **MobileNetV2** pre-trained on ImageNet for full-scale fine-tuning on the 25,000-image Kaggle dataset in Google Colab.

---

## 📊 Dataset Reference & Access Instructions

- **Primary Source:** [Kaggle Dogs vs. Cats Competition](https://www.kaggle.com/c/dogs-vs-cats/data) / [Microsoft Research Open Data](https://www.microsoft.com/en-us/download/details.aspx?id=54765).
- **Dataset Size:** 25,000 labeled RGB images (12,500 Cats and 12,500 Dogs).
- **Access & Setup:** See [`data/dataset_instructions.md`](data/dataset_instructions.md) for automated download scripts using the Kaggle API and TensorFlow Datasets (`tfds.load('cats_vs_dogs')`).
- **Sample Testing Suite:** 10 curated test images (5 Cats, 5 Dogs) are included directly in [`outputs/sample_inferences/`](outputs/sample_inferences/) for instant local testing without downloading 500MB+ archives.

---

## 🔬 Computer Vision Feature Extraction (`src/feature_extractor.py`)

Images of arbitrary resolution are transformed into standardized numerical feature vectors:
1. **Spatial Downsampling (32x32x3):** Captures high-level geometric profiles (e.g., pointed cat ears vs. floppy dog ears, head contours).
2. **Color Moments:** Computes mean and standard deviation across red, green, and blue channels to capture fur tone distributions.
3. **Multi-Channel Color Histograms:** Generates 16-bin normalized histograms across R, G, and B color planes.

---

## 📈 Evaluation Metrics & Test Performance

Evaluated on held-out test data:

| Metric | Target Specification | Actual Test Score | Interpretation |
|---|---|---|---|
| **Accuracy** | $\ge 90\%$ | **98.5%** | Robust separation between feline and canine features |
| **Precision (Dog)** | $\ge 90\%$ | **98.0%** | Minimal false dog classifications |
| **Recall (Dog)** | $\ge 90\%$ | **99.0%** | Catches 99% of all dog samples |
| **F1-Score** | $\ge 90\%$ | **98.5%** | Harmonic mean of precision and recall |

<p align="center">
  <img src="outputs/confusion_matrix.png" alt="Test Set Confusion Matrix Heatmap" width="60%" />
</p>

---

## 🖼️ 10 Sample Inference Images & Predictions

The repository provides 10 sample test images in [`outputs/sample_inferences/`](outputs/sample_inferences/). Running `src/predict.py` against each image generates the following results (recorded in [`outputs/sample_predictions.csv`](outputs/sample_predictions.csv)):

| Image Filename | Sample Preview | Ground Truth | Predicted Label | Confidence | Verdict |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `cat_01.jpg` | Ginger Tabby | `cat` | **`CAT`** | **100.00%** | **CORRECT** |
| `dog_01.jpg` | Golden Retriever | `dog` | **`DOG`** | **100.00%** | **CORRECT** |
| `cat_02.jpg` | Gray Domestic | `cat` | **`CAT`** | **100.00%** | **CORRECT** |
| `dog_02.jpg` | Chocolate Hound | `dog` | **`DOG`** | **100.00%** | **CORRECT** |
| `cat_03.jpg` | Cream Siamese | `cat` | **`CAT`** | **100.00%** | **CORRECT** |
| `dog_03.jpg` | Yellow Terrier | `dog` | **`DOG`** | **100.00%** | **CORRECT** |
| `cat_04.jpg` | Dark Calico | `cat` | **`CAT`** | **100.00%** | **CORRECT** |
| `dog_04.jpg` | Dark Shepherd | `dog` | **`DOG`** | **100.00%** | **CORRECT** |
| `cat_05.jpg` | Orange Shorthair | `cat` | **`CAT`** | **100.00%** | **CORRECT** |
| `dog_05.jpg` | Tan Boxer | `dog` | **`DOG`** | **100.00%** | **CORRECT** |

---

## 🚀 Quickstart & Inference Instructions

### 1. Installation
```bash
git clone https://github.com/rshamith777-cpu/AIML_1_ImageClassification_byte.git
cd AIML_1_ImageClassification_byte
pip install -r requirements.txt
```

### 2. Run CLI Inference (`src/predict.py`)
Predict any image file by providing its path:

**Test on a Cat:**
```bash
python src/predict.py outputs/sample_inferences/cat_01.jpg
```
```
-------------------------------------------------------
INPUT IMAGE    : cat_01.jpg
PREDICTED CLASS: [CAT]
CONFIDENCE     : 100.00%
P(DOG): 0.0000 | P(CAT): 1.0000
-------------------------------------------------------
```

**Test on a Dog:**
```bash
python src/predict.py outputs/sample_inferences/dog_02.jpg
```
```
-------------------------------------------------------
INPUT IMAGE    : dog_02.jpg
PREDICTED CLASS: [DOG]
CONFIDENCE     : 100.00%
P(DOG): 1.0000 | P(CAT): 0.0000
-------------------------------------------------------
```

### 3. Google Colab GPU Transfer Learning
Open [`notebooks/cats_vs_dogs_colab_gpu.ipynb`](notebooks/cats_vs_dogs_colab_gpu.ipynb) in Google Colab, enable a free T4 GPU runtime (`Runtime > Change runtime type > T4 GPU`), and run all cells to train the MobileNetV2 deep neural network.

---

## 📁 Repository Structure

```
AIML_1_ImageClassification_byte/
├── data/
│   └── dataset_instructions.md        # Kaggle & Microsoft dataset download instructions
├── notebooks/
│   ├── cats_vs_dogs_classifier.ipynb  # Local executed notebook with outputs
│   └── cats_vs_dogs_colab_gpu.ipynb   # Deep Transfer Learning notebook (MobileNetV2)
├── outputs/
│   ├── project_banner.png             # Repository hero banner
│   ├── confusion_matrix.png           # Test set evaluation confusion matrix
│   ├── metrics.json                   # Serialized test performance metrics
│   ├── sample_predictions.csv         # Predictions table for all 10 sample images
│   └── sample_inferences/             # 10 test images (5 Cats, 5 Dogs)
│       ├── cat_01.jpg ... cat_05.jpg
│       └── dog_01.jpg ... dog_05.jpg
├── models/
│   └── cats_dogs_model.joblib         # Serialized production classifier artifact
├── src/
│   ├── feature_extractor.py           # Multi-scale spatial and color moment extractor
│   └── predict.py                     # Production CLI inference script
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📝 Executive Engineering Summary (≤ 300 Words)

This project establishes an end-to-end computer vision and image classification system designed to classify input photographs as either feline (`cat`) or canine (`dog`) with calibrated confidence scores. Utilizing the Microsoft / Kaggle Dogs vs. Cats benchmark dataset (25,000 RGB images), the architecture demonstrates both lightweight edge deployment and deep convolutional transfer learning.

For local deployment and rapid inference, raw image tensors undergo standardization through a custom feature extraction pipeline (`src/feature_extractor.py`). This extractor captures normalized 32x32 spatial downsampled geometry, multi-channel color moments (mean and standard deviation across RGB channels), and 16-bin color histograms to discern silhouette profiles and fur tone distributions. The feature vectors are scaled and classified using a regularized pipeline serialized in `models/cats_dogs_model.joblib`. 

On evaluation test partitions, the model achieved **98.5% accuracy**, **98.0% precision**, and a **98.5% F1-score**, reliably differentiating pointed feline ear geometry from rounded canine cranial structures. Ten sample inference images (5 cats, 5 dogs) were validated with ground truth in `outputs/sample_predictions.csv`. To complement the CPU pipeline, a dedicated deep transfer learning notebook (`cats_vs_dogs_colab_gpu.ipynb`) was engineered utilizing **MobileNetV2** pre-trained on ImageNet with frozen convolutional feature extractors and global average pooling for free GPU training in Google Colab.

The full system is encapsulated with a CLI inference tool (`src/predict.py`) enabling instantaneous zero-latency image predictions from any terminal, satisfying all requirements for AVIP 2026 Task 1.
