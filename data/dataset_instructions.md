# Dataset Access & Preparation Guide

## Official Dataset Sources
1. **Kaggle Dogs vs. Cats Competition:**
   - Link: [kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data)
   - Size: ~543 MB (25,000 labeled images: 12,500 cats and 12,500 dogs)
   - CLI Download:
     ```bash
     kaggle competitions download -c dogs-vs-cats
     unzip dogs-vs-cats.zip
     unzip train.zip -d data/
     ```

2. **Microsoft Research Open Data (Cats and Dogs):**
   - Link: [microsoft.com/en-us/download/details.aspx?id=54765](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
   - Filtered dataset of 25,000 RGB photos in `PetImages/Cat` and `PetImages/Dog`.

## Local Organization
Once downloaded or extracted, place images into:
```
data/
├── train/
│   ├── cats/    # e.g., cat.0.jpg ... cat.9999.jpg
│   └── dogs/    # e.g., dog.0.jpg ... dog.9999.jpg
└── test/
    ├── cats/
    └── dogs/
```
For fast local experimentation, this repository also includes a lightweight CPU-optimized spatial feature pipeline with 10 verified test samples under `outputs/sample_inferences/`.
