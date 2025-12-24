# CENG476_PROJECT
# 🫁 Lung Cancer Detection from Chest CT-Scans (VGG16)

![Status](https://img.shields.io/badge/Status-Completed-success)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-blue)

## 📌 Project Overview
This project implements a Deep Learning model to classify Chest CT-Scan images into 4 distinct categories to assist in the early diagnosis of lung cancer. The solution utilizes **Transfer Learning (VGG16)** combined with a **Two-Stage Fine-Tuning Strategy** to achieve high performance even on a limited dataset.

### 🎯 Classes
1. **Adenocarcinoma**
2. **Large Cell Carcinoma**
3. **Squamous Cell Carcinoma**
4. **Normal** (Healthy)

---

## 🚀 Key Features
* **Transfer Learning:** VGG16 (pretrained on ImageNet) used as the backbone.
* **Handling Imbalance:** Automated **Class Weighting** (`sklearn`) applied to penalize misclassification of minority classes.
* **Training Strategy:** A robust **Two-Stage** approach:
    1.  *Feature Extraction:* Training only the classifier head.
    2.  *Fine-Tuning:* Unfreezing the last block (`block5_conv1`) with a low learning rate (`1e-5`).
* **Reproducibility:** Random seeds fixed to **31** for consistent results.
* **Optimization:** `ReduceLROnPlateau` and `EarlyStopping` implemented to prevent overfitting.

---

## 🛠️ How to Run the Code (Step-by-Step)

This project is designed to run seamlessly on **Google Colab**. Since the dataset is downloaded directly from Kaggle, you will need a Kaggle API token.

### Prerequisites
* A Google Account (for Colab).
* A Kaggle Account with an API Token (`kaggle.json`).

### 📥 Step 1: Get the Code
1. Download the `.ipynb` file from this repository.
2. Upload it to [Google Colab](https://colab.research.google.com/).

### 🔑 Step 2: Kaggle API Setup (Crucial)
**For security reasons, the API token is NOT included in the code.** You must provide your own.

1. Run the **first code cell** (Section 1: Kaggle Configuration).
2. The code will pause and prompt you to upload a file:
   > *"Upload your `kaggle.json` file now."*
3. Click the **"Choose Files"** button that appears and upload your `kaggle.json`.
   * *If you don't have one: Go to Kaggle -> Settings -> Create New API Token.*
4. The script will automatically configure the environment and download the dataset.

### ▶️ Step 3: Train & Evaluate
1. Run the remaining cells sequentially.
2. The notebook will:
   * Preprocess the data.
   * Train the model (approx. 15-20 mins on GPU).
   * Display Training/Validation graphs.
   * Show the **Confusion Matrix** and **Classification Report**.
   * Run a demo on random test images at the end.

---

## 📊 Results
The model was evaluated on a held-out test set (~315 images).

| Metric | Performance | Notes |
| :--- | :--- | :--- |
| **Overall Accuracy** | **~80%** | Solid performance given the small dataset size. |
| **F1-Score (Normal)** | **0.97** | Excellent at identifying healthy patients. |
| **Recall (Squamous)**| **0.92** | High sensitivity for Squamous Cell Carcinoma. |

*(Detailed metrics and confusion matrix are visualized within the notebook)*

---

## 📂 Project Structure
```text
Lung-Cancer-Detection-VGG16/
│
├── Lung_Cancer_Detection_Project.ipynb  <-- MAIN CODE (Run this)
├── README.md                            <-- Documentation
└── .gitignore                           <-- System files ignored
