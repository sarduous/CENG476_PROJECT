# 🫁 Lung Cancer Detection from Chest CT-Scans (VGG16)

![Status](https://img.shields.io/badge/Status-Completed-success)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-blue)
![Language](https://img.shields.io/badge/Python-3.x-yellow)

## 📌 Project Overview
This project implements a Deep Learning model to classify Chest CT-Scan images into 4 distinct categories to assist in the early diagnosis of lung cancer. The solution utilizes **Transfer Learning (VGG16)** combined with a **Two-Stage Fine-Tuning Strategy** to achieve high performance even on a limited dataset.

### 🎯 Classes
1. **Adenocarcinoma**
2. **Large Cell Carcinoma**
3. **Squamous Cell Carcinoma**
4. **Normal** (Healthy)

---

## 🚀 Key Features & Methodology
We implemented several advanced techniques to handle data imbalance and improve model robustness:

* **Transfer Learning:** Utilized **VGG16** (pretrained on ImageNet) as the feature extractor backbone.
* **Handling Imbalance:** Implemented **Automated Class Weighting** (`sklearn.utils.class_weight`) to penalize misclassification of minority classes.
* **Two-Stage Training Strategy:**
    1.  **Stage 1 (Feature Extraction):** Training the custom classifier head while keeping the VGG16 backbone frozen.
    2.  **Stage 2 (Fine-Tuning):** Unfreezing the last convolutional block (`block5_conv1`) with a very low learning rate (`1e-5`) to refine features.
* **Optimization:** Used `ReduceLROnPlateau` scheduler and `EarlyStopping` to prevent overfitting.
* **Reproducibility:** Fixed random seeds (Seed = 31) across TensorFlow, NumPy, and Python for consistent results.

---

## 🛠️ How to Run the Code

This project is designed to run seamlessly on **Google Colab**. Since the dataset is downloaded directly from Kaggle via API, you will need a personal Kaggle API token.

### Prerequisites
* A Google Account (for Colab).
* A Kaggle Account with an API Token (`kaggle.json`).

### 📥 Step 1: Open the Notebook
1. Download the file **`Lung_Cancer_Detection_VGG16.ipynb`** from this repository.
2. Upload it to [Google Colab](https://colab.research.google.com/).

### 🔑 Step 2: Kaggle API Setup (Important)
**For security reasons, the API token is NOT included in the code.** You must provide your own.

1. Run the **first code cell** (Section 1: Kaggle Configuration).
2. The code will pause and prompt you to upload a file with the message:
   > *"Upload your `kaggle.json` file now."*
3. Click the **"Choose Files"** button that appears and upload your `kaggle.json`.
   * *If you don't have a token, create one at: Kaggle Profile -> Settings -> Create New API Token.*
4. The script will automatically configure the environment, set permissions, and download the dataset.

#### 📄 Example `kaggle.json` Format
If you need to create the file manually, it should look exactly like this:

```json
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_API_KEY"
}
