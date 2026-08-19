# 🩺 Diabetes Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-green?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**A Machine Learning + Deep Learning powered Diabetes Prediction Web App**

Built with ANN, Random Forest, SMOTE, and deployed via Streamlit

[🔴 Live Demo](https://diabetes-ann-model-mahadi.streamlit.app/) · [📊 Dataset](#-dataset) · [🚀 Quick Start](#-quick-start) · [📖 Documentation](#-project-structure)
 
</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features) 
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Performance](#-model-performance)
- [Results & Comparison](#-results--comparison)
- [Feature Importance](#-feature-importance)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [How to Run](#-how-to-run)
- [Requirements](#-requirements)
- [Streamlit App Features](#-streamlit-app-features)
- [Risk Level Classification](#-risk-level-classification)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)
- [License](#-license)
- [Author](#-author)

---

## 🌟 Overview

The **Diabetes Prediction System** is an end-to-end Machine Learning project that predicts whether a patient is at risk of diabetes based on clinical measurements.

This project covers the **complete ML lifecycle**:

```text
🗂️ Data Loading → 🧹 Preprocessing → 📊 EDA → ⚙️ Feature Engineering
→ 🤖 Model Training → 📈 Evaluation → 🌐 Deployment (Streamlit)
```

Two core models were built and compared:

- **Random Forest Classifier** — Ensemble tree-based model
- **Artificial Neural Network (ANN)** — Deep learning with multiple improvement strategies

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧹 **Smart Preprocessing** | Replaced biologically impossible zero values with median imputation |
| 📊 **Rich EDA** | Histograms, boxplots, heatmaps, pairplots, and outcome comparisons |
| ⚖️ **Class Imbalance Handling** | SMOTE oversampling + Class weight balancing |
| 🤖 **Multiple Models** | Random Forest vs ANN (4 versions) |
| 📉 **Overfitting Prevention** | Dropout layers + Early stopping callbacks |
| 📈 **Comprehensive Metrics** | Accuracy, Precision, Recall, F1 Score, ROC-AUC |
| 🌐 **Web App** | Interactive Streamlit UI with risk meter |
| 💾 **Model Persistence** | Saved model + scaler for deployment |

---

## 🎬 Demo

```text
🩺 Diabetes Prediction System
├── Enter patient details (Glucose, BMI, Age, etc.)
├── Click "Predict Diabetes"
├── See: Probability % + Risk Level (Low/Medium/High)
├── View: Risk Meter + Health Recommendations
└── Result: ✅ Low Risk or ⚠️ High Risk
```

> **Note:** The app runs locally. Follow [Quick Start](#-quick-start) to launch it.

---

## 📊 Dataset

**Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

| Property | Value |
|---|---|
| Total Rows | 768 |
| Total Features | 8 input + 1 target |
| Target Variable | Outcome (0 = No Diabetes, 1 = Diabetes) |
| Class Distribution | 65.1% No Diabetes · 34.9% Diabetes |

### 📋 Features Description

| Feature | Description | Unit |
|---|---|---|
| Pregnancies | Number of pregnancies | Count |
| Glucose | Plasma glucose concentration | mg/dL |
| BloodPressure | Diastolic blood pressure | mm Hg |
| SkinThickness | Triceps skin fold thickness | mm |
| Insulin | 2-Hour serum insulin | mu U/ml |
| BMI | Body Mass Index | kg/m² |
| DiabetesPedigreeFunction | Genetic diabetes risk score | Score |
| Age | Patient age | Years |
| Outcome | Diabetes diagnosis | 0 / 1 |

---

## 📁 Project Structure

```text
diabetes-ann-model/
│
├── 📓 diabetes_prediction.ipynb   # Complete ML notebook
├── 🌐 app.py                      # Streamlit web application
├── 🗃️ diabetes.csv                # Dataset
├── 🤖 diabetes_ann_model.h5       # Saved ANN model
├── ⚙️ scaler.pkl                  # Saved StandardScaler
├── 📊 final_model_analysis.png    # Model analysis plots
├── 📄 requirements.txt            # Python dependencies
└── 📖 README.md                   # Project documentation
```

---

## 🔬 Machine Learning Pipeline

### 1️⃣ Data Preprocessing

```python
# Step 1: Identified biologically impossible zero values
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Zero value counts found:
# Glucose: 5  | BloodPressure: 35 | SkinThickness: 227
# Insulin: 374 | BMI: 11

# Step 2: Replace zeros with NaN → Fill with median
df[zero_cols] = df[zero_cols].replace(0, np.nan)
for col in zero_cols:
    df[col] = df[col].fillna(df[col].median())
```

### 2️⃣ Outlier Analysis

```python
# IQR-based outlier detection
# Pregnancies: 4 (0.5%)   | BloodPressure: 14 (1.8%)
# SkinThickness: 87 (11.3%) | Insulin: 346 (45.1%)
# DiabetesPedigreeFunction: 29 (3.8%)
```

> ⚠️ **Insulin Note:** 374 rows had zero insulin (biologically invalid). After median imputation, the median value 125.0 repeated ~49% of the time. This is a known dataset limitation.

### 3️⃣ Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # Maintains class balance
)
# Train: 614 samples | Test: 154 samples
```

### 4️⃣ Feature Scaling

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Fit only on train!
X_test  = scaler.transform(X_test)        # Transform test
```

### 5️⃣ Class Imbalance Strategies

```text
Strategy 1 → Class Weights      (ANN v2)
Strategy 2 → SMOTE Oversampling (ANN v3)
Strategy 3 → SMOTE + Weights    (ANN v4)

Before SMOTE: Class 0 = 400  |  Class 1 = 214
After  SMOTE: Class 0 = 400  |  Class 1 = 400 ✅
```

### 6️⃣ ANN Architecture

```text
Input Layer   →  8 features
              ↓
Dense(16, relu)  →  Dropout(0.3)
              ↓
Dense(8,  relu)  →  Dropout(0.2)
              ↓
Dense(1, sigmoid)
              ↓
Output        →  Probability [0, 1]

Optimizer : Adam
Loss      : Binary Crossentropy
Callbacks : EarlyStopping (patience=10, restore_best_weights=True)
```

---

## 📈 Model Performance

Complete comparison across all five models:

| Metric | Random Forest | ANN v1 (Baseline) | ANN v2 (Class Weights) | ANN v3 (SMOTE) | ANN v4 (Both) |
|---|---|---|---|---|---|
| **Accuracy** | 0.7403 | 0.7338 | 0.7338 | **0.7662** | 0.7013 |
| **Precision** | **0.6667** | 0.6327 | 0.5867 | 0.6552 | 0.5526 |
| **Recall** | 0.5185 | 0.5741 | **0.8148** | 0.7037 | 0.7778 |
| **F1 Score** | 0.5833 | 0.6019 | 0.6822 | **0.6786** | 0.6462 |
| **ROC AUC** | 0.8144 | 0.8202 | 0.8030 | **0.8230** | 0.7765 |

*CW = Class Weight · SMOTE = Synthetic Minority Oversampling*

> **Note:** Bold marks the best value per metric. ANN v2's raw F1 (0.6822) is marginally higher than ANN v3's, but ANN v3 is highlighted as the balanced pick because it pairs the best accuracy (76.62%) with the best ROC-AUC (82.30%).

---

## 📊 Results & Comparison

### 🏆 Best Model Selection

| Use Case | Recommended Model | Key Metric |
|---|---|---|
| Medical use (minimize missed cases) | ANN v2 (Class Weights) | Recall: **81.48%** |
| Balanced performance | ANN v3 (SMOTE) | F1: **67.86%** · AUC: **82.30%** |
| Overall accuracy | ANN v3 (SMOTE) | Accuracy: **76.62%** |

In medical applications, **Recall (Sensitivity) is more critical than Precision** — missing a diabetic patient is more costly than a false alarm. ANN v2 with class weights achieves the highest recall of **81.48%**, making it the recommended model for screening-oriented use.

### 🔍 Key Takeaways

- **SMOTE (ANN v3)** delivered the biggest accuracy and ROC-AUC boost over the baseline ANN.
- **Class weighting (ANN v2)** maximizes recall — best at catching true positives.
- **Random Forest** offers the best precision, useful when false alarms are costly.

---

## 🔥 Feature Importance

Based on Random Forest feature importance:

| Rank | Feature | Importance |
|---|---|---|
| 🥇 1 | **Glucose** | **38.3%** |
| 🥈 2 | BMI | 16.3% |
| 🥉 3 | Age | 11.2% |
| 4 | Insulin | 10.1% |
| 5 | Diabetes Pedigree Function | 8.5% |
| 6 | Pregnancies | 6.1% |
| 7 | Skin Thickness | 5.3% |
| 8 | Blood Pressure | 4.2% |

> 💡 **Key Insight:** Glucose level is by far the most important predictor — accounting for ~38% of the model's decisions.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.12 |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-Learn (Random Forest) |
| Deep Learning | TensorFlow / Keras (ANN) |
| Imbalance | Imbalanced-Learn (SMOTE) |
| Deployment | Streamlit |
| Persistence | Joblib (scaler), H5 (model) |

---

## 🚀 Quick Start

### ✅ Prerequisites

- Python >= 3.9
- pip (package manager)

### 📥 Installation

```bash
# 1. Clone the repository
git clone https://github.com/mahadiurrehman-pixel/diabetes-ann-model.git

# 2. Navigate to project directory
cd diabetes-ann-model

# 3. Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🌐 Run Streamlit Web App

```bash
streamlit run app.py
```

Then open your browser at → `http://localhost:8501`

### 📓 Run Jupyter Notebook

```bash
jupyter notebook diabetes_prediction.ipynb
```

---

## 📦 Requirements

```txt
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
imbalanced-learn>=0.11.0
streamlit>=1.28.0
joblib>=1.3.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🌐 Streamlit App Features

```text
┌─────────────────────────────────────────────┐
│           🩺 Diabetes Prediction             │
├─────────────────────────────────────────────┤
│  📝 Input Panel                             │
│     ├── Pregnancies     ├── Insulin         │
│     ├── Glucose         ├── BMI             │
│     ├── Blood Pressure  ├── Diabetes PF     │
│     └── Skin Thickness  └── Age             │
├─────────────────────────────────────────────┤
│  📊 Output Panel                            │
│     ├── Diabetes Probability (%)            │
│     ├── Risk Level (Low/Medium/High)        │
│     ├── Risk Meter (Progress Bar)           │
│     └── Health Recommendations              │
└─────────────────────────────────────────────┘
```

### 🚦 Risk Level Classification

| Probability | Risk Level | Action |
|---|---|---|
| < 30% | 🟢 Low Risk | Maintain healthy lifestyle |
| 30% – 70% | 🟡 Medium Risk | Monitor regularly |
| > 70% | 🔴 High Risk | Consult doctor immediately |

---

## 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/YourFeatureName

# 3. Commit your changes
git commit -m "Add: YourFeatureName"

# 4. Push to the branch
git push origin feature/YourFeatureName

# 5. Open a Pull Request
```

### 💡 Ideas for Contribution

- Add more ML models (XGBoost, SVM, Logistic Regression)
- Hyperparameter tuning with GridSearchCV
- Cross-validation implementation
- Docker containerization
- Deploy to Streamlit Cloud / Hugging Face Spaces
- Add SHAP explainability plots

---

## ⚠️ Disclaimer

> **This project is built for educational and research purposes only.**
> The predictions made by this model are **NOT a substitute for professional medical diagnosis or advice**.
> Always consult a qualified healthcare professional for medical decisions.

---

## 📄 License

```text
MIT License

Copyright (c) 2024 Mahadiur Rehman

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files, to deal
in the Software without restriction...
```

See [LICENSE](https://github.com/mahadiurrehman-pixel/diabetes-ann-model/blob/main/LICENSE) for full details.

---

## 👨‍💻 Author

<div align="center">

**Mahadiur Rehman** — [GitHub](https://github.com/mahadiurrehman-pixel)

*"Building intelligent systems that make a difference"*

</div>

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

Made with ❤️ and lots of ☕

</div>
