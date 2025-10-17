# 🏥 Hospital Readmission Prediction

## 🎯 Project Overview
This repository implements a machine learning solution to predict whether a patient will be readmitted to the hospital within 30 days.  
It leverages patient demographics, clinical data, diagnostics, medications, and hospital utilization features.  
The goal is to help hospitals identify **high-risk patients** and take **preventive actions** to reduce readmission rates.

### Key Objectives
- Understand and preprocess raw data (handle missing values, encoding, scaling).  
- Build and compare multiple classification models — Logistic Regression, Random Forest, and XGBoost.  
- Use a **modular and reproducible pipeline** structure.  
- Experiment with a **GPU-enabled SVM-style model** for comparison.  
- Interpret results, visualize metrics, and identify key predictive features.

---

## 📂 Repository Structure
  e:\Hospital_Readmission_Prediction\
  ├── data\
  │   ├── raw\                       # original CSVs (diabetic_data.csv)
  │   └── processed\                 # cleaned / feature-engineered CSVs (preprocessed_data.csv)
  ├── notebooks\
  │   ├── 01_data_exploration.ipynb
  │   ├── 02_data_cleaning.ipynb
  │   ├── 03_experiments.ipynb
  │   └── 04_model_training.ipynb
  ├── src\
  │   ├── __init__.py
  │   ├── data_preprocessing.py
  │   ├── feature_engineering.py
  │   ├── model_training.py
  │   ├── predict.py
  │   ├── evaluation.py
  │   ├── utils\
  │   │   ├── __init__.py
  │   │   ├── data_loader.py
  │   │   ├── train_pipeline.py
  │   │   ├── evaluater.py
  │   │   ├── visualizer.py
  │   │   ├── model_saver.py
  │   │   └── other_helpers.py
  │   └── preprocess_pipeline.py
  ├── models\
  │   ├── logistic_regression_model.pkl
  │   ├── random_forest_model.pkl
  |   ├── decision_tree_model.pkl
  │   ├── random_forest_model.pkl
  │   ├── tuned_random_forest_model.pkl
  |   ├── svm_model.pth
  │   ├── xgb_model.pkl
  │   └── tuned_XGB_model.pkl
  ├── reports\
  │   └── model_report.pdf
  |
  ├── requirements.txt
  ├── README.md
  ├── .gitignore


# 🧭 Key Modules & Scripts

### `src/utils/data_loader.py`
- Contains `load_data()`, `split_X_y()`, and `train_test_split_data()`
- Handles dataset ingestion and basic splitting logic

### `src/utils/evaluater.py`
- Contains `evaluate_model()` that computes metrics:
  - Accuracy  
  - F1 Score  
  - ROC-AUC  
  - Confusion Matrix  
- Optionally supports probability-based metrics

---

## 🧩 Pipeline & Training Logic
- **`build_pipeline(model, model_type, use_smote)`** → constructs an sklearn or imbalanced-learn pipeline including scaler, optional SMOTE, and the model.  
- **`train_pipeline(data_path, model, …)`** → executes the full workflow:  



---

## 🧼 Data Preprocessing Steps (Summary)

### 🔹 Missing Value Handling
- Replace `'?'` values with `NaN`
- Drop irrelevant or highly missing columns:
- **Identifier columns:** `encounter_id`, `patient_nbr`  
- **80% missing:** `weight`  
- **Constant-value columns:** `examide`, `citoglipton`
- Retain `max_glu_serum`, `A1Cresult` (impute with `'None'`)

### 🔹 Imputation
- Categorical: replace with mode or `'Unknown'` / `'None'`
- Diagnosis-related columns: replace missing with `'Unknown'`

### 🔹 Encoding
- Binary target encoding for `readmitted`
- Label encoding for nominal categorical features (`gender`, `race`, etc.)
- Ordinal encoding for medication-related columns

### 🔹 Outlier Treatment
- Apply **log transformation** on skewed numerical columns

### 🔹 Scaling
- **RobustScaler** → for tree-based models (Random Forest, XGBoost)  
- **StandardScaler** → for non-tree models (Logistic Regression, SVM)

### 🔹 Feature Engineering
- Bin `age` into categorical groups  
- Categorize diagnostic codes  
- Derive new features:
- `total_medications`
- `num_med_changes`

---

## 📈 Modeling & Experiments

### 🔸 Pipeline Flow
`Preprocessing → Train/Test Split → Optional SMOTE → Scaling → Model Training → Evaluation`

### 🔸 Models Tested
- Logistic Regression  
- Random Forest (tuned)  
- XGBoost (tuned)  
- Custom GPU-based SVM-style model

### 🔸 Notes
- Hyperparameter tuning for ensemble models (**Random Forest**, **XGBoost**) was **computationally intensive**.  
- Visual analysis with:
- **ROC Curves**
- **Confusion Matrix Heatmaps**
- Custom `src/utils` modules ensure **reproducibility** and **experiment consistency**.

---

## 🧠 Author
**Kuldeep Yadav**   
