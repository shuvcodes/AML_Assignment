# Assignment 02: Experiment Tracking

**Course:** Applied Machine Learning  
**Student Name:** Arnab Bera  
**Submission Date:** 15 Feb 2026  

---

# 1. Overview

This assignment demonstrates:

- Data Version Control using DVC  
- Model Version Control and Experiment Tracking using MLflow  
- Reproducible data splits  
- Comparison of benchmark models using AUCPR  

Dataset used: SMS Spam Classification Dataset (SMSSpamCollection)

---

# 2. Project Structure

```
Assignment-02/
│
├── dataset/
│   ├── raw_data.csv
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│
├── prepare.ipynb
├── train.ipynb
│
├── dvc_storage/
├── .dvc/
├── README.md
```

---

# 3. Part 1: Data Version Control (DVC)

## 3.1 Objective

The objective of this part is to:

- Track dataset versions using DVC  
- Ensure reproducibility of data splits  
- Demonstrate version checkout  

---

## 3.2 Dataset Preparation

Original dataset: `SMSSpamCollection`  
The dataset was converted to `raw_data.csv`.

Label mapping:
- ham → 0  
- spam → 1  

---

## 3.3 Data Splitting

Split ratio:
- 70% Train  
- 15% Validation  
- 15% Test  

Stratified splitting was used to preserve label distribution.

---

## 3.4 Version 1 (Initial Split)

Random Seed: 42  

Files tracked using DVC:

```
dvc add dataset/raw_data.csv
dvc add dataset/train.csv
dvc add dataset/validation.csv
dvc add dataset/test.csv
```

Committed using:

```
git commit -m "Version 1 split seed 42"
dvc push
```

---

## 3.5 Version 2 (Updated Split)

Random Seed changed to: 123  

Dataset split regenerated and tracked using DVC:

```
dvc add dataset/train.csv
dvc add dataset/validation.csv
dvc add dataset/test.csv
```

Committed using:

```
git commit -m "Version 2 split seed 123"
dvc push
```

---

## 3.6 Version Checkout

To restore Version 1:

```
git checkout <commit_id>
dvc checkout
```

To restore the latest version:

```
git checkout main
dvc checkout
```

The distribution of the target variable (number of 0s and 1s) was printed for train, validation, and test sets for both versions.

This demonstrates reproducibility and proper data version control.

---

# 4. Part 2: Model Version Control and Experiment Tracking (MLflow)

## 4.1 Objective

The objective of this part is to:

- Track experiments using MLflow  
- Log model parameters and evaluation metrics  
- Register multiple benchmark models  
- Compare models using AUCPR  

---

## 4.2 MLflow Setup

Experiment name:

```
Assignment_02_SMS_Spam
```

```

---

## 4.3 Text Preprocessing

- TF-IDF Vectorization  
- English stop words removed  
- Vectorizer fitted on training data  

---

## 4.4 Benchmark Models

Three benchmark models were trained and tracked:

### 1. Logistic Regression
- Parameters logged  
- AUCPR metric logged  
- Model artifact saved in MLflow  

### 2. Multinomial Naive Bayes
- Parameters logged  
- AUCPR metric logged  
- Model artifact saved in MLflow  

### 3. Linear Support Vector Machine
- Parameters logged  
- AUCPR metric logged  
- Model artifact saved in MLflow  

---

# 5. Evaluation Metric

Evaluation metric used:

Area Under Precision-Recall Curve (AUCPR)

Reason:
The dataset is slightly imbalanced, and AUCPR is more appropriate than accuracy for evaluating classification performance in such cases.

---

# 6. Results

All three models were tracked in MLflow.  

For each run:
- Model parameters were logged  
- AUCPR was recorded  
- Model artifacts were saved  
- Experiments were visible in MLflow UI  

The best model can be selected based on the highest AUCPR value.

---

# 7. Key Learnings

- DVC enables reproducible and versioned data management.  
- Git tracks metadata while DVC tracks large data files.  
- MLflow provides structured experiment tracking.  
- Model comparison becomes systematic using logged metrics.  
- Reproducibility is essential in machine learning workflows.  

---

# 8. Conclusion

This assignment successfully demonstrates:

- Data versioning using DVC  
- Decoupling of data and code  
- Experiment tracking using MLflow  
- Training and comparison of multiple benchmark models  
- Reproducible machine learning workflow  

The project ensures full traceability of both data versions and model experiments.
