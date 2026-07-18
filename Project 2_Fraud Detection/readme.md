
---

# Credit Card Fraud Detection Pipeline: Linear vs. Ensemble Architectures

## 📌 Project Overview

This repository contains a leak-free, production-ready machine learning framework built to capture fraudulent credit card transactions under extreme class imbalance. Using a total pool of **284,807 transactions**, the dataset contains a microscopic fraud footprint where **99.83% of transactions are legitimate and only 0.17% are fraudulent**.

To maintain clean code separation and prevent computational runtime collisions, the experiment is divided into two distinct Kaggle notebook environments evaluating a linear baseline vs. a non-linear ensemble engine.

---

## 🛠️ Core Preprocessing & Zero-Leakage Architecture

### 1. Stratified Input Fidelity

Before any mathematical transformations or over-sampling protocols were initialized, the raw dataset was locked inside a **80/20 Stratified Train/Test Split**. Passing `stratify=y` guarantees that the `0.17%` target distribution curve is mirrored in both the training partitions and the final evaluation sets, establishing a realistic baseline test framework.

### 2. The Imblearn Assembly Line Loop

To completely eliminate the risk of **Data Leakage (Trap #2)**, all preprocessing transformations (Feature Scaling and SMOTE) were restricted to live inside an automated `imblearn.pipeline.Pipeline`.

* **The Leakage Shield:** When wrapped inside cross-validation loops via `GridSearchCV`, the pipeline isolates transformations so that **SMOTE and Scaling are computed fold-by-fold strictly on the 80% training splits**.
* **Purity Matrix:** The 20% validation splits and final test sets remain completely untouched, unresampled, and realistically imbalanced, ensuring our metrics reflect actual live financial conditions.

---

## 📦 Notebook Architecture Blueprints

### 📓 Notebook 1: The Linear Engine (Logistic Regression)

* **Pipeline Sequence:** `Raw Train Fold ➔ StandardScaler ➔ SMOTE() ➔ LogisticRegression()`
* **Engineering Logic:** Feature scaling via `StandardScaler` is mandatory. Because Logistic Regression relies on gradient descent optimization and applies regularization penalties ($L1$/$L2$), extreme raw range variances in transactional amounts would otherwise completely distort the coefficient weights.
* **Hyperparameter Grid Targets:** `smote__k_neighbors: [3, 5, 7]`, `classifier__C: [0.01, 0.1, 1.0, 10.0]`

### 📓 Notebook 2: The Ensemble Engine (Random Forest)

* **Pipeline Sequence:** `Raw Train Fold ➔ SMOTE() ➔ RandomForestClassifier()`
* **Engineering Logic:** `StandardScaler` is explicitly omitted. Tree-based ensembles split feature spaces ordinally via truth thresholds (e.g., $X_j > \text{val}$). They are mathematically scale-invariant. Dropping the scalar optimizes system memory and processing efficiency during intensive synthetic matrix processing.
* **Hyperparameter Grid Targets:** `smote__k_neighbors: [3, 5, 7]`, `classifier__max_depth: [10, 20, None]`, `classifier__n_estimators: [100, 200]`

---

## 📈 Empirical Performance Comparison & Analysis

Standard classification accuracy was completely discarded, as a naive model predicting "Legitimate" 100% of the time yields an empty 99.83% accuracy score while letting all financial loss slip through. Both models were judged under strict financial metrics on the untouched test split:

| Performance Metric | Notebook 1: Logistic Regression | Notebook 2: Random Forest | Strategic Operational Impact |
| --- | --- | --- | --- |
| **ROC-AUC Score** | 0.9719 | **0.9823** | Both architectures show stellar capacity to separate fraud vs. legal distribution profiles. |
| **Strict Recall** | **0.9184** | 0.8776 | Logistic Regression stops a fraction more direct financial loss (~4 additional fraud vectors captured). |
| **Strict Precision** | 0.0591 | **0.5478** | **Massive Random Forest Victory.** Drastically isolates false alarms. |
| **F1-Score (Fraud)** | 0.1100 | **0.6700** | The harmonic balance heavily favors the non-linear ensemble classifier. |

### 🔍 Deep-Dive Diagnostic Evaluation

1. **The Linear Boundary Constraint (Logistic Regression):** When SMOTE generated thousands of synthetic minority vectors, it expanded the fraud feature space. Because Logistic Regression is constrained to drawing a flat linear hyperplane, it was forced to aggressively shift its decision boundary deep into legitimate data territory. This successfully trapped fraud (High Recall) but accidentally misclassified thousands of innocent transactions, resulting in a **dramatic 5.9% Precision score**.
2. **The Non-Linear Ensemble Advantage (Random Forest):** The Ensemble engine built multiple complex, localized decision regions across its deep tree structures. Instead of compromising massive feature territory, it "carved out" precise pockets surrounding the fraud clusters. This kept the vast majority of legitimate user data unblocked (**Precision jumped to 54.78%**) while maintaining an excellent fraud detection rate.

---

## 🏆 Final Production Verdict

For an enterprise banking deployment, **Notebook 2 (Random Forest Engine) is the clear production candidate**.

While Logistic Regression caught slightly more fraud, operating a fraud alert engine with a **5.9% precision rate creates catastrophic customer friction**, forcing customer support centers to manage millions of false decline calls and ruining user experience. The Random Forest model strikes the ideal economic balance: it keeps direct fraud leakage to a minimum while maintaining stable operational overhead by ensuring more than half of all flagged system warnings are true fraud incidents.

---


### Installation

Clone this repository and install the verified dependency contract:

```bash
pip install -r requirements.txt

```