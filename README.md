# Bank Transaction Fraud Detection – Machine Learning Project

## Overview

This project explores a **fraud detection problem** using bank transaction data. The objective is **not** to make hard fraud decisions, but to build a **risk-oriented predictive model** that estimates the probability of a transaction being fraudulent, supporting decision-making under uncertainty.

The project covers the full data science pipeline:

* Exploratory Data Analysis (EDA)
* Feature engineering
* Handling strong class imbalance
* Model training and evaluation
* Feature importance and explainability (SHAP)
* Risk-based prediction

---

## Dataset

The dataset was obtained from Kaggle:
**Bank Transaction Fraud Detection**

Due to size and licensing constraints, the full dataset is not included in this repository. A representative sample is used for preview and reproducibility. The complete dataset can be accessed directly on Kaggle.

---

## Problem Characteristics

* Highly imbalanced dataset (~5% fraud)
* Fraud patterns are subtle and overlap strongly with non-fraud transactions
* Standard accuracy metrics are misleading
* Focus on **ROC-AUC**, **relative risk**, and **probabilistic outputs**

---

## Feature Engineering

Key feature engineering steps include:

* Removal of non-informative identifiers (UUIDs, IDs, timestamps)
* Transformation of repeated identifiers into frequency-based features
* Selection of financially meaningful numerical variables

Final core features used for modeling:

* Transaction Amount
* Account Balance
* Customer Age

---

## Exploratory Analysis

Several exploratory analyses were performed, including:

* Fraud distribution and class imbalance
* Fraud rates across bank branches
* Relative fraud risk by bank (normalized by global average)

<img width="590" height="390" alt="IMBALANCE" src="https://github.com/user-attachments/assets/c7516dd5-25c5-40f4-8482-7a921d708f20" />

<img width="1189" height="490" alt="RELATIVE RISK" src="https://github.com/user-attachments/assets/87778f05-a066-4913-9a57-1438be1aa01c" />

---

## Modeling Approach

A **Random Forest Classifier** was chosen due to:

* Non-linear decision boundaries
* Robustness to feature scaling
* Interpretability via feature importance and SHAP

Class imbalance was handled using:

* Stratified train-test split
* Class weighting (`class_weight='balanced'`)

Model evaluation focused on:

* ROC-AUC
* Feature ablation (AUC drop)

<img width="590" height="490" alt="ROC_CURVE" src="https://github.com/user-attachments/assets/78821c2d-10f1-472f-857f-06eed9922015" />


---

## Feature Importance & Ablation Study

Two complementary approaches were used:

### Impurity-Based Importance

Random Forest feature importances highlighted the dominance of numerical variables.

### AUC-Based Feature Ablation

Each feature was removed individually, retraining the model to measure the **real impact on predictive performance**.

This confirmed that:

* Transaction Amount
* Account Balance
* Age

are the most influential features in fraud prediction.

<img width="792" height="590" alt="FEATURES" src="https://github.com/user-attachments/assets/46b3f0ec-0f7e-4f30-ad55-7a890a494d52" />

---

## Model Explainability (SHAP)

SHAP values were used to analyze how features influence predictions at an individual level.

Key observations:

* Most SHAP values are concentrated around the base value
* The model rarely receives strong evidence to push predictions toward fraud
* This behavior is consistent with strong class imbalance and subtle fraud patterns

<img width="767" height="259" alt="SHAP" src="https://github.com/user-attachments/assets/cefa8d31-f54d-4425-a399-c5c0923e9e7d" />

---

## Risk-Oriented Prediction

Instead of outputting a binary decision, the final model produces a **fraud risk score**:

> Probability that a transaction is fraudulent

This allows:

* Flexible thresholding
* Integration with business rules
* Manual review prioritization

#Example usage:

```python
predict_fraud_risk(model, amount=50000, balance=20000, age=60)
```

---

## Limitations

* Dataset limited to a single country and month
* Fraud patterns may evolve over time
* Model performance constrained by data imbalance and feature overlap

---

## Future Work

* Cost-sensitive learning
* Time-based validation
* Threshold optimization based on business cost
* Integration with real-time alert systems

---

## Technologies Used

* Python
* Pandas / NumPy
* Scikit-learn
* SHAP
* Matplotlib / Seaborn

---

## Author

**Bastián B.**
BSc in Mathematics
Aspiring Data Analyst / Data Scientist
