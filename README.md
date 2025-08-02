# Customer Churn Prediction

An end-to-end machine learning project designed to predict customer churn using historical customer data. This project showcases essential data science workflows including data cleaning, feature engineering, model training, evaluation, and interpretation.

---

## Table of Contents

* [Overview](#overview)
* [Project Goals](#project-goals)
* [Key Features](#key-features)
* [Dataset](#dataset)
* [Architecture](#architecture)
* [Approach](#approach)
* [Evaluation Metrics](#evaluation-metrics)
* [Project Setup](#project-setup)
* [Quickstart](#quickstart)
* [Example Workflow](#example-workflow)
* [Results](#results)
* [Extending the Project](#extending-the-project)
* [Folder Structure](#folder-structure)
* [Tech Stack](#tech-stack)

---

## Overview

Customer churn (also known as customer attrition) refers to when customers stop using a product or service. Predicting churn enables businesses to proactively engage customers who are at risk of leaving.

This project demonstrates:

* How to analyze customer data.
* Identify key factors that contribute to churn.
* Build machine learning models to predict churn probabilities.
* Provide actionable insights for retention strategies.

---

## Project Goals

* Perform **Exploratory Data Analysis (EDA)** to uncover patterns.
* Engineer meaningful features from raw data.
* Train multiple machine learning models (Logistic Regression, Random Forest, XGBoost).
* Optimize model performance with hyperparameter tuning.
* Interpret model results using **SHAP** values and feature importance.
* Provide a clear churn scoring pipeline that can be applied to new customer data.

---

## Key Features

* End-to-end ML workflow.
* Data preprocessing: handling missing values, encoding categorical variables, scaling.
* Class imbalance handling with **SMOTE** or class weighting.
* Cross-validation and hyperparameter tuning.
* Explainable ML using SHAP values.
* Model performance reporting (ROC-AUC, precision/recall, confusion matrix).

---

## Dataset

**Option 1:** Use the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn).
**Option 2:** Use your own business/customer dataset (if available).

The dataset includes columns like:

* Customer ID, Gender, Age, Contract Type
* MonthlyCharges, Tenure, TotalCharges
* Service options (Internet, Phone, Tech support)
* Churn (Yes/No)

---

## Architecture

1. **Data ingestion & cleaning** – Load dataset and prepare features.
2. **EDA** – Visualize churn patterns and correlations.
3. **Feature engineering** – Create derived variables (e.g., average monthly spend).
4. **Model training** – Logistic Regression, Random Forest, XGBoost.
5. **Evaluation** – Cross-validation, confusion matrix, ROC curve.
6. **Interpretation** – SHAP values for feature importance.
7. **Deployment-ready pipeline** – Save trained model using `joblib` or `pickle`.

---

## Approach

* Data preprocessing with `pandas` and `scikit-learn`.
* Train-test split.
* Baseline model (Logistic Regression) to set benchmark.
* Advanced models (Random Forest, XGBoost) for higher accuracy.
* Hyperparameter tuning with GridSearchCV or Optuna.
* Feature importance ranking.

---

## Evaluation Metrics

* **Accuracy**
* **Precision, Recall, F1-score**
* **ROC-AUC**
* **Confusion Matrix**

For imbalanced classes, emphasize precision, recall, and ROC-AUC.

---

## Project Setup

**Requirements:**

* Python 3.9+
* pandas, numpy
* scikit-learn
* xgboost, shap
* matplotlib, seaborn
* jupyterlab or notebook

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quickstart

1. Clone this repository.
2. Download and place the dataset in `data/`.
3. Open `notebooks/01-EDA.ipynb` for data exploration.
4. Run `notebooks/02-Modeling.ipynb` for model training and evaluation.

---

## Example Workflow

* Load data.
* Perform EDA (visualize churn vs. tenure, monthly charges).
* Train baseline model.
* Evaluate and tune models.
* Use SHAP to interpret why certain customers are likely to churn.

---

## Results

* Best model: ROC-AUC \~0.86.
* Key churn indicators: contract type, monthly charges, tenure.
* Actionable insight: Customers with month-to-month contracts and high monthly charges are at highest risk.

---

## Extending the Project

* Deploy a churn prediction API with FastAPI.
* Build a dashboard using Streamlit for interactive churn prediction.
* Integrate customer retention recommendations.

---

## Folder Structure

```
Customer-Churn-Prediction/
├── data/
|   ├── processed/
|       ├── telco_customer_churn_cleaned.csv
|   ├── raw/
|       ├── telco_customer_churn.csv
├── notebooks/
│   ├── 01-EDA.ipynb
│   └── 02-Modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── model_training.py
├── main.py
├── README.md
├── requirements.txt

```

---

## Tech Stack

* Python (pandas, numpy, scikit-learn, xgboost)
* Jupyter Notebook for analysis
* Matplotlib & Seaborn for visualization

---


