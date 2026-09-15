# 💳 Credit Wise Loaning System

> A machine-learning notebook focused on exploring loan-approval data, preprocessing mixed features and preparing a supervised learning workflow for credit decision support.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 🎯 Objective

Explore the factors associated with loan approval and build a clean preprocessing and modelling workflow that can be extended into a credit-decision application.

## 🔎 Workflow

```mermaid
flowchart LR
    A[Loan Data] --> B[Data Inspection]
    B --> C[Missing Value Handling]
    C --> D[EDA]
    D --> E[Encoding]
    E --> F[Correlation & Feature Analysis]
    F --> G[Train / Test Split]
    G --> H[ML Model]
```

## 🧠 Work covered

- Dataset inspection and descriptive analysis
- Missing-value imputation for numerical and categorical features
- Exploratory analysis of approval patterns
- Outlier and distribution analysis
- Label encoding and one-hot encoding
- Correlation analysis
- Train/test preparation and feature engineering

## 🛠️ Stack

`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Scikit-learn` `Jupyter`

## 📁 Repository

- `Credit_Wise.ipynb` — complete exploratory and preprocessing notebook

## ▶️ Run locally

```bash
git clone https://github.com/Blood79/Credit-Wise-Loaning-System.git
cd Credit-Wise-Loaning-System
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Credit_Wise.ipynb
```

## 🚀 Next improvements

- Benchmark multiple classifiers with cross-validation
- Add precision/recall and ROC-AUC comparison
- Add explainability for approval decisions
- Wrap the best model in a FastAPI service
- Add a lightweight credit-decision dashboard

## 👨‍💻 Author

**Ayush Kumar Gupta** — [GitHub](https://github.com/Blood79) · [LinkedIn](https://linkedin.com/in/ayush-kumar-gupta-43314b238)
