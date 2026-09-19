# 💳 Credit Wise Loaning System

> A recruiter-ready **credit decision-support ML prototype** built from the original exploratory notebook and refactored into a reusable preprocessing pipeline, model layer, evaluation module, Streamlit application, tests, Docker setup and CI/security automation.

**Author:** Ayush Kumar Gupta · [GitHub](https://github.com/Blood79) · [LinkedIn](https://linkedin.com/in/ayush-kumar-gupta-43314b238)

[![CI](https://github.com/Blood79/Credit-Wise-Loaning-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Blood79/Credit-Wise-Loaning-System/actions/workflows/ci.yml)
[![Security](https://github.com/Blood79/Credit-Wise-Loaning-System/actions/workflows/security.yml/badge.svg)](https://github.com/Blood79/Credit-Wise-Loaning-System/actions/workflows/security.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Project overview

Credit Wise demonstrates how a mixed-type loan dataset can move from raw tabular data to a reproducible machine-learning workflow.

The original notebook covered:

**Inspection → Missing-value handling → EDA → Encoding → Correlation analysis → Train/test split → Scaling → Logistic Regression / KNN / Naive Bayes → Feature engineering**

The refactored project extends that foundation with:

**Reusable preprocessing → model pipelines → evaluation metrics → synthetic demo data → interactive Streamlit UI → tests → Docker → CI → dependency auditing**

## Architecture

```text
                 ┌───────────────────────┐
                 │ CSV / Synthetic Demo  │
                 └───────────┬───────────┘
                             │
                  Validation + cleaning
                             │
               ┌─────────────▼─────────────┐
               │  ColumnTransformer        │
               │  Numeric + Categorical    │
               └─────────────┬─────────────┘
                             │
                    Model Pipeline
                    ┌────────┴────────┐
                    │                 │
             Logistic Regression   Random Forest
                    │                 │
                    └────────┬────────┘
                             │
                   Metrics + Confusion
                             │
                     Streamlit App
```

## Key engineering features

| Capability | Implementation |
|---|---|
| Mixed-type preprocessing | Median imputation + scaling for numeric columns; mode imputation + one-hot encoding for categorical columns |
| Leakage-resistant workflow | Preprocessing is fitted inside a scikit-learn pipeline on the training partition |
| Model comparison | Logistic Regression and Random Forest in the app; original notebook also contains KNN and Gaussian Naive Bayes |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC and confusion matrix |
| Demo without private data | Deterministic synthetic loan dataset generator |
| Interactive UI | Upload a CSV or run the synthetic demo directly in Streamlit |
| Reproducibility | Fixed random seeds, explicit dependencies and automated tests |
| Deployment | Dockerfile + Docker Compose |
| Quality gates | GitHub Actions CI + dependency audit |

## Run locally

### Clone

```bash
git clone https://github.com/Blood79/Credit-Wise-Loaning-System.git
cd Credit-Wise-Loaning-System
```

### Install

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Launch

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

### Run tests

```bash
pytest -q
```

### Docker

```bash
docker compose up --build
```

Then open port **8501**.

## Input data

The original notebook expects a file named `loan_approval_data.csv` with fields such as:

`Applicant_ID`, `Applicant_Income`, `Coapplicant_Income`, `Credit_Score`, `DTI_Ratio`, `Savings`, `Education_Level`, `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`, and `Loan_Approved`.

The refactored app also accepts a compatible CSV through the upload control.

## Repository layout

```text
Credit-Wise-Loaning-System/
├── app.py                         # Streamlit application
├── Credit_Wise.ipynb              # Original exploratory notebook
├── src/
│   ├── preprocessing.py           # Reusable preprocessing
│   ├── models.py                  # Model construction/training
│   ├── evaluation.py              # Metrics
│   └── synthetic.py               # Synthetic demo dataset
├── tests/
│   └── test_pipeline.py           # ML pipeline tests
├── docs/
│   ├── ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   └── MODEL_CARD.md
├── .github/workflows/
│   ├── ci.yml
│   └── security.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Responsible-use note

This repository is an educational **decision-support prototype**, not a production lending or underwriting system. Model metrics on one dataset or split do not establish suitability for real applicants.

A real deployment would require domain validation, data governance, fairness testing, privacy controls, model monitoring, human review, and applicable legal/regulatory processes. The application deliberately avoids presenting its output as a guaranteed credit decision.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Methodology](docs/METHODOLOGY.md)
- [Model Card](docs/MODEL_CARD.md)

## Original notebook

`Credit_Wise.ipynb` is preserved as the original exploratory artifact. The refactored code is intentionally separated so recruiters can inspect both the original analytical work and the engineering improvements.

## Roadmap

- Cross-validation benchmark report
- Explainability with SHAP
- Probability calibration and threshold analysis
- Dataset/data-dictionary validation
- Fairness and subgroup analysis
- FastAPI inference service
- Model/version tracking

## License

MIT © 2026 Ayush Kumar Gupta
