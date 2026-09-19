# Model card

## Intended use

Educational and portfolio demonstration of a supervised ML workflow for loan-approval prediction.

## Out of scope

This repository must not be presented as an automated lending, credit underwriting, eligibility, or adverse-action system.

## Models

- Logistic Regression with class balancing
- Random Forest with class balancing

## Data

The repository provides a **synthetic demo generator** so the application can run without shipping real people's financial data. Uploaded datasets are supported for experimentation.

## Evaluation

The app reports accuracy, precision, recall, F1 and ROC-AUC on the selected train/test split. No benchmark is claimed for an external population.

## Limitations

Performance depends on data quality, label definition, sampling, preprocessing choices, thresholding and population shift. Any real deployment would need domain validation, legal review, fairness analysis, monitoring, human oversight and an appropriate explanation process.
