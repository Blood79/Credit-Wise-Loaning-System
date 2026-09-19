# Methodology

## Source notebook

The original `Credit_Wise.ipynb` performs data inspection, mean/mode imputation, EDA, label encoding, one-hot encoding, correlation analysis, an 80/20 train-test split, standardization, and comparisons of Logistic Regression, KNN, and Gaussian Naive Bayes.

## Refactored workflow

1. Validate the target column.
2. Remove identifier columns such as `Applicant_ID`.
3. Split numeric and categorical features.
4. Impute numeric values with the median and categorical values with the most frequent value.
5. One-hot encode categorical features with unknown-category handling.
6. Standardize numeric features inside the training pipeline.
7. Train a selectable classifier.
8. Report accuracy, precision, recall, F1 and ROC-AUC where available.
9. Keep evaluation tied to the selected dataset and split.

The refactor keeps preprocessing inside a scikit-learn Pipeline/ColumnTransformer so fitting is learned from the training partition when the app evaluates a model.
