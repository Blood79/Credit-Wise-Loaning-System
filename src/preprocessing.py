from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Loan_Approved"
DROP_COLUMNS = {"Applicant_ID"}

@dataclass(frozen=True)
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    transformer: ColumnTransformer

def prepare_frame(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")
    clean = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).copy()
    clean = clean.dropna(axis=0, subset=[target])
    y = clean.pop(target)
    if y.dtype == "object":
        y = y.astype(str).str.strip().str.lower().map({"y": 1, "yes": 1, "approved": 1, "n": 0, "no": 0, "rejected": 0}).fillna(pd.Series(y).astype("category").cat.codes)
    return clean, pd.Series(y, index=clean.index, name=target).astype(int)

def build_transformer(X: pd.DataFrame) -> ColumnTransformer:
    numerical = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, numerical), ("cat", cat_pipe, categorical)], remainder="drop")
