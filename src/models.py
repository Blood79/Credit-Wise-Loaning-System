from __future__ import annotations
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from .preprocessing import build_transformer, prepare_frame
import pandas as pd

@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline

def build_model(name: str, X: pd.DataFrame) -> Pipeline:
    transformer = build_transformer(X)
    if name == "Logistic Regression":
        estimator: ClassifierMixin = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif name == "Random Forest":
        estimator = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    else:
        raise ValueError("Supported models: Logistic Regression, Random Forest")
    return Pipeline([("preprocess", transformer), ("model", estimator)])

def train_model(df: pd.DataFrame, model_name: str = "Logistic Regression") -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    X, y = prepare_frame(df)
    pipeline = build_model(model_name, X)
    pipeline.fit(X, y)
    return pipeline, X, y
