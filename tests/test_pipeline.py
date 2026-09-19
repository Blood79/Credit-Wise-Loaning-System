import pandas as pd
from src.preprocessing import prepare_frame, build_transformer
from src.models import build_model
from src.synthetic import make_demo_data

def test_demo_data_shape_and_target():
    df = make_demo_data(80)
    X, y = prepare_frame(df)
    assert len(X) == 80 and y.nunique() == 2
    assert "Applicant_ID" not in X.columns

def test_pipeline_trains_on_mixed_features():
    df = make_demo_data(120)
    X, y = prepare_frame(df)
    model = build_model("Logistic Regression", X)
    model.fit(X, y)
    assert len(model.predict(X)) == len(y)
