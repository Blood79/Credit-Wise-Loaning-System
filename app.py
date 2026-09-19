from __future__ import annotations
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from src.evaluation import evaluate
from src.models import build_model
from src.preprocessing import prepare_frame
from src.synthetic import make_demo_data

st.set_page_config(page_title="Credit Wise", page_icon="💳", layout="wide")
st.title("💳 Credit Wise Loaning System")
st.caption("Explainable credit-decision support prototype — not an autonomous lending decision system.")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload loan CSV", type=["csv"])
    use_demo = st.toggle("Use synthetic demo data", value=not uploaded)
    model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

df = pd.read_csv(uploaded) if uploaded else make_demo_data()
X, y = prepare_frame(df)

m1, m2, m3 = st.columns(3)
m1.metric("Rows", len(df)); m2.metric("Features", X.shape[1]); m3.metric("Approval rate", f"{100*y.mean():.1f}%")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
with right:
    st.subheader("Target distribution")
    st.bar_chart(y.value_counts().sort_index().rename(index={0:"Rejected",1:"Approved"}))

if st.button("Train and evaluate", type="primary"):
    if y.nunique() < 2: st.error("The target must contain both classes."); st.stop()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_model(model_name, X_train)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline, "predict_proba") else None
    metrics = evaluate(y_test, pred, prob)
    st.subheader("Evaluation")
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy","precision","recall","f1","roc_auc"]): col.metric(key.replace("_"," ").title(), metrics.get(key, "n/a"))
    st.write("Confusion matrix"); st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]), use_container_width=True)
    st.info("Metrics describe this dataset/split only. Credit decisions require domain validation, fairness testing, monitoring, and human oversight.")

st.divider()
st.subheader("Project scope")
st.markdown("This portfolio project demonstrates preprocessing, mixed-type feature handling, supervised classification, evaluation, and a deployable UI. It intentionally avoids representing a model score as a guaranteed lending decision.")
