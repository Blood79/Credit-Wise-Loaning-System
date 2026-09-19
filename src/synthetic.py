from __future__ import annotations
import numpy as np
import pandas as pd

def make_demo_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    credit = rng.integers(550, 851, n)
    income = rng.integers(20000, 160000, n)
    dti = np.round(rng.uniform(0.05, 0.75, n), 3)
    savings = rng.integers(1000, 200000, n)
    age = rng.integers(21, 66, n)
    employment = rng.choice(["Salaried", "Self-Employed", "Contract"], n)
    education = rng.choice(["Graduate", "Postgraduate", "High School"], n)
    purpose = rng.choice(["Home", "Education", "Vehicle", "Personal", "Business"], n)
    gender = rng.choice(["Male", "Female"], n)
    score = 0.008 * (credit - 550) + 0.000008 * income - 2.8 * dti + 0.000002 * savings + 0.015 * (age - 30)
    probability = 1 / (1 + np.exp(-(score - 1.9)))
    approved = rng.binomial(1, probability)
    return pd.DataFrame({"Applicant_ID": [f"DEMO{i:05d}" for i in range(n)], "Applicant_Income": income, "Coapplicant_Income": rng.integers(0, 80000, n), "Credit_Score": credit, "DTI_Ratio": dti, "Savings": savings, "Age": age, "Employment_Status": employment, "Education_Level": education, "Loan_Purpose": purpose, "Gender": gender, "Loan_Approved": approved})
