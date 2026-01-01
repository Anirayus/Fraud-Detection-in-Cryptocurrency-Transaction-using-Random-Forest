# fraud_8features_app.py

import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model & scaler
# =========================
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# Ensure feature names exist
if not hasattr(scaler, "feature_names_in_"):
    st.error(
        "⚠️ scaler.pkl does not contain 'feature_names_in_'. "
        "Please retrain and resave the scaler using a newer sklearn version."
    )
    st.stop()

feature_cols = list(scaler.feature_names_in_)

# =========================
# DEFAULT VALUES (MEAN)
# =========================
if hasattr(scaler, "mean_"):
    default_values = {col: float(mu) for col, mu in zip(feature_cols, scaler.mean_)}
else:
    default_values = {col: 0.0 for col in feature_cols}

# =========================
# 8 IMPORTANT FEATURES (WITH ERC20 & MAX VALUE)
# =========================
important_features = [
    "ERC20 uniq rec token name",
    "Time Diff between first and last (Mins)",
    "avg val received",
    "Avg min between received tnx",
    "total Ether sent",
    "max value received",
    "Received Tnx",
    "Sent tnx",
]

# =========================
# UI
# =========================
st.title("⟠ Cryptocurrency Fraud Detection (Random Forest)")
st.subheader("Key in the data of Ethereum transaction")

with st.form("single_tx_form"):
    cols = st.columns(2)
    user_inputs = {}

    for i, feat in enumerate(important_features):
        col = cols[i % 2]
        default_val = 0.0

        user_inputs[feat] = col.number_input(
            label=feat,
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.4f",
        )

    submitted = st.form_submit_button("🔍 Check")

# =========================
# PREDICTION (NO SETTINGS)
# =========================
if submitted:
    # Start with mean values
    data_dict = default_values.copy()

    # Override with user inputs
    for feat, val in user_inputs.items():
        if feat in data_dict:
            data_dict[feat] = float(val)

    X_single = pd.DataFrame([data_dict])[feature_cols]
    X_scaled = scaler.transform(X_single)

    # Probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        p_legit = float(proba[0])
        p_fraud = float(proba[1])
    else:
        pred_raw = model.predict(X_scaled)[0]
        p_fraud = 1.0 if pred_raw == 1 else 0.0
        p_legit = 1.0 - p_fraud

    # Fixed threshold = 0.50
    label = "FRAUD" if p_fraud >= 0.50 else "LEGIT"

    st.markdown("---")
    st.subheader("Prediction Result")

    if label == "FRAUD":
        st.error(f"🔴 Prediction: **{label}**")
    else:
        st.success(f"🟢 Prediction: **{label}**")

    st.write(
        f"Probability → LEGIT: **{p_legit:.3f}**, FRAUD: **{p_fraud:.3f}**"
    )

