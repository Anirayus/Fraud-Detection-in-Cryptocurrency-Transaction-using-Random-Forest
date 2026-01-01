import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pandas.errors import ParserError

# Load trained model and scaler
model = joblib.load("rf_model.pkl")

try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None

st.title("⟠ Cryptocurrency Fraud Detection (Random Forest)")
st.write("Upload transaction CSV file to detect fraud.")

# Upload File
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(
    uploaded_file,
    encoding="latin1",
    engine="python",
)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # ============================
    # 1) Guna feature yang scaler expect
    # ============================
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        feature_cols = list(scaler.feature_names_in_)
    else:
        st.error("Scaler tidak mempunyai 'feature_names_in_'. Pastikan scaler.pkl dihasilkan dengan sklearn versi baru.")
        st.stop()


    # ============================
    # 2) Bina X ikut EXACT order features yang scaler expect
    # ============================
    X = df[feature_cols]

    # ============================
    # 3) Scale guna scaler yang sama
    # ============================
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X  # fallback, tapi jarang perlu

    # ============================
    # 4) Prediction
    # ============================
    preds = model.predict(X_scaled)

    df["Fraud_Prediction"] = preds
    df["Fraud_Label"] = df["Fraud_Prediction"].map({1: "FRAUD", 0: "LEGIT"})
    
    total_fraud = int((df["Fraud_Label"] == "FRAUD").sum())
    total_legit = int((df["Fraud_Label"] == "LEGIT").sum())

    st.write(f"🔴 Total FRAUD: **{total_fraud}**")
    st.write(f"🟢 Total LEGIT: **{total_legit}**")

    cols = ["Fraud_Prediction", "Fraud_Label"] + [c for c in df.columns if c not in ["Fraud_Prediction", "Fraud_Label"]]
    df = df[cols]
    
    st.subheader("Prediction Results")
    st.dataframe(df)

    # ============================
    # 5) Download result
    # ============================
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="💾 Download",
        data=csv,
        file_name="fraud_prediction.csv",
        mime="text/csv"
    )

     # Debug: tengok apa scaler expect & apa CSV ada
    st.write("✅ Features expected by model/scaler:")
    st.write(feature_cols)

    st.write("✅ Columns in uploaded CSV:")
    st.write(list(df.columns))

    # Check missing columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error("❌ Ada column yang model expect tapi tak jumpa dalam CSV yang di-upload.")
        st.write("Missing columns:", missing)
        st.stop()