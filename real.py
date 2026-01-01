import streamlit as st
import pandas as pd
import joblib
from pandas.errors import ParserError


# Load model & scaler

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

if not hasattr(scaler, "feature_names_in_"):
    st.error("Scaler missing feature_names_in_. Retrain your scaler.")
    st.stop()

feature_cols = list(scaler.feature_names_in_)


# Default values (mean)

if hasattr(scaler, "mean_"):
    default_values = {col: float(mu) for col, mu in zip(feature_cols, scaler.mean_)}
else:
    default_values = {col: 0.0 for col in feature_cols}


# Important features (manual input)

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


# PAGE 1: Tester

def show_one_data_page():
    st.title("⟠ Cryptocurrency Fraud Detection (Random Forest)")
    st.subheader("Key in the data of Ethereum transaction")

    with st.form("single_tx_form"):
        cols = st.columns(2)
        user_inputs = {}

        for i, feat in enumerate(important_features):
            col = cols[i % 2]
            user_inputs[feat] = col.number_input(
                label=feat,
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.4f",
            )

        submitted = st.form_submit_button("🔍 Check")

    if submitted:
        data_dict = default_values.copy()

        for feat, val in user_inputs.items():
            if feat in data_dict:
                data_dict[feat] = float(val)

        X_single = pd.DataFrame([data_dict])[feature_cols]
        X_scaled = scaler.transform(X_single)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            p_legit, p_fraud = float(proba[0]), float(proba[1])
        else:
            pred_raw = model.predict(X_scaled)[0]
            p_fraud = 1.0 if pred_raw == 1 else 0.0
            p_legit = 1.0 - p_fraud

        label = "FRAUD" if p_fraud >= 0.50 else "LEGIT"

        st.markdown("---")
        st.subheader("Prediction Result")

        if label == "FRAUD":
            st.error(f"🔴 Prediction: {label}")
        else:
            st.success(f"🟢 Prediction: {label}")

        st.write(f"Probability → LEGIT: {p_legit:.3f}, FRAUD: {p_fraud:.3f}")


# PAGE 2: trning result

def show_file_data_page():
    st.title("⟠ Cryptocurrency Fraud Detection (Random Forest)")
    st.write("Upload transaction CSV file to detect fraud.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="latin1", engine="python")
        except ParserError as e:
            st.error(f"CSV read error: {e}")
            return

        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error("CSV missing required columns:")
            st.write(missing)
            return

        X = df[feature_cols]
        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled)
        df["Fraud_Prediction"] = preds
        df["Fraud_Label"] = df["Fraud_Prediction"].map({1: "FRAUD", 0: "LEGIT"})

        total_fraud = (df["Fraud_Label"] == "FRAUD").sum()
        total_legit = (df["Fraud_Label"] == "LEGIT").sum()

        st.write(f"🔴 Total FRAUD: {total_fraud}")
        st.write(f"🟢 Total LEGIT: {total_legit}")

        st.subheader("Prediction Results")
        st.dataframe(df)

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download CSV", csv_out, "fraud_prediction.csv")


# Sidebar Navigation

st.sidebar.title("Menu")
page = st.sidebar.radio("", ("Test Result", "Traning Result"))

if page == "Test Result":
    show_one_data_page()
elif page == "Training Result":
    show_file_data_page()
