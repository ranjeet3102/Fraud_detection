import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- Setup ------------------- #
st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")
st.title("🚨 Insurance Fraud Detection Dashboard")

# Resolve model path dynamically
BASE_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(BASE_DIR, "..", "models", "xgb_pipeline.pkl")

REQUIRED_FEATURES = ['age', 'policy_sales_channel', 'gender', 'previously_insured',
                     'vehicle_age', 'vehicle_damage']

@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)

pipeline = load_pipeline()

# ---------------- Sidebar ---------------- #
mode = st.sidebar.radio("Select Input Mode", ["📤 Upload CSV", "✍️ Manual Entry"])

# ---------------- CSV Upload ---------------- #
if mode == "📤 Upload CSV":
    st.subheader("Upload a CSV for Fraud Prediction")
    csv_file = st.file_uploader("Upload your file", type=["csv"])

    if csv_file:
        df = pd.read_csv(csv_file)
        st.write("✅ Uploaded data preview:")
        st.dataframe(df.head())

        if all(col in df.columns for col in REQUIRED_FEATURES):
            predictions = pipeline.predict(df[REQUIRED_FEATURES])
            probabilities = pipeline.predict_proba(df[REQUIRED_FEATURES])[:, 1]

            df['is_fraud_prediction'] = predictions
            df['fraud_probability'] = probabilities

            st.success("✔️ Predictions complete")
            st.write(df[['is_fraud_prediction', 'fraud_probability']])

            # Pie Chart
            st.subheader("🔎 Fraud Prediction Summary")
            fig, ax = plt.subplots()
            df['is_fraud_prediction'].value_counts().plot.pie(
                autopct='%1.1f%%',
                labels=["Not Fraud", "Fraud"],
                colors=["lightgreen", "salmon"],
                ax=ax
            )
            ax.set_ylabel("")
            st.pyplot(fig)

            # Download
            csv_out = df.to_csv(index=False).encode('utf-8')
            st.download_button("📅 Download Results", data=csv_out, file_name="fraud_predictions.csv")
        else:
            st.warning(f"❌ Missing columns. Expected: {REQUIRED_FEATURES}")

# ---------------- Manual Input ---------------- #
else:
    st.subheader("Enter Customer Data Manually")

    input_data = {
        'age': st.number_input("Age", min_value=18, max_value=100, value=30),
        'policy_sales_channel': st.number_input("Policy Sales Channel", min_value=0, max_value=200, value=26),
        'gender': st.selectbox("Gender (0=Female, 1=Male)", [0, 1]),
        'previously_insured': st.selectbox("Previously Insured", [0, 1]),
        'vehicle_age': st.selectbox("Vehicle Age (0=<1yr, 1=1-2yr, 2=>2yr)", [0, 1, 2]),
        'vehicle_damage': st.selectbox("Vehicle Damage (0=No, 1=Yes)", [0, 1])
    }

    input_df = pd.DataFrame([input_data])

    if st.button("🔍 Predict"):
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0, 1]
        st.success(f"Prediction: {'Fraud' if prediction == 1 else 'Not Fraud'}")
        st.info(f"Probability of fraud: {probability:.2f}")
