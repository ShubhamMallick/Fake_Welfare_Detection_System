# ==========================================
# Duplicate Detection Streamlit App
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------
# MUST be the first Streamlit command
# ------------------------------------------
st.set_page_config(
    page_title="Duplicate Beneficiary Detection",
    page_icon="🔍",
    layout="centered"
)

# ------------------------------------------
# Load dataset and trained pipeline
# ------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("duplicate_detection_50000_v4.csv")


@st.cache_resource
def load_pipeline():
    return joblib.load("duplicate_detection_pipeline_xgb.pkl")


df = load_data()
pipeline = load_pipeline()

# ------------------------------------------
# Title
# ------------------------------------------
st.title("🔍 Duplicate Beneficiary Detection System")
st.write(
    "AI system for detecting **duplicate welfare beneficiaries** "
    "using network linkage features and **XGBoost prediction**."
)

st.markdown("---")

# ==========================================
# 🧪 Demo Buttons (session-state)
# ==========================================
st.subheader("🧪 Quick Demo")

if "demo_row" not in st.session_state:
    st.session_state.demo_row = None

col1, col2 = st.columns(2)

with col1:
    if st.button("Load Normal Example"):
        st.session_state.demo_row = df[df["is_duplicate"] == 0].sample(1)

with col2:
    if st.button("Load Duplicate Example"):
        st.session_state.demo_row = df[df["is_duplicate"] == 1].sample(1)

# Row to display
if st.session_state.demo_row is not None:
    row = st.session_state.demo_row.iloc[0]
else:
    row = df.sample(1).iloc[0]

# ==========================================
# 📝 Manual Input
# ==========================================
st.subheader("📝 Enter Beneficiary Details")

aadhaar = st.text_input("Aadhaar ID", value=row["aadhaar_like_id"])
name = st.text_input("Name", value=row["name"])
household = st.text_input("Household ID", value=row["household_id"])
phone = st.text_input("Phone Number", value=row["phone_number"])
bank = st.text_input("Bank Account", value=row["bank_account"])

district_list = sorted(df["district"].unique())
district = st.selectbox(
    "District",
    district_list,
    index=district_list.index(row["district"])
)

# ==========================================
# 🔢 Real-time linkage feature calculation
# ==========================================
# Convert to strings for consistent comparison
aadhaar_str = str(aadhaar)
phone_str = str(phone)
bank_str = str(bank)
household_str = str(household)

aadhaar_count = df[df["aadhaar_like_id"].astype(str) == aadhaar_str].shape[0]
phone_count = df[df["phone_number"].astype(str) == phone_str].shape[0]
bank_count = df[df["bank_account"].astype(str) == bank_str].shape[0]
household_size = df[df["household_id"].astype(str) == household_str].shape[0]

st.markdown("### 📊 Linkage Statistics")

st.write({
    "aadhaar_count": aadhaar_count,
    "phone_count": phone_count,
    "bank_count": bank_count,
    "household_size": household_size
})

# ------------------------------------------
# Optional debug panel
# ------------------------------------------
with st.expander("🔧 Debug: Model Input Features"):
    st.write({
        "aadhaar": aadhaar_str,
        "phone": phone_str,
        "bank": bank_str,
        "household": household_str,
        "counts": [aadhaar_count, phone_count, bank_count, household_size]
    })

# ==========================================
# 🔮 Prediction
# ==========================================
if st.button("🔍 Predict Duplicate Risk"):

    # Ensure consistent data types with training
    input_df = pd.DataFrame([{
        "aadhaar_like_id": aadhaar_str,
        "name": str(name),
        "household_id": household_str,
        "phone_number": phone_str,
        "bank_account": bank_str,
        "district": str(district),
        "aadhaar_count": float(aadhaar_count),
        "phone_count": float(phone_count),
        "bank_count": float(bank_count),
        "household_size": float(household_size)
    }])

    # ---- Correct inference using full pipeline ----
    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    duplicate_prob = prob * 100
    normal_prob = (1 - prob) * 100

    st.markdown("---")
    st.subheader("📢 Prediction Result")

    if pred == 1:
        st.error(
            f"⚠️ Duplicate Detected\n\n"
            f"Duplicate Risk: **{duplicate_prob:.2f}%**"
        )

        st.write("### Possible Reasons")
        st.write("- Shared phone or bank across many beneficiaries")
        st.write("- Same Aadhaar appearing multiple times")
        st.write("- Large household cluster")
        st.write("- Cross-district duplication pattern")

    else:
        st.success(
            f"✅ Appears Genuine\n\n"
            f"Confidence Genuine: **{normal_prob:.2f}%**"
        )

# ==========================================
#  Related Records Viewer
# ==========================================
st.markdown("---")
st.subheader("📂 Matching Records in Dataset")

if st.checkbox("Show related beneficiaries"):

    related = df[
        (df["aadhaar_like_id"].astype(str) == aadhaar_str) |
        (df["phone_number"].astype(str) == phone_str) |
        (df["bank_account"].astype(str) == bank_str) |
        (df["household_id"].astype(str) == household_str)
    ]

    st.dataframe(related.head(50))

# ==========================================
# Footer
# ==========================================
st.markdown("---")
st.caption("AI-Powered Duplicate Detection • XGBoost • Based on Training Notebook")
