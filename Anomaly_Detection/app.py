import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("isolation_forest_model.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Welfare Fraud Detection",
    page_icon="⚠️",
    layout="centered"
)

st.title("⚠️ Welfare Beneficiary Fraud Detection System")
st.write(
    "Enter complete beneficiary information. "
    "The system will compute anomaly indicators and predict **fraud risk**."
)

# =====================================================
# 🔹 INPUT: All 10 original dataset features
# =====================================================

st.header("🧾 Identity & Household Details")

aadhaar_like_id = st.text_input("Aadhaar-like ID (12 digits)")
household_id = st.text_input("Household ID")

st.header("💰 Financial Details")

annual_income = st.number_input(
    "Annual Income (₹)",
    min_value=0,
    max_value=300000,
    value=50000,
    step=1000
)

scheme_enrolled = st.selectbox(
    "Scheme Enrolled",
    ["Pension", "Scholarship", "MGNREGA", "PM-KISAN", "Food-Ration"]
)

st.header("📍 Location Details")

district = st.selectbox(
    "District",
    [f"District_{i}" for i in range(1, 51)]
)

st.header("🏦 Contact & Banking Details")

bank_account = st.text_input("Bank Account ID")
phone_number = st.text_input("Phone Number")

# =====================================================
# 🔹 ENGINEERED FEATURES (user-provided for demo realism)
# In real deployment → computed from database
# =====================================================

st.header("📊 Network & Duplication Indicators")

registrations_per_aadhaar = st.number_input(
    "Registrations per Aadhaar",
    min_value=1,
    max_value=10,
    value=1
)

bank_shared_count = st.number_input(
    "Bank Shared Count",
    min_value=1,
    max_value=15,
    value=1
)

phone_shared_count = st.number_input(
    "Phone Shared Count",
    min_value=1,
    max_value=15,
    value=1
)

# =====================================================
# 🔹 Prediction
# =====================================================

if st.button("🔍 Check Fraud Risk"):

    # Model uses only numeric anomaly features
    model_input = pd.DataFrame(
        [[
            annual_income,
            registrations_per_aadhaar,
            bank_shared_count,
            phone_shared_count
        ]],
        columns=[
            "annual_income",
            "registrations_per_aadhaar",
            "bank_shared_count",
            "phone_shared_count"
        ]
    )

    # Prediction
    score = -model.decision_function(model_input)[0]
    prediction = model.predict(model_input)[0]  # -1 anomaly, 1 normal

    # =================================================
    # 🔹 Result Display
    # =================================================
    st.subheader("📢 Prediction Result")

    if prediction == -1:
        st.error(f"⚠️ High Fraud Risk Detected\n\nAnomaly Score: **{score:.4f}**")

        st.write(
            """
            **Possible reasons:**
            - Duplicate Aadhaar registrations  
            - Shared bank account among many beneficiaries  
            - Shared phone number indicating collusion  
            - Income inconsistent with scheme eligibility  
            """
        )

    else:
        st.success(f"✅ Beneficiary Appears Normal\n\nAnomaly Score: **{score:.4f}**")

        st.write(
            "No strong anomaly patterns detected based on the trained AI model."
        )

# =====================================================
# Footer
# =====================================================

st.markdown("---")
st.caption("AI-Powered Welfare Fraud Detection • Hackathon Demo • Streamlit App")
