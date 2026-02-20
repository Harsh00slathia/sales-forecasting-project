import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Retail Profit Predictor",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("rf_pipeline.pkl")

# -----------------------------
# Title Section
# -----------------------------
st.title("📊 Retail Profit Prediction")
st.write("Predict whether a retail order will be **Profitable** or **Loss-Making**.")

st.divider()

# -----------------------------
# Input Section
# -----------------------------
sales = st.number_input("Sales", min_value=0.0)
discount = st.number_input("Discount", min_value=0.0, max_value=1.0)

category = st.selectbox(
    "Category",
    ["Technology", "Furniture", "Office Supplies"]
)

sub_category = st.selectbox(
    "Sub-Category",
    ["Chairs", "Tables", "Phones"]  # Make sure this matches training data
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🚀 Predict"):

    input_data = pd.DataFrame({
        "Sales": [sales],
        "Discount": [discount],
        "Category": [category],
        "Sub-Category": [sub_category]
    })

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    profit_prob = probability[0][1] * 100

    st.divider()

    if prediction[0] == 1:
        st.success("✅ This Order is Profitable")
    else:
        st.error("⚠️ This Order is Likely Loss-Making")

    st.metric("Profit Probability", f"{profit_prob:.2f}%")
