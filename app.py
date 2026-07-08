import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/credit_model.pkl")

st.set_page_config(page_title="Credit Scoring Model", page_icon="💳")

st.title("💳 Credit Scoring Prediction")
st.write("Enter the customer details below to predict creditworthiness.")

age = st.number_input("Age", 18, 100, 25)
income = st.number_input("Income", 10000, 500000, 50000)
loan = st.number_input("Loan Amount", 1000, 100000, 10000)
history = st.selectbox("Credit History", ["Good", "Bad"])
employment = st.number_input("Employment Years", 0, 40, 2)
debts = st.number_input("Existing Debts", 0, 20, 1)

history_value = 1 if history == "Good" else 0

if st.button("Predict"):

    data = pd.DataFrame({
        "Age":[age],
        "Income":[income],
        "LoanAmount":[loan],
        "CreditHistory":[history_value],
        "EmploymentYears":[employment],
        "ExistingDebts":[debts]
    })

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("✅ Good Credit")
    else:
        st.error("❌ Bad Credit")