import streamlit as st
import pandas as pd
import json
from MLproject.preprocessAPI import data_preprocessing, prediction

st.set_page_config(page_title="Credit Scoring Prediction", layout="wide")
st.title("Form Input Data Kredit")

# Daftar kolom (18)
columns = [
    "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Age",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Outstanding_Debt", "Monthly_Inhand_Salary",
    "Monthly_Balance", "Amount_invested_monthly", "Total_EMI_per_month",
    "Credit_History_Age"
]

# Input kategorikal
credit_mix = st.selectbox("Credit Mix", ["Good", "Bad", "Standard"])
payment_min = st.selectbox("Payment of Min Amount", ["Yes", "No"])
payment_behavior = st.selectbox("Payment Behaviour", [
    "Low_spent_Small_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments",
    "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments"
])

# Input numerik (dengan default sesuai modul)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=23)
    num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, value=3)
    num_credit_card = st.number_input("Number of Credit Cards", min_value=0, value=4)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0, max_value=100, value=3)
    num_of_loan = st.number_input("Number of Loans", min_value=0, value=4)
    delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0, value=3)
    num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, value=7)
    changed_credit_limit = st.number_input("Changed Credit Limit (%)", value=11.27)
    num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=5)

with col2:
    outstanding_debt = st.number_input("Outstanding Debt ($)", value=809.98)
    monthly_inhand_salary = st.number_input("Monthly Inhand Salary ($)", value=1824.80)
    monthly_balance = st.number_input("Monthly Balance ($)", value=186.26)
    amount_invested_monthly = st.number_input("Amount Invested Monthly ($)", value=236.64)
    total_emi_per_month = st.number_input("Total EMI per Month ($)", value=49.50)
    credit_history_age = st.number_input("Credit History Age (months)", min_value=0, value=216)

# Tombol prediksi
if st.button("Prediksi"):
    # Gabungkan semua input menjadi list
    data_list = [
        credit_mix, payment_min, payment_behavior, age,
        num_bank_accounts, num_credit_card, interest_rate, num_of_loan,
        delay_from_due_date, num_of_delayed_payment, changed_credit_limit,
        num_credit_inquiries, outstanding_debt, monthly_inhand_salary,
        monthly_balance, amount_invested_monthly, total_emi_per_month,
        credit_history_age
    ]

    df = pd.DataFrame([data_list], columns=columns)
    new_data = data_preprocessing(data=df)

    # Konversi ke format JSON API
    json_output = {
        "dataframe_split": {
            "columns": new_data.columns.tolist(),
            "data": new_data.values.tolist()
        }
    }
    data_testing = json.dumps(json_output)

    result = prediction(data_testing)

    st.subheader("Hasil Prediksi")
    # Karena result mungkin berupa list angka, kita bisa memetakan ke label (opsional)
    label_map = {0: "Poor", 1: "Standard", 2: "Good"}
    if result and len(result) > 0:
        predicted_label = label_map.get(result[0], result[0])
        st.success(f"Credit Score: **{predicted_label}**")
    else:
        st.error("Gagal mendapatkan prediksi")