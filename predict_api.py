import requests
import json
import pandas as pd
from MLproject.preprocessAPI import data_preprocessing

url = "http://127.0.0.1:5004/invocations"
headers = {"Content-Type": "application/json"}

# Data mentah contoh (18 kolom, seperti modul)
columns = [
    "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Age",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Outstanding_Debt", "Monthly_Inhand_Salary",
    "Monthly_Balance", "Amount_invested_monthly", "Total_EMI_per_month",
    "Credit_History_Age"
]

data_list = ["Good", "No", "Low_spent_Small_value_payments", 23, 3, 4, 3, 4,
             3, 7, 11.27, 5, 809.98, 1824.80, 186.26, 236.64, 49.50, 216]

df = pd.DataFrame([data_list], columns=columns)
processed_df = data_preprocessing(data=df)

# Konversi ke format dataframe_split
json_output = {
    "dataframe_split": {
        "columns": processed_df.columns.tolist(),
        "data": processed_df.values.tolist()
    }
}
data_testing = json.dumps(json_output)

response = requests.post(url, data=data_testing, headers=headers)
result = response.json()
print("Response:", result)
print("Predictions:", result.get("predictions"))