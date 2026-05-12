import os
import mlflow
import json
import pandas as pd
from mlflow.models import validate_serving_input
# from mlflow.utils import convert_input_example_to_serving_input

# ⬇️ Sesuaikan tracking URI dengan yang Anda gunakan

# Hitung path absolut ke mlflow.db yang berada di root proyek (satu level di atas folder MLproject)
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlflow.db"))
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
# mlflow.set_tracking_uri("sqlite:///../mlflow.db")  # relatif dari folder MLproject ke root

# ⬇️ Ganti dengan run_id Anda
run_id = "a6f62bd0684d472d95e4be78cd577770"  # CONTOH, ganti dengan punya Anda
model_uri = f"runs:/{run_id}/model"

# --------------------- 1. Data contoh untuk validasi (sama persis dengan modul) ---------------------
data = {
    "columns": [
        "Age",
        "Credit_Mix",
        "Payment_of_Min_Amount",
        "Payment_Behaviour",
        "pc1_1",
        "pc1_2",
        "pc1_3",
        "pc1_4",
        "pc1_5",
        "pc2_1",
        "pc2_2"
    ],
    "data": [
        [0.7142857142857142, 1, 1, 3, -0.4381534490735855, 0.1711382783346808, 0.0773630019922211, -0.0401910461904993, 0.049590092121234, -0.1448249280763024, -0.0606673847105827],
        [0.4523809523809524, 2, 2, 5, 0.4778277065736656, -0.1050643177401745, -0.185971337955209, -0.3789896489656689, 0.1718128833126148, -0.2417658875286998, 0.0066502389514661],
        [0.4999999999999999, 3, 1, 1, -0.2172441359177029, 0.0068230171993729, 0.0319554863404481, -0.0402970285419156, 0.0861914654821804, 0.779882880094656, 0.1309092693530689],
        [0.4523809523809524, 1, 1, 6, -0.6893954147621222, 0.1842207645520866, 0.1887210184373613, -0.1923419820570049, 0.0561499495340574, 0.5840348959263311, 0.0673148184570155],
        [0.8333333333333333, 3, 0, 5, -0.287145204288708, -0.2462885871184559, 0.1247759677401836, -0.0544947505767573, 0.0941141473487672, 0.0526512725593595, -0.1827263807556981]
    ]
}

# --------------------- 2. Konversi ke format serving input (persis modul) ---------------------
# input_data = json.loads(convert_input_example_to_serving_input(data))   # dari modul
# input_data = {"dataframe_split": input_data["inputs"]}                  # dari modul (ganti key)

# Buat payload sesuai format serving input
payload = {"inputs": data}
input_data = json.loads(json.dumps(payload))    # menghasilkan dict yang sama
input_data = {"dataframe_split": input_data["inputs"]}

print("Input data structure for validation:")
print(json.dumps(input_data, indent=2))

# --------------------- 3. Validasi input terhadap model (persis modul) ---------------------
print("\n=== Validating serving input ===")
validation_result = validate_serving_input(model_uri, input_data)
print(validation_result)

# --------------------- 4. Prediksi langsung (persis modul) ---------------------
print("\n=== Loading model and predicting ===")
model = mlflow.pyfunc.load_model(model_uri)                             # dari modul
df = pd.DataFrame(
    input_data["dataframe_split"]["data"],
    columns=input_data["dataframe_split"]["columns"]
)                                                                       # dari modul
pred = model.predict(df)                                                # dari modul
print("Predictions:", pred)                                             # dari modul