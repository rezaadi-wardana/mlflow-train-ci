import pandas as pd
import requests
import json

def data_preprocessing(data):
    """
    Pipeline tiruan. Akan diganti dengan pipeline asli nanti.
    """
    # Mapping kategori (contoh)
    credit_mix_map = {"Good": 1, "Standard": 2, "Bad": 3}
    payment_min_map = {"Yes": 1, "No": 0}
    payment_beh_map = {
        "Low_spent_Small_value_payments": 1,
        "Low_spent_Medium_value_payments": 2,
        "Low_spent_Large_value_payments": 3,
        "High_spent_Small_value_payments": 4,
        "High_spent_Medium_value_payments": 5,
        "High_spent_Large_value_payments": 6
    }

    data["Credit_Mix"] = data["Credit_Mix"].map(credit_mix_map)
    data["Payment_of_Min_Amount"] = data["Payment_of_Min_Amount"].map(payment_min_map)
    data["Payment_Behaviour"] = data["Payment_Behaviour"].map(payment_beh_map)

    # Buat kolom PCA tiruan (11 kolom sesuai model)
    # Di kehidupan nyata, gunakan pipeline PCA yang sesungguhnya.
    pca_cols = ["pc1_1", "pc1_2", "pc1_3", "pc1_4", "pc1_5", "pc2_1", "pc2_2"]
    for i, col in enumerate(pca_cols):
        # Isi dengan 0.0 (hanya untuk demonstrasi tampilan)
        data[col] = 0.0

    # Pilih hanya kolom yang sesuai signature model
    required_columns = [
        "Age", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
        "pc1_1", "pc1_2", "pc1_3", "pc1_4", "pc1_5", "pc2_1", "pc2_2"
    ]
    return data[required_columns]


def prediction(data_json):
    """Kirim data ke API dan kembalikan hasil prediksi."""
    url = "http://127.0.0.1:5002/invocations"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=data_json, headers=headers)
    result = response.json().get("predictions")
    # Contoh inverse transform (jika Anda punya encoder_target.joblib)
    # result_target = joblib.load("model/encoder_target.joblib")
    # final_result = result_target.inverse_transform(result)
    # return final_result
    return result