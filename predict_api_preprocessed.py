import requests
import json

with open("input.json") as f:
    payload = json.load(f)

response = requests.post(
    "http://127.0.0.1:5002/invocations",
    json=payload,
    headers={"Content-Type": "application/json"}
)
res = response.json()
print("Response:", res)
print("Predictions:", res.get("predictions"))