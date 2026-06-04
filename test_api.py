import requests

response = requests.post(
    #"http://127.0.0.1:8000/evaluate",
    "https://gd-secret-api.onrender.com/evaluate",
    json={"x": 2.5}
)

print(response.json())