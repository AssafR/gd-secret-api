import requests
from service_ip import url, evaluate_url, evaluate_with_grad_url


response = requests.post(
    evaluate_url,
    json={
        "problem_id": "bowl_2d",
        "x": [10.0, -3.0]
    }
)

print("Answer:")
print(response.json())

response = requests.post(
    evaluate_with_grad_url,
    json={
        "problem_id": "bowl_2d",
        "x": [10.0, -3.0]
    }
)

print("Answer:")
print(response.json())