import requests
from service_ip import url, evaluate_url, evaluate_with_grad_url

response = requests.post(
    evaluate_with_grad_url,
    json={
        "problem_id": "parabola_1d",
        "x": [2.5]
    }
)

print("Answer:")
print(response.json())