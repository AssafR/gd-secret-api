import requests
from service_ip import url, evaluate_url

response = requests.post(
    evaluate_url,
    json={"x": 2.5}
)

print("Answer: \n",response.json())


import requests

response = requests.post(
    evaluate_url,
    json={
        "problem_id": "bowl_2d",
        "x": [10.0, -3.0]
    }
)

print("Answer:")
print(response.json())