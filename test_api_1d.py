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
        "problem_id": "parabola_1d",
        "x": [2.5]
    }
)

print("Answer:")
print(response.json())