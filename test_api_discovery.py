import requests
import json

from service_ip import discovery_url

response = requests.get(
    discovery_url
)


# Beautify JSON response
print("Answer: \n", json.dumps(response.json(), indent=4))
