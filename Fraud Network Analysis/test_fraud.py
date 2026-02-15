import requests

response = requests.post('http://localhost:5002/predict', json={"beneficiary_id": "BEN00001"})

print(response.json())
