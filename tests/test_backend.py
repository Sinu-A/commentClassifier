import requests

url = "http://localhost:8001/predict/"
data = {"text": ""}


response = requests.post(url, json=data)
print(response.json())