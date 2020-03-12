import requests

URL = 'http://0.0.0.0:5000/launch/1/1'

session = requests.session()
r = requests.post(URL)
# print(r.json())