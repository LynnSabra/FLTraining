import requests

r = requests.post("http://localhost:5555/predict", data={'model': 'test_model'},files={'image': open('test.jpg', 'rb')})

print(r.text)
