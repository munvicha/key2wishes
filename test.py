import json
import requests
l = ["vui vẻ", 'khỏe mạnh']
data = {
    'kws': l
}
res = requests.post('http://127.0.0.1:8000/api', json=data)
data = json.loads(res.text)

print(type(res))
print("Text:", data.get('text'))
print('Time:', data.get('time'))