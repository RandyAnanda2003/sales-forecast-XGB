import requests

url = 'http://127.0.0.1:5000/predict'
files = {'file': open('data baru euy.csv', 'rb')}

response = requests.post(url, files=files)
result = response.json()

# Cetak hasil prediksi dan analisis Gemini
print(result['predictions'])
print("\nAnalisis Gemini:\n", result['analysis'])
