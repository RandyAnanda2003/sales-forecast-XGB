import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import google.generativeai as genai
import joblib
from sklearn.model_selection import train_test_split

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true)))) * 100

def create_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

def forecast_future(model, last_window, steps):
    preds = []
    window = last_window.copy()
    for _ in range(steps):
        p = model.predict(window.reshape(1, -1))[0]
        preds.append(p)
        window = np.roll(window, -1)
        window[-1] = p
    return np.array(preds)

def get_gemini_analysis(historical_df, predictions_json):
    historical_md = historical_df.reset_index()[['date', 'total_sales']].rename(
        columns={'total_sales': 'actual_sales'}
    ).to_markdown(index=False)
    
    future_dates = list(predictions_json.keys())
    future_sales = [item['total_sales'] for item in predictions_json.values()]
    future_df = pd.DataFrame({'date': future_dates, 'predicted_sales': future_sales})
    future_md = future_df.to_markdown(index=False)
    
    data_str = f"=== Data Historis ===\n{historical_md}\n\n=== Prediksi 30 Hari ===\n{future_md}"
    
    prompt = f"""
    Anda adalah analis bisnis e-commerce profesional. **JANGAN BANYAK BASA BASI. tolong buat paragraf singkat/ringkas saja, buat dalam satu paragraf seperti contoh output. ingat anda berbicara kepada pedagang bukan developer model !**

    Tugas Anda:
    - Jelaskan tren penjualan (naik/turun/stabil) dalam 30 hari ke depan.
    - Berikan alasan prediksi tersebut (misalnya: libur Lebaran/libur sekolah, akhir pekan, hari kerja, tanggal cantik seperti 11.11, dll).
    - Berikan saran strategi bisnis kepada pedagang e-commerce agar bisa memanfaatkan momen atau mengatasi penurunan penjualan.
    - Gunakan Bahasa Indonesia yang profesional dan mudah dimengerti.
    
    ini adalah tanggal penting :
    # tanggal cantik promo :
    "2025-01-01", "2025-01-11", "2025-01-25",
    "2025-02-02", "2025-02-14", "2025-02-25",
    "2025-03-03", "2025-03-08", "2025-03-25",
    "2025-04-04", "2025-04-25",
    "2025-05-05", "2025-05-25",
    "2025-06-06", "2025-06-25",
    "2025-07-07", "2025-07-25",
    "2025-08-08", "2025-08-17", "2025-08-25",
    "2025-09-09", "2025-09-25",
    "2025-10-10", "2025-10-25",
    "2025-11-11", "2025-11-25",
    "2025-12-12", "2025-12-25", "2025-12-31"
    # 2025 Holidays : 
    "2025-01-01", "2025-01-27", "2025-01-29", "2025-03-29", "2025-03-31",
    "2025-04-18", "2025-04-20", "2025-05-01", "2025-05-12", "2025-05-29",
    "2025-06-01", "2025-06-06", "2025-06-27", "2025-08-17", "2025-09-05",
    "2025-12-25"
    # Ramadan periods (1 month before Lebaran) : ("2025-03-01", "2025-03-30")
    # Lebaran periods (1 week after Idul Fitri) : ("2025-03-31", "2025-04-07")   # Idul Fitri 1446 H
    # School holiday periods : ("2025-06-15", "2025-07-15"),("2025-12-15", "2026-01-05")

    Berikut data penjualan (30 hari terakhir dan 30 hari ke depan):
    {data_str}
    
    **Contoh output:**
    Berdasarkan tren penjualan sebelumnya, diprediksi akan terjadi penurunan penjualan setelah libur Lebaran karena masyarakat kembali fokus bekerja. Disarankan untuk membuat promo pasca-Lebaran agar menarik perhatian pembeli. Perlu diingat bahwa kondisi pasar bisa berubah sewaktu-waktu, maka strategi ini hanya sebagai acuan awal.
    """
    
    genai.configure(api_key="AIzaSyDXiL2KIckvagWgQq2aqTDfgxx-YkbND0w")
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    MAX_RETRIES = 3
    for retry in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            
            if response.candidates and response.candidates[0].content.parts:
                text = response.candidates[0].content.parts[0].text
                if text.strip():
                    return text.strip()
            
            raise ValueError("Respons kosong")
        except Exception as e:
            if retry == MAX_RETRIES - 1:
                return f"Error: {str(e)}"
            continue
    
    return "Gagal mendapatkan analisis"

# 1. Load new data
df = pd.read_csv('data baru euy.csv')
df = df.copy()[['date', 'total_sales']]
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 2. Remove outliers (1% lowest and highest)
q_low, q_high = df['total_sales'].quantile([0.01, 0.99])
df = df[(df['total_sales'] > q_low) & (df['total_sales'] < q_high)]

# 3. Load existing model and scaler
model = joblib.load('sales_model.joblib')
scaler = joblib.load('sales_scaler.joblib')

# 4. Scale new data
scaled = scaler.transform(df[['total_sales']])

# 5. Create sequences with window size 30
WINDOW = 30
X, y = create_windows(scaled, WINDOW)

# 6. Retrain model with new data
model.fit(
    X, y,
    eval_set=[(X, y)],
    verbose=False
)

# Save updated model
joblib.dump(model, 'sales_model.joblib')

# 7. Forecast next 30 days
last_win = scaled[-WINDOW:, 0]
future_steps = 30
future_scaled = forecast_future(model, last_win, future_steps)
future_preds = scaler.inverse_transform(future_scaled.reshape(-1, 1)).flatten()

future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_steps)

# Create predictions dictionary
predictions_dict = {
    str(date.date()): {"total_sales": float(sales)}
    for date, sales in zip(future_dates, future_preds)
}

# Get Gemini analysis
historical_data = df.tail(30)
gemini_recommendation = get_gemini_analysis(historical_data, predictions_dict)

# Prepare final output
output_data = {
    "predictions": predictions_dict,
    "analysis": gemini_recommendation
}

# Output only the JSON
print(json.dumps(output_data, indent=4, ensure_ascii=False))
