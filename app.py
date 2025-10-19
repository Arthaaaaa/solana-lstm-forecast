# app.py
from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)

MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.save"
DATA_PATH = "Solana_daily_data_2018_2024.csv"
TIMESTEPS = 60

# Muat model & scaler saat startup
model, scaler = None, None
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    print("⚠️ Model atau scaler tidak ditemukan. Jalankan train_lstm.py dulu.")

# -------------------------------------------------------
# TEMPLATE HTML MODERN DENGAN PLOTLY INTERAKTIF
# -------------------------------------------------------
HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Solana LSTM Forecast</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">

<style>
    * {
        box-sizing: border-box;
    }
    body {
        font-family: 'Poppins', sans-serif;
        background: radial-gradient(circle at 20% 20%, rgba(58,12,163,0.4), transparent 70%),
                    radial-gradient(circle at 80% 80%, rgba(0,255,255,0.25), transparent 70%),
                    radial-gradient(circle at 30% 90%, rgba(34,197,94,0.25), transparent 70%),
                    #000;
        color: #e5e5e5;
        margin: 0;
        padding: 2rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        min-height: 100vh;
        overflow-x: hidden;
    }

    .container {
        background: #1a1a1a;
        border-radius: 20px;
        box-shadow: 0 0 30px rgba(0,0,0,0.6);
        padding: 2rem;
        width: 90%;
        max-width: 1000px;
        text-align: center;
        position: relative;
        z-index: 1;
    }

    img.logo {
        width: 120px;
        margin-top: 10px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    h2 {
        color: #a855f7;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    p {
        color: #cbd5e1;
        margin-bottom: 1.5rem;
    }

    .button-group {
        margin-bottom: 1rem;
    }

    button {
        background: linear-gradient(90deg, #8b5cf6, #22d3ee);
        color: white;
        border: none;
        padding: 10px 18px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 600;
        transition: transform 0.2s, opacity 0.2s;
    }

    button:hover {
        transform: scale(1.05);
        opacity: 0.9;
    }

    .reset {
        background: linear-gradient(90deg, #f87171, #ef4444);
        margin-left: 10px;
    }

    .result {
        background: #262626;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 0 20px rgba(255,255,255,0.05);
        width: 100%;
        margin-top: 1rem;
    }

    h3 {
        color: #a5b4fc;
        text-align: center;
        margin-top: 1rem;
    }

    /* Floating bottom logo gif */
    .floating-logo {
        position: fixed;
        bottom: 80px;
        right: 100px;
        width: 120px;
        height: 120px;
        opacity: 0.9;
        z-index: 10;
        transition: transform 0.3s ease;
    }
    .floating-logo:hover {
        transform: scale(1.1);
    }
</style>
</head>

<body>

<div class="container">
    <img src="/logo.png" class="logo" alt="Solana Logo">
    <h2>Solana LSTM Forecast</h2>
    <p>Prediksi harga Solana berbasis model LSTM dengan tampilan interaktif dan nuansa khas Solana.</p>

    <div class="button-group">
        <form action="/predict" method="post" style="display:inline;">
            <button type="submit">🔮 Predict 1 day</button>
        </form>
        <form action="/reset" method="post" style="display:inline;">
            <button type="submit" class="reset">♻️ Reset</button>
        </form>
    </div>

    <div class="result">
        {% if graph %}
            {{ graph | safe }}
        {% endif %}
        {% if pred %}
            <h3>Predicted Close (next day): {{ pred }}</h3>
        {% endif %}
    </div>
</div>

<!-- floating animated logo gif -->
<img src="/logo3d.gif" class="floating-logo" alt="Solana 3D Logo">

</body>
</html>
"""



# -------------------------------------------------------
# FUNGSI UTAMA UNTUK MEMBUAT GRAFIK INTERAKTIF
# -------------------------------------------------------
def generate_forecast_graph(df):
    # Ambil data historis terakhir
    last_close = df['Close'].values[-TIMESTEPS:].tolist()

    # Tambahkan prediksi sebelumnya kalau ada
    if os.path.exists("pred_sequence.txt"):
        with open("pred_sequence.txt", "r") as f:
            prev_preds = [float(x.strip()) for x in f.readlines()]
        last_close.extend(prev_preds)

    # Prediksi 7 hari ke depan
    future_preds = []
    seq = last_close.copy()
    for _ in range(7):
        scaled = scaler.transform(np.array(seq[-TIMESTEPS:]).reshape(-1, 1))
        X = scaled.reshape((1, TIMESTEPS, 1))
        pred_scaled = model.predict(X)
        pred = scaler.inverse_transform(pred_scaled)[0, 0]
        future_preds.append(pred)
        seq.append(pred)

    # Buat tanggal untuk prediksi ke depan
    last_date = df['Date'].max()
    future_dates = [last_date + datetime.timedelta(days=i + 1) for i in range(7)]

    # Plotly figure
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#3b82f6', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_preds,
        mode='lines+markers',
        name='7-Day Forecast',
        line=dict(color='#f97316', width=3, dash='dash'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='📈 Solana Price Forecast (Next 7 Days)',
        xaxis_title='Date',
        yaxis_title='Close Price (USD)',
        template='plotly_white',
        hovermode='x unified',
        font=dict(family='Poppins, sans-serif', size=14, color='#333'),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# -------------------------------------------------------
# ROUTES FLASK
# -------------------------------------------------------
@app.route('/')
def index():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    graph_html = generate_forecast_graph(df)
    return render_template_string(HTML, graph=graph_html)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return "Model belum tersedia. Jalankan train_lstm.py untuk membuat model.", 500

    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    last_close = df['Close'].values[-TIMESTEPS:].tolist()

    if os.path.exists("pred_sequence.txt"):
        with open("pred_sequence.txt", "r") as f:
            prev_preds = [float(x.strip()) for x in f.readlines()]
        last_close.extend(prev_preds)

    last_close = last_close[-TIMESTEPS:]

    scaled = scaler.transform(np.array(last_close).reshape(-1,1))
    X = scaled.reshape((1, TIMESTEPS, 1))
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)[0,0]

    with open("pred_sequence.txt", "a") as f:
        f.write(f"{pred}\n")

    graph_html = generate_forecast_graph(df)
    return render_template_string(HTML, pred=round(float(pred), 4), graph=graph_html)

@app.route('/reset', methods=['POST'])
def reset():
    if os.path.exists("pred_sequence.txt"):
        os.remove("pred_sequence.txt")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    graph_html = generate_forecast_graph(df)
    return render_template_string(HTML, pred="Prediksi di-reset.", graph=graph_html)

from flask import send_from_directory

@app.route('/logo.png')
def logo():
    return send_from_directory('.', 'logo.png')

@app.route('/logo3d.gif')
def logo3d():
    return send_from_directory('.', 'logo3d.gif')


# -------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

# -------------------------------------------------------