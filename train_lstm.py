# train_lstm.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib
import matplotlib.pyplot as plt

# --- konfigurasi sederhana ---
DATA_PATH = "Solana_daily_data_2018_2024.csv"
MODEL_DIR = "."
TIMESTEPS = 60          # gunakan 60 hari terakhir untuk memprediksi hari berikutnya
EPOCHS = 20             # untuk mulai: 15 epoch (bisa ditingkatkan)
BATCH_SIZE = 32

os.makedirs(MODEL_DIR, exist_ok=True)

# --- muat data ---
df = pd.read_csv(DATA_PATH)
# contoh kolom yang biasa ada: Date, Open, High, Low, Close, Volume
# pastikan kolom tanggal bernama 'Date' atau sesuaikan
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# kita prediksi 'Close'
series = df['Close'].values.reshape(-1, 1)

# normalisasi
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(series)

# create sequences
X, y = [], []
for i in range(TIMESTEPS, len(scaled)):
    X.append(scaled[i - TIMESTEPS:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# split train/test (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# bangun model LSTM sederhana
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# callback: simpan model terbaik
checkpoint_path = os.path.join(MODEL_DIR, "best_model.h5")
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

# latih
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

# load model terbaik
model.load_weights(checkpoint_path)

# prediksi pada test set
pred = model.predict(X_test)
pred_inv = scaler.inverse_transform(pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# simpan model final (agar mudah dipakai di Flask)
model.save("model.h5")

# simpan scaler
joblib.dump(scaler, "scaler.save")

# simpan grafik performa (opsional)
plt.figure(figsize=(10,6))
plt.plot(y_test_inv, label='Actual')
plt.plot(pred_inv, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted (Test set)')
plt.savefig("test_plot.png")
print("Selesai training. Model & scaler disimpan")
