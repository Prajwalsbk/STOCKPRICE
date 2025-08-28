import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import pickle
import datetime

# ====== CONFIGURATION ======
TICKER = 'AAPL'
START_DATE = '2015-01-01'
END_DATE = str(datetime.date.today())
SEQ_LEN = 60
EPOCHS = 5
BATCH_SIZE = 32

# ====== 1. Load Stock Data ======
print(f"Fetching stock data for {TICKER}...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)
df = df[['Close']]  # We'll predict 'Close' prices only

# ====== 2. Normalize Data ======
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# ====== 3. Prepare Sequences ======
x, y = [], []
for i in range(SEQ_LEN, len(scaled_data)):
    x.append(scaled_data[i-SEQ_LEN:i, 0])
    y.append(scaled_data[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # (samples, time_steps, features)

# ====== 4. Build LSTM Model ======
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# ====== 5. Train the Model ======
print("Training the model...")
model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

# ====== 6. Save Model and Scaler ======
model.save('lstm_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model saved as 'lstm_model.h5' and scaler as 'scaler.pkl'")
