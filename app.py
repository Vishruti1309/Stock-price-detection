import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load model
model = load_model(r"C:\Users\drsol\Desktop\stock\Stock Predictions Model.keras")

# App UI
st.header('📈 Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2015-01-01'
end = '2025-12-31'

# Download data
try:
    data = yf.download(stock, start, end)
except Exception as e:
    st.error("Error downloading stock data.")
    st.stop()

st.subheader('Stock Data')
st.write(data)

# Train–test split
data_train = data.Close[:int(len(data)*0.80)]
data_test = data.Close[int(len(data)*0.80):]

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(data_train).reshape(-1, 1))

# Prepare last 100 days
pas_100_days = pd.DataFrame(data_train.tail(100))
final_data = pd.concat([pas_100_days, data_test], ignore_index=True)
final_scaled = scaler.transform(final_data.values.reshape(-1, 1))

# Create features
x, y = [], []
for i in range(100, len(final_scaled)):
    x.append(final_scaled[i-100:i])
    y.append(final_scaled[i, 0])

x = np.array(x)
y = np.array(y)

# --- Moving Averages ---
data['MA50'] = data['Close'].rolling(50).mean()
data['MA100'] = data['Close'].rolling(100).mean()
data['MA200'] = data['Close'].rolling(200).mean()

# === PRICE vs MA50 vs MA100 ===
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Price')
plt.plot(data['MA50'], label='MA50')
plt.plot(data['MA100'], label='MA100')
plt.title('Price vs MA50 vs MA100')
plt.legend()
st.pyplot(plt)
plt.close()

# === PRICE vs MA100 vs MA200 ===
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Price')
plt.plot(data['MA100'], label='MA100')
plt.plot(data['MA200'], label='MA200')
plt.title('Price vs MA100 vs MA200')
plt.legend()
st.pyplot(plt)
plt.close()

# Predictions
predictions = model.predict(x)

# Reverse scaling
scale_factor = 1 / scaler.scale_[0]
predicted_prices = predictions * scale_factor
actual_prices = y * scale_factor

# Final result chart
st.subheader("Predicted vs Actual Prices")
result_df = pd.DataFrame({
    "Actual Price": actual_prices,
    "Predicted Price": predicted_prices.flatten()
})

st.line_chart(result_df)

