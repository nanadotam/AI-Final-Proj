import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objs as go

# Define paths
model_path = "models/v2/cocoa_price_lstm_model.keras"
scaler_path = 'models/v2/cocoa_price_scaler.pkl'
data_path = 'data/cocoa_prices.csv'

# Check if paths exist
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
if not os.path.exists(scaler_path):
    st.error(f"Scaler file not found: {scaler_path}")
if not os.path.exists(data_path):
    st.error(f"Data file not found: {data_path}")

# Load the trained model and scaler
model = load_model(model_path, compile=False)
with open(scaler_path, 'rb') as f:
    scaler = joblib.load(f)

# Ensure the feature names match
feature_columns = ['Days', 'MA_7', 'MA_30', 'MA_90', 'Price_Diff', 'Volatility_7', 'Volatility_30', 'Volatility_90']

# Function to forecast future values using the data range directly
def forecast_future_values_direct(model, data, time_steps, forecast_period):
    predictions = []
    input_sequence = data[-time_steps:]
    
    for _ in range(forecast_period):
        pred = model.predict(input_sequence[np.newaxis, :, :])
        predictions.append(pred[0, 0])
        # Ensure the new prediction has the correct shape
        new_pred = np.zeros((1, input_sequence.shape[1]))
        new_pred[0, 0] = pred[0, 0]
        input_sequence = np.append(input_sequence[1:], new_pred, axis=0)
        
    return np.array(predictions)

# Load the data
df = pd.read_csv(data_path)

st.title('Cocoa Price Prediction - London Market')

st.header('Current Price')
latest_price = df.iloc[-1]['London futures (£ sterling/tonne)']
st.write(f"Current Price: £{latest_price}")

st.header('Predict Cocoa Prices for the Next 30 Days')
if st.button('Predict'):
    # Prepare the data for scaling
    X = df[feature_columns]
    X = X.fillna(0)  # Handle missing values
    X_scaled = scaler.transform(X)
    time_steps = 100
    forecast_days = 30
    predictions = forecast_future_values_direct(model, X_scaled, time_steps, forecast_days)
    
    latest_date = pd.to_datetime(df['Date']).max()
    dates = pd.date_range(start=latest_date, periods=forecast_days + 1).tolist()[1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=predictions, mode='lines+markers'))
    fig.update_layout(
        title='Cocoa Price Prediction for Next 30 Days - London Market', 
        xaxis_title='Date', 
        yaxis_title='Price (£/tonne)',
        yaxis=dict(tickformat='$.2f')  
    )
    
    st.plotly_chart(fig)

st.header('View Historical Prices')
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

if st.button('View Prices'):
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    historical_data = df.loc[mask]

    dates = historical_data['Date'].tolist()
    prices = historical_data['London futures (£ sterling/tonne)'].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers'))
    fig.update_layout(
        title=f'Cocoa Prices from {start_date} to {end_date} - London Market', 
        xaxis_title='Date', 
        yaxis_title='Price (£/tonne)'
    )

    st.plotly_chart(fig)
