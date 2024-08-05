import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objs as go

# Define paths
model_path = 'models/cocoa_price_lstm_model.keras'
scaler_path = 'models/scaler.pkl'
data_path = 'data/preprocessed_ICCO_daily_prices.csv'

# Check if paths exist
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
if not os.path.exists(scaler_path):
    st.error(f"Scaler file not found: {scaler_path}")
if not os.path.exists(data_path):
    st.error(f"Data file not found: {data_path}")

# Load the trained model and scaler
model = load_model(model_path, compile=False)
scaler = joblib.load(scaler_path)

# Ensure the feature names match
feature_columns = ['ICCO daily price (US$/tonne)', 'MA_7', 'MA_30', 'MA_90', 'Price_Diff', 'Volatility_7', 'Volatility_30', 'Volatility_90']

# Function to forecast future values
def forecast_future_values(model, data, time_steps, forecast_period):
    predictions = []
    input_sequence = data[-time_steps:].reshape(1, time_steps, data.shape[1])
    
    for _ in range(forecast_period):
        pred = model.predict(input_sequence)
        predictions.append(pred[0, 0])
        new_pred = np.zeros((1, 1, data.shape[1]))
        new_pred[0, 0] = pred[0, 0]
        input_sequence = np.append(input_sequence[:, 1:, :], new_pred, axis=1)
        
    return np.array(predictions)

# Load the data
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])

st.title('Cocoa Price Prediction')

st.header('Current Price')
latest_price = df.iloc[-1]['New York futures (US$/tonne)']
st.write(f"Current Price: ${latest_price}")

st.header('Predict Cocoa Prices for the Next 30 Days')
if st.button('Predict'):
    data_scaled = scaler.transform(df[feature_columns])
    time_steps = 100
    forecast_days = 30
    predictions = forecast_future_values(model, data_scaled, time_steps, forecast_days)
    
    # Debugging statements to check the prediction values
    st.write(f"Scaled Predictions: {predictions}")

    # Inverse transform the predictions
    inverse_transformed = scaler.inverse_transform(np.concatenate([predictions.reshape(-1, 1), 
                                                                   np.zeros((forecast_days, len(feature_columns) - 1))], axis=1))[:, 0]
    
    # Debugging statements to check the inverse transformed predictions
    st.write(f"Inverse Transformed Predictions: {inverse_transformed}")

    # Convert predictions to thousands of dollars
    predictions_in_thousands = inverse_transformed / 1000

    latest_date = pd.to_datetime(df['Date']).max()
    dates = pd.date_range(start=latest_date, periods=forecast_days + 1).tolist()[1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=predictions_in_thousands, mode='lines+markers'))
    fig.update_layout(
        title='Cocoa Price Prediction for Next 30 Days', 
        xaxis_title='Date', 
        yaxis_title='Price (Thousands of US$/tonne)',
        yaxis=dict(tickformat='$,.2f')  
    )
    
    st.plotly_chart(fig)

st.header('View Historical Prices')
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

if st.button('View Prices'):
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    historical_data = df.loc[mask]

    dates = historical_data['Date'].tolist()
    prices = historical_data['New York futures (US$/tonne)'].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers'))
    fig.update_layout(
        title=f'Cocoa Prices from {start_date} to {end_date}', 
        xaxis_title='Date', 
        yaxis_title='Price (US$/tonne)'
    )

    st.plotly_chart(fig)
