from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the Keras model
model = load_model('/path/to/your/model.h5')

# Load your dataset (assuming you have a pickled scaler and dataframe for consistency)
scaler = pickle.load(open('/path/to/your/scaler.pkl', 'rb'))
df = pd.read_csv('/path/to/your/ICCO_daily_prices.csv')

# Preprocess the data
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['ICCO daily price (US$/tonne)'] = pd.to_numeric(df['ICCO daily price (US$/tonne)'].str.replace(',', ''), errors='coerce')
    df = df.dropna()
    df = df.sort_values('Date')
    return df

# Function to forecast future values
def forecast_future_values(model, data, time_steps, forecast_period):
    predictions = []
    input_sequence = data[-time_steps:]

    for _ in range(forecast_period):
        pred = model.predict(input_sequence.reshape(1, time_steps, -1))
        predictions.append(pred[0, 0])
        input_sequence = np.append(input_sequence[1:], pred, axis=0)

    return np.array(predictions)

# Preprocess the initial data
df = preprocess_data(df)
scaled_data = scaler.transform(df[['ICCO daily price (US$/tonne)']])
time_steps = 100  # Assuming a time step of 100, as per your previous code

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    forecast_days = int(request.form.get('forecast', 0))
    start_date = request.form.get('start_date')
    month = request.form.get('month')
    window_size = int(request.form.get('window_size', 0))
    past_start_date = request.form.get('past_start_date')
    past_end_date = request.form.get('past_end_date')
    forecast_30 = request.form.get('forecast_30') == 'on'
    specific_period_start = request.form.get('specific_period_start')
    specific_period_days = int(request.form.get('specific_period_days', 0))

    # Initialize results
    results = {
        'forecast': [],
        'month_forecast': [],
        'window_size_forecast': [],
        'historical_data': [],
        'forecast_30_days': [],
        'specific_period_forecast': []
    }

    # Forecast future values
    if forecast_days > 0:
        results['forecast'] = forecast_future_values(model, scaled_data, time_steps, forecast_days).tolist()

    # Forecast for a specific month
    if month:
        month_data = df[df['Date'].dt.month == int(month)]
        scaled_month_data = scaler.transform(month_data[['ICCO daily price (US$/tonne)']])
        results['month_forecast'] = forecast_future_values(model, scaled_month_data, time_steps, 7).tolist()

    # Forecast using a specific window size
    if window_size > 0:
        results['window_size_forecast'] = forecast_future_values(model, scaled_data, time_steps, window_size).tolist()

    # Historical data between specific dates
    if past_start_date and past_end_date:
        mask = (df['Date'] >= past_start_date) & (df['Date'] <= past_end_date)
        historical_data = df.loc[mask]
        results['historical_data'] = historical_data.to_dict(orient='records')

    # Forecast for the next 30 days
    if forecast_30:
        results['forecast_30_days'] = forecast_future_values(model, scaled_data, time_steps, 30).tolist()

    # Forecast for a specific period
    if specific_period_start and specific_period_days > 0:
        specific_start_index = df[df['Date'] == specific_period_start].index[0]
        specific_period_data = scaled_data[specific_start_index:specific_start_index+time_steps]
        results['specific_period_forecast'] = forecast_future_values(model, specific_period_data, time_steps, specific_period_days).tolist()

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
