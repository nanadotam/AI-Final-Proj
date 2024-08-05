from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Load the Keras model
model = load_model('models/cocoa_price_lstm_model.keras')

# Function to load preprocessed data
def load_preprocessed_data():
    df = pd.read_csv('data/preprocessed_ICCO_daily_prices.csv')
    return df

# Function to forecast future values
def forecast_future_values(model, data, time_steps, forecast_period):
    predictions = []
    input_sequence = data[-time_steps:].reshape(1, time_steps, data.shape[1])

    for _ in range(forecast_period):
        pred = model.predict(input_sequence)
        predictions.append(pred[0, 0])
        new_pred = np.zeros((1, 1, data.shape[1]))
        new_pred[0, 0, 0] = pred[0, 0]
        input_sequence = np.append(input_sequence[:, 1:, :], new_pred, axis=1)

    return np.array(predictions)

# Function to plot the graph
def plot_graph(dates, prices, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price (US$/tonne)')
    plot_html = pio.to_html(fig, full_html=False)
    return plot_html

@app.route('/')
def index():
    df = load_preprocessed_data()
    current_price = df.iloc[-1]['ICCO daily price (US$/tonne)']
    return render_template('index.html', current_price=current_price)

@app.route('/predict', methods=['POST'])
def predict():
    df = load_preprocessed_data()
    data = df[['ICCO daily price (US$/tonne)', 'MA_7', 'MA_30', 'MA_90', 'Price_Diff']].values
    time_steps = 100

    forecast_days = 30
    predictions = forecast_future_values(model, data, time_steps, forecast_days)
    dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_days + 1).tolist()[1:]

    plot_html = plot_graph(dates, predictions, 'Cocoa Price Prediction for Next 30 Days')
    return jsonify({'plot_html': plot_html})

@app.route('/historical', methods=['POST'])
def historical():
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    df = load_preprocessed_data()
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    historical_data = df.loc[mask]

    dates = historical_data['Date'].tolist()
    prices = historical_data['ICCO daily price (US$/tonne)'].tolist()

    plot_html = plot_graph(dates, prices, f'Cocoa Prices from {start_date} to {end_date}')
    return jsonify({'plot_html': plot_html})

if __name__ == '__main__':
    app.run(debug=True)
