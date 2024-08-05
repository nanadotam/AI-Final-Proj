from flask import Flask, render_template, request, redirect, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('models/cocoa_price_lstm_model.keras', compile=False)
scaler = joblib.load('models/scaler.pkl')

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

# Initialize Dash app
dash_app = Dash(__name__, server=app, url_base_pathname='/dashboard/')

dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.H1('Cocoa Price Prediction Dashboard'),
    dcc.Graph(id='cocoa-price-graph')
])

@app.route('/')
def index():
    df = pd.read_csv('data/preprocessed_ICCO_daily_prices.csv')
    current_price = df.iloc[-1]['New York futures (US$/tonne)']
    return render_template('index.html', current_price=current_price)

@app.route('/predict', methods=['POST'])
def predict():
    return redirect('/dashboard/')

@app.route('/historical', methods=['POST'])
def historical():
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    df = pd.read_csv('data/preprocessed_ICCO_daily_prices.csv')
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    historical_data = df.loc[mask]

    dates = historical_data['Date'].tolist()
    prices = historical_data['New York futures (US$/tonne)'].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers'))
    fig.update_layout(title=f'Cocoa Prices from {start_date} to {end_date}', xaxis_title='Date', yaxis_title='Price (US$/tonne)')

    return jsonify(fig.to_json())

# Dash callback to update the graph
@dash_app.callback(
    Output('cocoa-price-graph', 'figure'),
    Input('url', 'pathname')
)
def update_graph(pathname):
    if pathname == '/dashboard/':
        df = pd.read_csv('data/preprocessed_ICCO_daily_prices.csv')
        
        # Ensure we use the correct feature set for scaling
        feature_columns = ['ICCO daily price (US$/tonne)']
        data = df[feature_columns].values
        
        # Scale the data using the scaler fitted with the correct features
        data_scaled = scaler.transform(data)

        time_steps = 100
        forecast_days = 30
        predictions = forecast_future_values(model, data_scaled, time_steps, forecast_days)
        dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_days + 1).tolist()[1:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=predictions, mode='lines+markers'))
        fig.update_layout(title='Cocoa Price Prediction for Next 30 Days', xaxis_title='Date', yaxis_title='Price (US$/tonne)')

        return fig
    return go.Figure()

if __name__ == '__main__':
    app.run(debug=True)
