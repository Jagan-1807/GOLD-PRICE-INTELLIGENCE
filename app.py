from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
import warnings
import os

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Load the preprocessed data and model
try:
    # Load cleaned data
    df = pd.read_pickle('gold_price_data_clean.pkl')
    
    # Load trained model
    with open('gold_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    with open('gold_price_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # Model is trained on same-day % returns of these assets, not raw price levels
    features = ['SPX_return', 'USO_return', 'SLV_return', 'EUR/USD_return']
    
except Exception as e:
    print(f"Error loading files: {e}")
    df = None
    model = None
    scaler = None
    features = []

@app.route('/')
def home():
    """Main page with prediction form and recent data"""
    if df is not None:
        latest_data = df.tail(10).to_dict('records')
        return render_template('index.html', 
                            latest_data=latest_data,
                            features=features,
                            model_loaded=model is not None)
    return render_template('index.html', model_loaded=False)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None or scaler is None:
        return "Model not loaded properly", 500
    
    try:
        # Validate and get input values from form
        try:
            # Form fields now represent % change (e.g. user enters "1.2" for +1.2%)
            input_data = {
                'SPX': float(request.form.get('spx', '')),
                'USO': float(request.form.get('uso', '')),
                'SLV': float(request.form.get('slv', '')),
                'EUR/USD': float(request.form.get('eur_usd', ''))
            }
        except ValueError:
            return "Invalid input: Please ensure all fields are filled with valid numbers.", 400
        
        # Convert entered percentages (e.g. 1.2) to decimal returns (0.012)
        # to match the pct_change() convention used in training
        feature_values = [input_data[k] / 100 for k in ['SPX', 'USO', 'SLV', 'EUR/USD']]
        
        # Scale features
        features_scaled = scaler.transform([feature_values])
        
        # Make prediction (predicts return)
        predicted_return = model.predict(features_scaled)[0]
        last_known_gld = df['GLD'].iloc[-1]
        predicted_price = last_known_gld * (1 + predicted_return)
        
        return render_template('prediction.html',
                            prediction=round(predicted_price, 2),
                            predicted_return_pct=round(predicted_return * 100, 3),
                            input_data=input_data)
    
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", 400

@app.route('/historical')
def show_historical():
    """Display all historical data"""
    if df is not None:
        historical_data = df.to_dict('records')
        return render_template('historical.html',
                            historical_data=historical_data,
                            columns=df.columns.tolist())
    return "Historical data not available", 500

if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)