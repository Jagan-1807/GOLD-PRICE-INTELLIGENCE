from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
import warnings

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
        
    # Get feature names for the form
    features = ['SPX', 'USO', 'SLV', 'EUR/USD']
    
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
        # Get input values from form
        input_data = {
            'SPX': float(request.form['spx']),
            'USO': float(request.form['uso']),
            'SLV': float(request.form['slv']),
            'EUR/USD': float(request.form['eur_usd'])
        }
        
        # Prepare features in correct order
        feature_values = [input_data[feature] for feature in features]
        
        # Scale features
        features_scaled = scaler.transform([feature_values])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return render_template('prediction.html',
                            prediction=round(prediction, 2),
                            input_data=input_data)
    
    except Exception as e:
        return f"Error making prediction: {str(e)}", 400

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
    app.run(debug=True, host='0.0.0.0', port=5000)