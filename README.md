# Gold Price Intelligence

An end-to-end Machine Learning pipeline and web application that predicts gold prices (GLD) based on financial indicators (S&P 500, Crude Oil, Silver, and EUR/USD exchange rate).

## Problem Statement
Predicting the price of gold is complex due to its correlation with various macroeconomic factors. This project aims to demonstrate an end-to-end machine learning solution that fetches historical financial data, trains a Random Forest regression model to learn these correlations, and serves predictions via a simple web interface.

## Tech Stack
* **Backend Framework:** Flask
* **Machine Learning:** scikit-learn (Random Forest)
* **Data Manipulation:** pandas, numpy
* **Data Ingestion:** yfinance
* **Frontend:** HTML/CSS (Jinja templates), AOS (Animate on Scroll), vanilla-tilt.js

## Setup & Run Instructions

### 1. Install Dependencies
Ensure you have Python installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Update Data and Train Model
Before running the app, fetch the latest data and train the model. This will generate the necessary `.pkl` files for the web app:

```bash
python update_model.py
```
## Model Caveats & Limitations

* **Same-Day Co-movement, Not Forecasting:** This model evaluates *contemporaneous* (same-day) relationships between GLD and related assets (SPX, USO, SLV, EUR/USD). It is a diagnostic tool that estimates today's gold price movements given today's broader market movements, rather than predicting tomorrow's future price.
* **Return vs. Absolute Price:** Tree-based models (like Random Forests) cannot extrapolate beyond the raw ranges seen in training. Predicting absolute price on a chronological split produced severe overfitting and a negative R². To fix this, the model regresses on percentage returns, and the predicted absolute price is reconstructed algebraically from the last known price + predicted return. Evaluation is reported on a held-out chronological test set (R², RMSE, MAE).

### 3. Run the Web Application
Start the Flask development server:

```bash
python app.py
```

Then, open your browser and navigate to `http://localhost:5000/`. 

## Screenshot
*(Consider taking a screenshot of your running web app at `http://localhost:5000/` and saving it as `screenshot.png` in the project root to display it here).*
<!-- ![Web UI Screenshot](screenshot.png) -->
