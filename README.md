# 🥇 Gold Price Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/scikit--learn-Random%20Forest-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

An **end-to-end Machine Learning web application** that estimates the same-day movement in GLD (Gold ETF) price based on the percentage returns of four correlated financial assets: the S&P 500 (SPX), Crude Oil ETF (USO), Silver ETF (SLV), and the EUR/USD exchange rate.

---

## 📌 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [How It Works](#-how-it-works)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Setup & Run Instructions](#-setup--run-instructions)
6. [Model Methodology](#-model-methodology)
7. [Model Caveats & Limitations](#%EF%B8%8F-model-caveats--limitations)
8. [Web Application Features](#-web-application-features)

---

## 🎯 Problem Statement

Gold is one of the most complex assets to predict because its price is driven by a confluence of macroeconomic forces: inflation expectations, currency strength, equity market sentiment, and commodity correlations. This project demonstrates a complete, production-aware data science workflow — from raw data ingestion all the way to a live Flask web application — using four highly correlated financial instruments as predictive features.

The core research question is: **Given how the broader market moved today, how much did gold likely move?**

---

## ⚙️ How It Works

```
yfinance API ──► Data Fetching ──► Feature Engineering (% returns) ──► Model Training
                                                                              │
                                                                              ▼
User Form Input ──► /predict Route ──► Scaler ──► RandomForestRegressor ──► Predicted Return
                                                                              │
                                                                              ▼
                                              last_known_price × (1 + predicted_return) = Predicted GLD Price
```

1. **`update_model.py`** fetches daily OHLC data from Yahoo Finance via `yfinance`, engineers percentage return features, trains a `RandomForestRegressor` on an 80% chronological training split, evaluates it on the held-out 20% test split, then refits on the full dataset and saves the model and scaler as `.pkl` files.

2. **`app.py`** (Flask server) loads the saved model and scaler on startup, serves the web UI, and handles prediction requests by scaling the user's input and reconstructing the absolute price from the model's return prediction.

3. **`templates/`** contains the Jinja2 HTML templates for the main page, the prediction result page, and the historical data page.

---

## 🛠️ Tech Stack

| Layer | Technology |
|:---|:---|
| **Language** | Python 3.10+ |
| **Backend Framework** | Flask |
| **ML Library** | scikit-learn (RandomForestRegressor) |
| **Data Ingestion** | yfinance |
| **Data Manipulation** | pandas, numpy |
| **Frontend Styling** | Vanilla CSS (Glassmorphism / dark theme) |
| **Scroll Animations** | AOS — Animate On Scroll |
| **Font** | Google Fonts — Outfit |

---

## 📁 Project Structure

```
GPP/
│
├── app.py                  # Flask web application (routes, prediction logic)
├── update_model.py         # Data fetching, feature engineering, model training
├── requirements.txt        # Python dependency list
├── .gitignore              # Excludes .pkl files, __pycache__, .env
├── README.md               # This file
│
├── templates/
│   ├── index.html          # Main page with prediction form and market chart
│   ├── prediction.html     # Displays the prediction result
│   └── historical.html     # Full historical data table
│
└── gld_price_data.csv      # Latest downloaded market data (auto-generated)
```

> **Note:** `.pkl` files (`gold_price_model.pkl`, `gold_price_scaler.pkl`, `gold_price_data_clean.pkl`) are intentionally **not tracked** in this repository. They are regenerated locally by running `update_model.py`.

---

## 🚀 Setup & Run Instructions

### Prerequisites
- Python 3.10 or higher
- An active internet connection (for fetching live market data via `yfinance`)

### 1. Clone the Repository

```bash
git clone https://github.com/Jagan-1807/GOLD-PRICE-INTELLIGENCE.git
cd GOLD-PRICE-INTELLIGENCE
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

This step downloads historical market data (2008–present), engineers features, trains the model, and saves the `.pkl` artifacts needed by the web app:

```bash
python update_model.py
```

**Expected output:**
```
Fetching updated data from yfinance...
Data fetched. Total rows: ~4620
Training RandomForestRegressor model...
Test R^2: 0.5887 | RMSE: 0.007873 | MAE: 0.005800
Saved gold_price_scaler.pkl
Saved gold_price_model.pkl
Update complete!
```

### 4. Run the Web Application

```bash
python app.py
```

Open your browser and navigate to **`http://localhost:5000`**.

> **Security Note:** The app only runs in debug mode if you explicitly set the environment variable `FLASK_DEBUG=1`. Never run with `debug=True` on a public-facing host.

---

## 📊 Model Methodology

### Feature Engineering
The four input features are the **same-day percentage returns** (via `pct_change()`) of:
- `SPX_return` — S&P 500 Index daily return
- `USO_return` — Crude Oil ETF daily return
- `SLV_return` — Silver ETF daily return
- `EUR/USD_return` — Euro/US Dollar exchange rate daily return

### Why Returns, Not Raw Prices?
Raw price levels (e.g., "SPX is at 5000") carry almost no information about whether GLD moved up or down on a given day. The meaningful signal is **co-movement**: when the S&P 500 *rises* 1.2%, does gold tend to move in a particular direction? This return-on-return framing is the standard, economically sound approach for this type of model.

### Why Chronological Split?
Since this is sequential time-series data, a **random shuffle split would leak future information into training**. We use a strict chronological 80/20 split: the first 80% of rows (by date) are used for training, and the last 20% are held out as the test set. The model never sees future data during training.

### Evaluation
Metrics are reported **only on the held-out test set**:
- **R²**: 0.5887 (the model explains ~59% of the variance in GLD's same-day return)
- **RMSE**: 0.007873
- **MAE**: 0.005800

After evaluation, the model and scaler are **refit on the full dataset** before being saved — this is standard practice (validate first, then use all available history for the final deployed model).

### Price Reconstruction
The model predicts a **return value** (e.g., +0.109%). The absolute predicted GLD price is then reconstructed as:

```
Predicted Price = last_known_GLD_price × (1 + predicted_return)
```

---

## ⚠️ Model Caveats & Limitations

### 1. Same-Day Co-movement, Not Future Forecasting
This model estimates **today's** GLD movement given **today's** broader market movements. It is a **nowcasting / explanatory model**, not a forecasting model. If your SPX, USO, SLV, and EUR/USD inputs are from today's market, the output is an estimate of today's GLD price — not tomorrow's.

The high R² (0.59) is largely explained by the strong same-day correlation between SLV (Silver) and GLD (Gold), which move together because they share the same macroeconomic drivers.

### 2. Random Forest Cannot Extrapolate
Tree-based models split data into regions based on what they've seen during training. If you enter extreme percentage changes (e.g., SPX down 15% in a single day — a crash scenario), the model will silently cap its prediction at the most extreme return it observed in training, rather than extrapolating correctly. This is a fundamental property of all tree-based models.

### 3. Data Freshness
The model is only as accurate as the data it was last trained on. Re-run `update_model.py` periodically to retrain on the most recent market data.

---

## 🌐 Web Application Features

- **Live Prediction Form** — Enter the current day's percentage changes for SPX, USO, SLV, and EUR/USD to get a predicted GLD price and the underlying return %.
- **Market Snapshot Chart** — A Chart.js line graph showing the most recent 10 data points of all tracked assets.
- **Full Historical Data Table** — Browse the complete dataset used for model training (2008–present).
- **Scroll Animations** — AOS (Animate on Scroll) for smooth, progressive page reveals.
- **Glassmorphism UI** — A premium dark theme with blurred glass cards and a dynamic parallax background.
- **Input Validation** — Empty or non-numeric form fields return a friendly error message rather than a server traceback.
