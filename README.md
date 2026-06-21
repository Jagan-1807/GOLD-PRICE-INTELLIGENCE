# 🥇 Gold Price Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/scikit--learn-Random%20Forest-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

An **end-to-end Machine Learning web application** that estimates the same-day movement in GLD (Gold ETF) price based on the percentage returns of four correlated financial assets: the S&P 500 (SPX), Crude Oil ETF (USO), Silver ETF (SLV), and the EUR/USD exchange rate.

---

## 📸 Screenshot

![Gold Price Intelligence App](static/screenshot.png)

---

## 📌 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Dataset Source](#-dataset-source)
3. [Features Used](#-features-used)
4. [Model Type & Metrics](#-model-type--metrics)
5. [How It Works](#-how-it-works)
6. [Tech Stack](#-tech-stack)
7. [Project Structure](#-project-structure)
8. [How to Run](#-how-to-run)
9. [Model Caveats & Limitations](#️-model-caveats--limitations)
10. [Web Application Features](#-web-application-features)

---

## 🎯 Problem Statement

Gold is one of the most complex assets to predict because its price is driven by a confluence of macroeconomic forces: inflation expectations, currency strength, equity market sentiment, and commodity correlations. This project demonstrates a complete, production-aware data science workflow — from raw data ingestion all the way to a live Flask web application — using four highly correlated financial instruments as predictive features.

The core research question is: **Given how the broader market moved today, how much did gold likely move?**

---

## 📦 Dataset Source

- **Source:** [Yahoo Finance](https://finance.yahoo.com/) via the [`yfinance`](https://github.com/ranaroussi/yfinance) Python library
- **Tickers pulled:** `^GSPC` (S&P 500), `GLD`, `USO`, `SLV`, `EURUSD=X`
- **Time range:** 2008-01-01 → present (daily OHLC, adjusted Close used)
- **Rows:** ~4,600 trading days (after inner join + dropna)
- **Refresh:** Run `python update_model.py` at any time to fetch fresh data and retrain

The raw CSV (`gld_price_data.csv`) and cleaned pickle (`gold_price_data_clean.pkl`) are auto-generated on each run of `update_model.py`.

---

## 🔢 Features Used

The model uses **same-day percentage returns** (via `pct_change()`) — not raw price levels — as input features:

| Feature | Description | Ticker |
|:---|:---|:---|
| `SPX_return` | S&P 500 Index daily % return | `^GSPC` |
| `USO_return` | Crude Oil ETF daily % return | `USO` |
| `SLV_return` | Silver ETF daily % return | `SLV` |
| `EUR/USD_return` | Euro/Dollar exchange rate daily % return | `EURUSD=X` |

**Target:** `GLD_return` — the same-day percentage return of the GLD gold ETF.

> **Why returns, not raw prices?**  
> Raw price levels carry almost no information about whether GLD moved up or down *today*. The meaningful signal is co-movement: when SPX rises 1.2%, does gold tend to move in a particular direction? Returns are approximately stationary and encode exactly this correlation structure. Using raw prices would also cause Random Forest to fail badly on a chronological split (it can't extrapolate beyond price ranges seen during training).

---

## 📊 Model Type & Metrics

### Model
- **Algorithm:** `RandomForestRegressor` (scikit-learn)
- **Hyperparameters:** `n_estimators=100`, `random_state=42`, all other params at sklearn defaults
- **Training split:** Strict **chronological 80/20 split** (no shuffling — prevents future data leakage)
  - Training: first ~3,680 rows (2008 – ~2022)
  - Test: last ~920 rows (~2022 – present)
- **Scaling:** `StandardScaler` fit on train, applied to both train and test

### Test-Set Metrics (held-out 20% — data the model never saw during training)

| Metric | Value | Interpretation |
|:---|:---|:---|
| **R²** | **0.5887** | The model explains ~59% of the variance in GLD's same-day return |
| **RMSE** | **0.007873** | Average prediction error of ±0.79% in daily return terms |
| **MAE** | **0.005800** | Median absolute prediction error of ±0.58% in daily return terms |

> **Context:** An R² of ~0.59 is competitive for single-day financial return prediction. The dominant driver is the very high same-day correlation between SLV (Silver) and GLD (Gold), which are both driven by the same macro factors. The model does **not** predict tomorrow's price — it nowcasts today's co-movement.

### Deployment
After test evaluation, the model and scaler are **refit on the full dataset** before saving — this is standard practice (validate first on a held-out set, then use all available history for the final deployed model).

### Price Reconstruction
The model predicts a return. The app then reconstructs the absolute predicted GLD price as:

```
Predicted Price = last_known_GLD_price × (1 + predicted_return)
```

---

## ⚙️ How It Works

```
yfinance API ──► Data Fetching ──► Feature Engineering (% returns) ──► Model Training
                                                                              │
                                                                              ▼
User Form Input ──► /predict Route ──► StandardScaler ──► RandomForestRegressor ──► Predicted Return
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
├── app.py                    # Flask web application (routes, prediction logic)
├── update_model.py           # Data fetching, feature engineering, model training
├── gold_price_pred.ipynb     # EDA & model exploration notebook
├── requirements.txt          # Python dependency list
├── .gitignore                # Excludes .pkl files, __pycache__, .ipynb_checkpoints, .env
├── README.md                 # This file
│
├── templates/
│   ├── index.html            # Main page with prediction form and market chart
│   ├── prediction.html       # Displays the prediction result
│   └── historical.html       # Full historical data table
│
└── static/
    ├── style.css             # Shared CSS styles
    └── screenshot.png        # App UI screenshot
```

> **Note:** `.pkl` files (`gold_price_model.pkl`, `gold_price_scaler.pkl`, `gold_price_data_clean.pkl`) are intentionally **not tracked** in this repository. They are regenerated locally by running `update_model.py`.

---

## 🚀 How to Run

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

> **Security Note:** The app only runs in debug mode if you explicitly set the environment variable `FLASK_DEBUG=1`. It is off by default. Never run with `debug=True` on a public-facing host.

---

## ⚠️ Model Caveats & Limitations

### 1. Same-Day Co-movement, Not Future Forecasting
This model estimates **today's** GLD movement given **today's** broader market movements. It is a **nowcasting / explanatory model**, not a forecasting model. If your SPX, USO, SLV, and EUR/USD inputs are from today's market, the output is an estimate of today's GLD price — not tomorrow's.

The R² of 0.59 is largely explained by the strong same-day correlation between SLV (Silver) and GLD (Gold), which move together because they share the same macroeconomic drivers.

### 2. Random Forest Cannot Extrapolate
Tree-based models split data into regions based on what they've seen during training. If you enter extreme percentage changes (e.g., SPX down 15% in a single day — a crash scenario), the model will silently cap its prediction at the most extreme return it observed in training, rather than extrapolating correctly. This is a fundamental property of all tree-based models.

### 3. Data Freshness
The model is only as accurate as the data it was last trained on. Re-run `python update_model.py` periodically to retrain on the most recent market data.

---

## 🌐 Web Application Features

- **Live Prediction Form** — Enter the current day's percentage changes for SPX, USO, SLV, and EUR/USD (with sliders) to get a predicted GLD price and the underlying return %.
- **Market Snapshot Chart** — A Chart.js line graph showing the most recent 10 data points of GLD and SPX.
- **Full Historical Data Table** — Browse the complete dataset used for model training (2008–present).
- **Scroll Animations** — AOS (Animate on Scroll) for smooth, progressive page reveals.
- **Glassmorphism UI** — A premium dark theme with blurred glass cards and a dynamic parallax background.
- **Input Validation** — Empty or non-numeric form fields return a friendly error message rather than a server traceback.
