import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Fetching updated data from yfinance...")
    # Tickers:
    # SPX: ^GSPC
    # GLD: GLD
    # USO: USO
    # SLV: SLV
    # EUR/USD: EURUSD=X
    
    tickers = {
        'SPX': '^GSPC',
        'GLD': 'GLD',
        'USO': 'USO',
        'SLV': 'SLV',
        'EUR/USD': 'EURUSD=X'
    }
    
    data_frames = []
    
    for name, ticker in tickers.items():
        print(f"Downloading {name} ({ticker})...")
        df = yf.download(ticker, start="2008-01-01")['Close']
        if isinstance(df, pd.DataFrame):
            # Sometimes yfinance returns a DataFrame for 'Close' if multiple tickers requested, but here it's single
            df = df.iloc[:, 0]
        df.name = name
        data_frames.append(df)
        
    df_merged = pd.concat(data_frames, axis=1)
    df_merged.dropna(inplace=True)
    df_merged.reset_index(inplace=True)
    df_merged.rename(columns={'Date': 'Date'}, inplace=True)
    
    # Format Date as MM/DD/YYYY similar to old file
    df_merged['Date'] = df_merged['Date'].dt.strftime('%m/%d/%Y')
    
    print(f"Data fetched. Total rows: {len(df_merged)}")
    
    # Save to CSV
    df_merged.to_csv('gld_price_data.csv', index=False)
    print("Saved gld_price_data.csv")
    
    # Save to clean pkl as used in app.py
    df_merged.to_pickle('gold_price_data_clean.pkl')
    print("Saved gold_price_data_clean.pkl")
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    # Target: next-step return instead of raw price level
    # (Random Forests can't extrapolate beyond price ranges seen in training,
    # so predicting absolute price fails badly on a chronological split.
    # Returns are roughly stationary, which fixes this.)
    df_merged['GLD_lag1'] = df_merged['GLD'].shift(1)
    df_merged['GLD_return'] = (df_merged['GLD'] - df_merged['GLD_lag1']) / df_merged['GLD_lag1']
    
    # Use same-day % returns of each asset as features, not raw price levels.
    # Levels carry almost no information about GLD's day-over-day % change;
    # the co-movement (correlation of returns) is the actual signal.
    for col in ['SPX', 'USO', 'SLV', 'EUR/USD']:
        df_merged[f'{col}_return'] = df_merged[col].pct_change()

    df_merged.dropna(inplace=True)

    feature_cols = ['SPX_return', 'USO_return', 'SLV_return', 'EUR/USD_return']
    X = df_merged[feature_cols]
    y = df_merged['GLD_return']
    
    # Chronological split — NOT shuffled, since this is time-series data
    split_idx = int(len(df_merged) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on UNSEEN future data
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f"Test R^2: {r2:.4f} | RMSE: {rmse:.6f} | MAE: {mae:.6f}")
    
    # Re-fit scaler/model on FULL data before saving, so the deployed
    # model uses all available history (standard practice once validated)
    X_scaled_full = StandardScaler().fit_transform(X)
    scaler = StandardScaler().fit(X)
    model.fit(scaler.transform(X), y)

    # Save Scaler
    with open('gold_price_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved gold_price_scaler.pkl")
        
    # Save Model
    with open('gold_price_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved gold_price_model.pkl")

    print("Update complete!")

if __name__ == '__main__':
    main()
