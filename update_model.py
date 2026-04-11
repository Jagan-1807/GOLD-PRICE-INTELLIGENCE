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
    
    # Prepare for training
    X = df_merged[['SPX', 'USO', 'SLV', 'EUR/USD']]
    y = df_merged['GLD']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Model
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save Scaler
    with open('gold_price_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved gold_price_scaler.pkl")
        
    # Save Model
    with open('gold_price_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved gold_price_model.pkl")

    # Evaluate
    score = model.score(X_scaled, y)
    print(f"Model R^2 score on full data: {score:.4f}")
    
    print("Update complete!")

if __name__ == '__main__':
    main()
