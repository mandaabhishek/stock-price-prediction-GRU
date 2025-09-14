# src/utils.py
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def fetch_data(ticker, start="2012-01-01", end=None):
    """Return DataFrame with Close prices or None on failure."""
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end)
    if df.empty or 'Close' not in df.columns:
        return None
    return df[['Close']].dropna()

def create_dataset(data_array, time_step=60):
    """Build X,y sequences from scaled numpy array shape (n,1)."""
    X, y = [], []
    for i in range(len(data_array) - time_step - 1):
        X.append(data_array[i:(i + time_step), 0])
        y.append(data_array[i + time_step, 0])
    return np.array(X), np.array(y)

def get_scaler_and_scaled(data_values):
    """Return (scaler, scaled_array). data_values expected as 2D np array."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data_values)
    return scaler, scaled
