# src/utils.py
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def fetch_data(ticker, start="2012-01-01", end=None):
    """Fetch historical Close prices for ticker using yfinance."""
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty or 'Close' not in df.columns:
        return None
    return df[['Close']].dropna()

def get_scaler_and_scaled(data_values):
    """Return fitted MinMaxScaler and scaled values (expects 2D array)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data_values)
    return scaler, scaled

def create_dataset(data_array, time_step=60):
    """
    Convert scaled numpy array (n,1) into X,y for supervised learning.
    X shape: (samples, time_step)
    y shape: (samples,)
    """
    X, y = [], []
    for i in range(len(data_array) - time_step - 1):
        X.append(data_array[i:(i + time_step), 0])
        y.append(data_array[i + time_step, 0])
    return np.array(X), np.array(y)

