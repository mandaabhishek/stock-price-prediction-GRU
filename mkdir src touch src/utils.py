#Handles data+preprocessing
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def fetch_data(ticker):
    """Fetch stock data from Yahoo Finance."""
    df = yf.download(ticker, start="2012-01-01", end="2025-01-01")
    if df.empty or 'Close' not in df.columns:
        return None
    return df[['Close']].dropna()

def create_dataset(data, time_step=60):
    """Convert series into supervised dataset for LSTM/GRU."""
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)
