import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Fetch stock data
def fetch_data(ticker, loading_label):
    try:
        loading_label.config(text="📊 Loading historical data...", foreground="blue")
        root.update_idletasks()
        df = yf.download(ticker, start="2012-01-01", end="2025-01-01")
        loading_label.config(text="")
        if df.empty or 'Close' not in df.columns:
            return None
        return df[['Close']].dropna()
    except Exception as e:
        loading_label.config(text="")
        print("Error fetching data:", e)
        return None

# Prepare dataset
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Build model with Bidirectional LSTM
def train_model(X, y):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=10, verbose=0)
    return model

# Main prediction routine
def run_prediction():
    ticker = ticker_entry.get().strip().upper()
    if not ticker:
        messagebox.showwarning("Input Error", "Please enter a valid stock ticker.")
        return

    loading_label.config(text="⏳ Fetching data...", foreground="blue")
    root.update_idletasks()
    df = fetch_data(ticker, loading_label)
    if df is None or df.empty:
        messagebox.showerror("Data Error", f"Failed to load data for ticker '{ticker}'.")
        return

    data = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    loading_label.config(text="⚙️ Training model...", foreground="blue")
    root.update_idletasks()
    model = train_model(X, y)

    # Forecast next 50 days
    future_steps = 50
    test_data = list(scaled_data[-time_step:].flatten())
    future_preds_scaled = []

    for _ in range(future_steps):
        x_input = np.array(test_data[-time_step:]).reshape(1, time_step, 1)
        pred = model.predict(x_input, verbose=0)
        test_data.append(pred[0][0])
        future_preds_scaled.append(pred[0][0])

    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))

    # Metrics calculation
    split = int(len(scaled_data) * 0.8)
    test_actual = scaled_data[split:]
    test_X, test_y = create_dataset(test_actual, time_step)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
    test_predictions = model.predict(test_X, verbose=0)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test = scaler.inverse_transform(test_y.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    mae = mean_absolute_error(y_test, test_predictions)
    mse = mean_squared_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)
    mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100
    mean_actual = np.mean(y_test)
    accuracy = 100 * (1 - (rmse / mean_actual))

    result_label.config(
        text=(f"📈 Predicted Prices (Next 50 Days):\n" +
              f"{', '.join([f'{p[0]:.2f}' for p in future_preds[:5]])}...\n\n"
              f"📉 RMSE: ₹{rmse:.2f}   |   MAE: ₹{mae:.2f}\n"
              f"🔁 MSE: ₹{mse:.2f}   |   R² Score: {r2:.4f}\n"
              f"📊 MAPE: {mape:.2f}%   |   Approx. Accuracy: {accuracy:.2f}%"),
        foreground="green"
    )

    loading_label.config(text="✅ Done!", foreground="green")

    # Plotting
    for widget in plot_frame.winfo_children():
        widget.destroy()

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

    fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=100)
    ax.plot(df.index[-100:], df['Close'][-100:], label='Historical (Close)')
    ax.plot(future_dates, future_preds, label='Forecast (50 Days)', color='red', linestyle='--', marker='o', markersize=3)
    ax.set_title(f"{ticker} - Last 100 Days & 50-Day Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.grid(True)
    ax.legend()

    fig.autofmt_xdate()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # Save forecast to CSV
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_preds.flatten()})
    prediction_df.to_csv(f'{ticker}_50_day_prediction.csv', index=False)

# Threaded execution
def start_prediction_thread():
    threading.Thread(target=run_prediction, daemon=True).start()

# GUI window setup
def close_app():
    root.destroy()

root = tk.Tk()
root.title("📈 Stock Price Predictor - 50 Day Forecast")
root.geometry("880x700")
root.configure(bg="#f2f2f2")

style = ttk.Style()
style.configure('TLabel', background="#f2f2f2", font=('Segoe UI', 12))
style.configure('TEntry', font=('Segoe UI', 12))
style.configure('TButton', font=('Segoe UI', 12, 'bold'))
style.map('TButton', background=[('active', '#45a049')])

# Input widgets
frame = ttk.Frame(root, padding="20 10 20 0")
frame.pack(fill='x')

label = ttk.Label(frame, text="Enter Stock Ticker (e.g., AAPL, TSLA, INFY.NS):")
label.pack(side='left')

ticker_entry = ttk.Entry(frame, width=20)
ticker_entry.pack(side='left', padx=10)

predict_button = ttk.Button(frame, text="🔍 Predict", command=start_prediction_thread)
predict_button.pack(side='left', padx=5)

# Output labels
result_label = ttk.Label(root, text="", font=('Segoe UI', 14))
result_label.pack(pady=10)

loading_label = ttk.Label(root, text="", font=('Segoe UI', 12, 'italic'))
loading_label.pack()

# Plotting frame
plot_frame = tk.Frame(root, bg="#ffffff", relief="groove", bd=2)
plot_frame.pack(pady=10, fill='both', expand=True, padx=20)

# Exit button
close_button = ttk.Button(root, text="❌ Close", command=close_app)
close_button.pack(pady=10)

root.protocol("WM_DELETE_WINDOW", close_app)
root.mainloop()

#COMMENT