# src/gui.py
import os
import threading
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model

from src.utils import fetch_data, get_scaler_and_scaled, create_dataset
from src.train import train_model
from src.predict import predict_future, evaluate_model

MODEL_PATH = "models/gru_model.h5"
TIME_STEP = 60
DEFAULT_FUTURE_DAYS = 5

def run_prediction_threaded(ticker, future_days, loading_label, result_label, plot_frame):
    def worker():
        try:
            loading_label.config(text="📊 Fetching historical data...", foreground="blue")
            df = fetch_data(ticker)
            if df is None or df.empty:
                loading_label.config(text="")
                messagebox.showerror("Data Error", f"Failed to load data for ticker '{ticker}'.")
                return

            loading_label.config(text="🔁 Scaling data...", foreground="blue")
            data = df.values  # shape (n,1)
            scaler, scaled_data = get_scaler_and_scaled(data)

            loading_label.config(text="⚙️ Preparing dataset...", foreground="blue")
            X, y = create_dataset(scaled_data, TIME_STEP)
            if len(X) == 0:
                loading_label.config(text="")
                messagebox.showerror("Data Error", "Not enough historical data for given time_step.")
                return
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Load existing model if exists
            if os.path.exists(MODEL_PATH):
                loading_label.config(text="🔁 Loading existing model...", foreground="blue")
                model = load_model(MODEL_PATH)
            else:
                loading_label.config(text="⚙️ Training GRU model (this may take a while)...", foreground="blue")
                model = train_model(X, y, model_path=MODEL_PATH, epochs=10, batch_size=32, verbose=1)

            loading_label.config(text="🔮 Forecasting...", foreground="blue")
            preds = predict_future(model, scaler, scaled_data, time_step=TIME_STEP, future_steps=future_days)

            # Evaluate (optional)
            metrics = evaluate_model(model, scaler, scaled_data, time_step=TIME_STEP) or {}

            # Update GUI in main thread
            root.after(0, update_gui, df, ticker, preds, metrics, loading_label, result_label, plot_frame)
        except Exception as e:
            root.after(0, lambda: (messagebox.showerror("Error", str(e)), loading_label.config(text="")))
    threading.Thread(target=worker, daemon=True).start()

def update_gui(df, ticker, preds, metrics, loading_label, result_label, plot_frame):
    # result text
    txt = f"📈 Predicted Prices (Next {len(preds)} days):\n"
    txt += ", ".join([f"{p:.2f}" for p in preds[:min(10, len(preds))]])
    txt += "\n\n"
    if metrics:
        txt += (f"RMSE: {metrics['rmse']:.2f}  |  MAE: {metrics['mae']:.2f}  |  R2: {metrics['r2']:.4f}\n"
                f"MAPE: {metrics['mape']:.2f}%")
    result_label.config(text=txt, foreground="#006400")
    loading_label.config(text="✅ Done!", foreground="green")

    # Plot last 100 days + forecast
    for w in plot_frame.winfo_children():
        w.destroy()

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(preds))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(df.index[-100:], df['Close'][-100:], label='Historical (Last 100)', color="#2f4f4f")
    ax.plot(future_dates, preds, label=f'Forecast ({len(preds)} days)', linestyle='--', color="#ff4500', marker='o', markersize=4)
    ax.set_title(f"{ticker} - Forecast ({len(preds)} days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # save csv
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': preds})
    csv_name = f"{ticker}_{len(preds)}day_prediction.csv"
    pred_df.to_csv(csv_name, index=False)

def start_gui():
    global root
    root = tk.Tk()
    root.title("Stock Price Predictor (GRU) - 5 Day Forecast")
    root.geometry("920x700")
    root.configure(bg="#eaf6f6")

    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    style.configure('TLabel', background="#eaf6f6", font=('Segoe UI', 12))
    style.configure('TEntry', font=('Segoe UI', 12))
    style.configure('TButton', font=('Segoe UI', 12, 'bold'))

    top_frame = ttk.Frame(root, padding="12 12 12 6")
    top_frame.pack(fill='x', padx=16, pady=10)

    ttk.Label(top_frame, text="🔎 Enter Stock Ticker (e.g., AAPL, INFY.NS):", font=('Segoe UI', 12, 'bold')).pack(side='left')
    ticker_entry = ttk.Entry(top_frame, width=20)
    ticker_entry.pack(side='left', padx=8)
    ttk.Label(top_frame, text="Forecast days:").pack(side='left', padx=(12,0))
    days_entry = ttk.Entry(top_frame, width=6)
    days_entry.insert(0, str(DEFAULT_FUTURE_DAYS))
    days_entry.pack(side='left', padx=6)

    loading_label = ttk.Label(root, text="", font=('Segoe UI', 11, 'italic'))
    loading_label.pack(pady=(6,2))
    result_label = ttk.Label(root, text="", font=('Segoe UI', 12), wraplength=880, justify='left')
    result_label.pack(pady=6)

    plot_frame = tk.Frame(root, bg="#ffffff", relief="ridge", bd=2)
    plot_frame.pack(fill='both', expand=True, padx=16, pady=12)

    def on_predict():
        ticker = ticker_entry.get().strip().upper()
        if not ticker:
            messagebox.showwarning("Input Error", "Please enter a valid stock ticker.")
            return
        try:
            future_days = int(days_entry.get().strip())
            if future_days <= 0:
                future_days = DEFAULT_FUTURE_DAYS
        except:
            future_days = DEFAULT_FUTURE_DAYS
        # start threaded prediction
        start_button.config(state='disabled')
        run_prediction_threaded(ticker, future_days, loading_label, result_label, plot_frame)
        start_button.config(state='normal')

    start_button = ttk.Button(top_frame, text="🔍 Predict", command=on_predict)
    start_button.pack(side='left', padx=8)

    close_button = ttk.Button(root, text="❌ Close", command=lambda: root.destroy())
    close_button.pack(pady=8)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

if __name__ == "__main__":
    start_gui()
