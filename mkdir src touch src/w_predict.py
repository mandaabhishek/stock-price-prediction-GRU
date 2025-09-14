# src/predict.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def predict_future(model, scaler, scaled_data, time_step=60, future_steps=5):
    """
    Generate future_steps predictions using iterative forecasting.
    Returns unscaled numpy array shape (future_steps,).
    """
    buffer = list(scaled_data[-time_step:].flatten())
    preds_scaled = []
    for _ in range(future_steps):
        x_input = np.array(buffer[-time_step:]).reshape(1, time_step, 1)
        pred = model.predict(x_input, verbose=0)
        next_val = float(pred[0][0])
        buffer.append(next_val)
        preds_scaled.append(next_val)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds

def evaluate_model(model, scaler, scaled_data, time_step=60):
    """
    Evaluate model on last 20% of data (if possible).
    Returns metrics dict or None if not enough test samples.
    """
    split = int(len(scaled_data) * 0.8)
    test_actual = scaled_data[split:]
    X_test, y_test = None, None
    # If test_actual too small -> return None
    if len(test_actual) <= time_step + 1:
        return None
    # create X_test, y_test relative to test_actual
    X_test = []
    y_test = []
    for i in range(len(test_actual) - time_step - 1):
        X_test.append(test_actual[i:(i + time_step), 0])
        y_test.append(test_actual[i + time_step, 0])
    X_test = np.array(X_test).reshape(-1, time_step, 1)
    y_test = np.array(y_test)
    preds = model.predict(X_test, verbose=0)
    preds_unscaled = scaler.inverse_transform(preds)
    y_true_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = mean_squared_error(y_true_unscaled, preds_unscaled, squared=False)
    mae = mean_absolute_error(y_true_unscaled, preds_unscaled)
    mse = mean_squared_error(y_true_unscaled, preds_unscaled)
    r2 = r2_score(y_true_unscaled, preds_unscaled)
    mape = np.mean(np.abs((y_true_unscaled - preds_unscaled) / y_true_unscaled)) * 100
    return {"rmse": rmse, "mae": mae, "mse": mse, "r2": r2, "mape": mape}
