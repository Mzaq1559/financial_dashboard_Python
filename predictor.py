import numpy as np
import pandas as pd

def linear_regression_numpy(x, y):
    """Ordinary Least Squares using NumPy linear algebra."""
    X = np.column_stack([np.ones(len(x)), x])
    # OLS formula: β = (X'X)^-1 X'y
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return beta  # [intercept, slope]

def predict_linear(series, forecast_days=30):
    """Fit linear regression on close prices and forecast."""
    arr = series.dropna().values
    x = np.arange(len(arr))
    
    beta = linear_regression_numpy(x, arr)
    intercept, slope = beta
    
    # Predict future
    future_x = np.arange(len(arr), len(arr) + forecast_days)
    predictions = intercept + slope * future_x
    
    # In-sample predictions
    in_sample = intercept + slope * x
    
    return in_sample, predictions

def moving_average_forecast(series, window=20, forecast_days=10):
    """Forecast using rolling average of last `window` days."""
    arr = series.dropna().values
    last_window = arr[-window:]
    forecast = []
    
    window_data = list(last_window)
    for _ in range(forecast_days):
        next_val = np.mean(window_data[-window:])
        forecast.append(next_val)
        window_data.append(next_val)
    
    return np.array(forecast)

def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

def rmse(actual, predicted):
    return np.sqrt(mse(actual, predicted))

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def evaluate_model(series, window=20):
    """Train/test split and evaluate MA forecast."""
    arr = series.dropna().values
    split = int(len(arr) * 0.8)
    
    train, test = arr[:split], arr[split:]
    
    # Use MA on train to predict test length
    forecasts = []
    window_data = list(train)
    for _ in range(len(test)):
        pred = np.mean(window_data[-window:])
        forecasts.append(pred)
        window_data.append(pred)
    
    forecasts = np.array(forecasts)
    
    return {
        "MSE": mse(test, forecasts),
        "RMSE": rmse(test, forecasts),
        "MAE": mae(test, forecasts)
    }