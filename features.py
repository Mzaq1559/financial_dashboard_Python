import pandas as pd
import numpy as np

def moving_average(series, window):
    return series.rolling(window=window).mean()

def exponential_moving_average(series, span):
    return series.ewm(span=span, adjust=False).mean()

def daily_returns(series):
    return series.pct_change()

def cumulative_returns(series):
    returns = daily_returns(series)
    return (1 + returns).cumprod() - 1

def volatility(series, window=30):
    """Annualized rolling volatility."""
    return daily_returns(series).rolling(window).std() * np.sqrt(252)

def sharpe_ratio(series, risk_free_rate=0.02):
    """Annualized Sharpe Ratio."""
    ret = daily_returns(series).dropna()
    excess = ret - risk_free_rate / 252
    return (excess.mean() / excess.std()) * np.sqrt(252)

def bollinger_bands(series, window=20):
    ma = moving_average(series, window)
    std = series.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return upper, ma, lower

def rsi(series, period=14):
    """Relative Strength Index using NumPy."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def momentum(series, period=10):
    """Price momentum: current price / price n periods ago."""
    return series / series.shift(period) - 1

def add_all_features(df):
    """Add all features to a stock DataFrame."""
    close = df['Close']
    
    df['MA_20'] = moving_average(close, 20)
    df['MA_50'] = moving_average(close, 50)
    df['EMA_20'] = exponential_moving_average(close, 20)
    df['Daily_Return'] = daily_returns(close)
    df['Cumulative_Return'] = cumulative_returns(close)
    df['Volatility_30'] = volatility(close, 30)
    df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = bollinger_bands(close)
    df['RSI_14'] = rsi(close)
    df['Momentum_10'] = momentum(close, 10)
    
    return df