import pandas as pd
import numpy as np

def rolling_statistics(series, window=30):
    """Compute rolling mean, std, skewness, kurtosis."""
    stats = pd.DataFrame()
    stats['Rolling_Mean'] = series.rolling(window).mean()
    stats['Rolling_Std'] = series.rolling(window).std()
    stats['Rolling_Skew'] = series.rolling(window).skew()
    stats['Rolling_Kurt'] = series.rolling(window).kurt()
    return stats

def correlation_matrix(close_prices_df):
    """Pearson correlation using NumPy."""
    returns = close_prices_df.pct_change().dropna()
    return returns.corr()

def detect_trend(series, short_window=20, long_window=50):
    """Detect uptrend/downtrend using MA crossover."""
    short_ma = series.rolling(short_window).mean()
    long_ma = series.rolling(long_window).mean()
    
    signals = pd.Series(index=series.index, dtype=float)
    signals[short_ma > long_ma] = 1    # Uptrend
    signals[short_ma < long_ma] = -1   # Downtrend
    return signals

def autocorrelation(series, lags=20):
    """Compute autocorrelation for given lags using NumPy."""
    arr = series.dropna().values
    n = len(arr)
    mean = arr.mean()
    var = ((arr - mean) ** 2).sum()
    
    acf = []
    for lag in range(lags + 1):
        cov = ((arr[:n-lag] - mean) * (arr[lag:] - mean)).sum()
        acf.append(cov / var)
    return acf