import pandas as pd
import numpy as np
import os


def load_stock_data(data_dir="data/"):
    """Load all CSVs and return a dict of DataFrames."""
    stocks = {}

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            ticker = file.replace(".csv", "")
            filepath = os.path.join(data_dir, file)

            # yfinance CSV structure:
            # Row 0: ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']  <- real column names
            # Row 1: ['Ticker', 'AAPL', 'AAPL', ...]                      <- ticker name row (junk)
            # Row 2: ['Date', NaN, NaN, ...]                               <- label row (junk)
            # Row 3+: actual OHLCV data

            # Use row 0 as header, skip rows 1 and 2
            df = pd.read_csv(filepath, header=0, skiprows=[1, 2])

            # Row 0 column 'Price' actually contains the date values
            df.rename(columns={'Price': 'Date'}, inplace=True)

            # Drop any rows where Date cannot be parsed as a real date
            df = df[pd.to_datetime(df['Date'], errors='coerce').notna()]
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            # Convert all columns to numeric (coerce bad values to NaN)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows that are entirely NaN
            df.dropna(how='all', inplace=True)

            stocks[ticker] = df
            print(f"  Loaded {ticker}: {len(df)} rows | columns: {df.columns.tolist()}")

    return stocks


def clean_data(df):
    """
    Handle missing values intelligently:
    - Forward fill then backward fill for small gaps
    - Linear interpolation for any remaining gaps
    """
    df = df.copy()
    df = df.ffill().bfill()
    df = df.interpolate(method='linear')
    return df


def normalize(series):
    """
    Min-Max normalization using NumPy.
    Scales values to range [0, 1].
    """
    arr = series.values.astype(float)
    min_val = arr.min()
    max_val = arr.max()

    if max_val == min_val:
        return np.zeros_like(arr)

    return (arr - min_val) / (max_val - min_val)


def standardize(series):
    """
    Z-score standardization using NumPy.
    Centers data around mean with unit standard deviation.
    """
    arr = series.values.astype(float)
    mean = arr.mean()
    std = arr.std()

    if std == 0:
        return np.zeros_like(arr)

    return (arr - mean) / std


def align_stocks(stocks):
    """
    Align all stocks to a common date index.
    Returns a DataFrame where each column is the Close price of one stock.
    """
    close_prices = pd.DataFrame({
        ticker: df['Close'] for ticker, df in stocks.items()
    })

    # Drop dates where ALL stocks are missing
    close_prices.dropna(how='all', inplace=True)

    # Fill remaining gaps
    close_prices = close_prices.ffill().bfill()

    return close_prices


def get_summary(stocks):
    """
    Print a quick summary of all loaded stocks.
    """
    print("\n" + "=" * 60)
    print(f"{'STOCK':<10} {'ROWS':<8} {'START':<15} {'END':<15} {'MISSING%'}")
    print("=" * 60)

    for ticker, df in stocks.items():
        start = df.index.min().strftime('%Y-%m-%d')
        end = df.index.max().strftime('%Y-%m-%d')
        missing_pct = df['Close'].isna().mean() * 100
        print(f"{ticker:<10} {len(df):<8} {start:<15} {end:<15} {missing_pct:.2f}%")

    print("=" * 60)