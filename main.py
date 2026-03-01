import os
import pandas as pd
import numpy as np
from data_loader import load_stock_data, clean_data, align_stocks
from features import add_all_features, sharpe_ratio
from time_series import rolling_statistics, correlation_matrix, detect_trend
from predictor import predict_linear, moving_average_forecast, evaluate_model
from portfolio import monte_carlo_simulation, optimal_portfolio
from visualizer import generate_all_plots

# Create output folders
os.makedirs("output", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)

# ── 1. Load & Clean Data ─────────────────────────────────────────────────────
print("=" * 55)
print("  STEP 1: Loading & Cleaning Data")
print("=" * 55)
stocks = load_stock_data("data/")
for ticker in stocks:
    stocks[ticker] = clean_data(stocks[ticker])

# ── 2. Feature Engineering ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 2: Engineering Features")
print("=" * 55)
sharpe_dict = {}
for ticker in stocks:
    stocks[ticker] = add_all_features(stocks[ticker])
    sr = sharpe_ratio(stocks[ticker]['Close'])
    sharpe_dict[ticker] = sr
    print(f"  {ticker} Sharpe Ratio: {sr:.4f}")

# ── 3. Time-Series Analysis ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 3: Time-Series & Correlation Analysis")
print("=" * 55)
close_prices = align_stocks(stocks)
corr = correlation_matrix(close_prices)
print("\nCorrelation Matrix:")
print(corr.round(3))

# ── 4. Prediction ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 4: Predictive Modeling")
print("=" * 55)
forecasts = {}
for ticker in stocks:
    print(f"\n  Evaluating model for {ticker}...")
    metrics = evaluate_model(stocks[ticker]['Close'])
    print(f"    RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
    forecast = moving_average_forecast(stocks[ticker]['Close'], window=20, forecast_days=10)
    forecasts[ticker] = forecast
    print(f"    10-day Forecast: {forecast.round(2)}")

# ── 5. Portfolio Optimization ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 5: Portfolio Optimization (Monte Carlo)")
print("=" * 55)
sim = monte_carlo_simulation(close_prices, n_portfolios=10000)
best = optimal_portfolio(sim)
tickers = close_prices.columns.tolist()

print(f"\n  Optimal Portfolio (Max Sharpe Ratio):")
print(f"    Expected Return : {best['Return']:.2%}")
print(f"    Risk (Std Dev)  : {best['Risk']:.2%}")
print(f"    Sharpe Ratio    : {best['Sharpe']:.4f}")
print("    Weights:")
for t, w in zip(tickers, best['Weights']):
    print(f"      {t}: {w:.2%}")

# ── 6. Save CSV Results ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 6: Saving Results")
print("=" * 55)
sim[['Return', 'Risk', 'Sharpe']].to_csv("output/monte_carlo_results.csv", index=False)
corr.to_csv("output/correlation_matrix.csv")
for ticker, df in stocks.items():
    df.to_csv(f"output/{ticker}_features.csv")
print("  CSVs saved to output/")

# ── 7. Generate All Plots ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 7: Generating Visualizations")
print("=" * 55)
generate_all_plots(
    stocks      = stocks,
    corr_matrix = corr,
    sim_results = sim,
    optimal     = best,
    sharpe_dict = sharpe_dict,
    forecasts   = forecasts
)

print("\n" + "=" * 55)
print("  ALL DONE! Check the output/ folder.")
print("=" * 55)