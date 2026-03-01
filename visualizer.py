import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Output directory ────────────────────────────────────────────────────────
PLOT_DIR = "output/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. PRICE & MOVING AVERAGES
# ════════════════════════════════════════════════════════════════════════════
def plot_price_and_ma(df, ticker):
    """
    Plot Close price with MA20, MA50, EMA20 and Bollinger Bands.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df['Close'],   label='Close',  color='#1f77b4', linewidth=1.2)
    ax.plot(df.index, df['MA_20'],   label='MA 20',  color='orange',  linewidth=1, linestyle='--')
    ax.plot(df.index, df['MA_50'],   label='MA 50',  color='red',     linewidth=1, linestyle='--')
    ax.plot(df.index, df['EMA_20'],  label='EMA 20', color='green',   linewidth=1, linestyle=':')

    if 'BB_Upper' in df.columns:
        ax.fill_between(df.index, df['BB_Lower'], df['BB_Upper'],
                        alpha=0.1, color='purple', label='Bollinger Bands')
        ax.plot(df.index, df['BB_Upper'], color='purple', linewidth=0.6, linestyle='-.')
        ax.plot(df.index, df['BB_Lower'], color='purple', linewidth=0.6, linestyle='-.')

    ax.set_title(f'{ticker} — Price & Moving Averages', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/{ticker}_price_ma.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 2. RSI
# ════════════════════════════════════════════════════════════════════════════
def plot_rsi(df, ticker):
    """
    Plot RSI with overbought (70) and oversold (30) lines.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df.index, df['RSI_14'], color='#e377c2', linewidth=1.2, label='RSI 14')
    ax.axhline(70, color='red',   linestyle='--', linewidth=1, label='Overbought (70)')
    ax.axhline(30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
    ax.fill_between(df.index, df['RSI_14'], 70,
                    where=(df['RSI_14'] >= 70), alpha=0.2, color='red')
    ax.fill_between(df.index, df['RSI_14'], 30,
                    where=(df['RSI_14'] <= 30), alpha=0.2, color='green')

    ax.set_title(f'{ticker} — RSI (14)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/{ticker}_rsi.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 3. DAILY RETURNS DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════
def plot_returns_distribution(df, ticker):
    """
    Histogram of daily returns with mean and ±1 std lines.
    """
    returns = df['Daily_Return'].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(returns, bins=80, color='#1f77b4', edgecolor='white', alpha=0.8)

    mean = returns.mean()
    std  = returns.std()
    ax.axvline(mean,       color='red',    linestyle='--', linewidth=1.5, label=f'Mean: {mean:.4f}')
    ax.axvline(mean + std, color='orange', linestyle='--', linewidth=1,   label=f'+1 Std: {mean+std:.4f}')
    ax.axvline(mean - std, color='orange', linestyle='--', linewidth=1,   label=f'-1 Std: {mean-std:.4f}')

    ax.set_title(f'{ticker} — Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/{ticker}_returns_dist.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 4. CUMULATIVE RETURNS (ALL STOCKS)
# ════════════════════════════════════════════════════════════════════════════
def plot_cumulative_returns(stocks):
    """
    Compare cumulative returns of all stocks on one chart.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for (ticker, df), color in zip(stocks.items(), colors):
        cum_ret = df['Cumulative_Return'].dropna()
        ax.plot(cum_ret.index, cum_ret * 100, label=ticker, color=color, linewidth=1.5)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Cumulative Returns — All Stocks', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/cumulative_returns.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 5. VOLATILITY OVER TIME
# ════════════════════════════════════════════════════════════════════════════
def plot_volatility(stocks):
    """
    Plot rolling 30-day annualized volatility for all stocks.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for (ticker, df), color in zip(stocks.items(), colors):
        vol = df['Volatility_30'].dropna()
        ax.plot(vol.index, vol * 100, label=ticker, color=color, linewidth=1.2)

    ax.set_title('Rolling 30-Day Annualized Volatility', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/volatility.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 6. CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(corr_matrix):
    """
    Display the stock correlation matrix as a colour heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    tickers = corr_matrix.columns.tolist()
    data    = corr_matrix.values

    im = ax.imshow(data, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Correlation')

    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_yticklabels(tickers, fontsize=11)

    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{data[i, j]:.2f}",
                    ha='center', va='center', fontsize=10,
                    color='black' if abs(data[i, j]) < 0.8 else 'white')

    ax.set_title('Stock Return Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f"{PLOT_DIR}/correlation_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 7. MONTE CARLO EFFICIENT FRONTIER
# ════════════════════════════════════════════════════════════════════════════
def plot_efficient_frontier(sim_results, optimal):
    """
    Scatter plot of simulated portfolios coloured by Sharpe ratio,
    with the optimal portfolio highlighted.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    sc = ax.scatter(
        sim_results['Risk']   * 100,
        sim_results['Return'] * 100,
        c=sim_results['Sharpe'],
        cmap='viridis',
        alpha=0.4,
        s=5
    )
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')

    # Highlight optimal portfolio
    ax.scatter(
        optimal['Risk']   * 100,
        optimal['Return'] * 100,
        color='red', marker='*', s=300, zorder=5, label='Optimal (Max Sharpe)'
    )
    ax.annotate(
        f"  Sharpe: {optimal['Sharpe']:.2f}\n  Return: {optimal['Return']*100:.1f}%\n  Risk: {optimal['Risk']*100:.1f}%",
        xy=(optimal['Risk'] * 100, optimal['Return'] * 100),
        fontsize=9, color='red'
    )

    ax.set_title('Monte Carlo Efficient Frontier (10,000 Portfolios)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Risk / Std Dev (%)')
    ax.set_ylabel('Expected Annual Return (%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/efficient_frontier.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 8. PRICE FORECAST
# ════════════════════════════════════════════════════════════════════════════
def plot_forecast(df, ticker, forecast_values, forecast_days=10):
    """
    Plot last 60 days of actual Close price + the MA forecast.
    """
    close = df['Close'].dropna()
    last_60 = close.iloc[-60:]

    # Build future date index (business days)
    last_date    = close.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=forecast_days + 1)[1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(last_60.index, last_60.values,
            label='Actual (last 60 days)', color='#1f77b4', linewidth=1.5)
    ax.plot(future_dates, forecast_values,
            label=f'{forecast_days}-day MA Forecast', color='orange',
            linewidth=1.8, linestyle='--', marker='o', markersize=4)

    ax.axvline(last_date, color='grey', linestyle=':', linewidth=1)
    ax.set_title(f'{ticker} — Short-Term Price Forecast', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/{ticker}_forecast.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 9. SHARPE RATIO BAR CHART
# ════════════════════════════════════════════════════════════════════════════
def plot_sharpe_comparison(sharpe_dict):
    """
    Bar chart comparing Sharpe ratios across all tickers.
    """
    tickers = list(sharpe_dict.keys())
    values  = list(sharpe_dict.values())
    colors  = ['#2ca02c' if v >= 1 else '#1f77b4' if v >= 0.5 else '#d62728' for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(tickers, values, color=colors, edgecolor='white', width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(1.0, color='green', linestyle='--', linewidth=1, label='Good (≥ 1.0)')
    ax.axhline(0.5, color='orange', linestyle='--', linewidth=1, label='Acceptable (≥ 0.5)')
    ax.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Stock')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = f"{PLOT_DIR}/sharpe_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 10. RISK vs RETURN SCATTER
# ════════════════════════════════════════════════════════════════════════════
def plot_risk_vs_return(stocks):
    """
    Scatter plot: annualized return vs annualized volatility per stock.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for ticker, df in stocks.items():
        ret = df['Daily_Return'].dropna()
        annual_return = ret.mean() * 252 * 100
        annual_vol    = ret.std()  * np.sqrt(252) * 100
        ax.scatter(annual_vol, annual_return, s=120, zorder=5)
        ax.annotate(f'  {ticker}', xy=(annual_vol, annual_return), fontsize=10)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Risk vs Return (Annualized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Risk — Annualized Volatility (%)')
    ax.set_ylabel('Expected Return (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/risk_vs_return.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION — call this from main.py
# ════════════════════════════════════════════════════════════════════════════
def generate_all_plots(stocks, corr_matrix, sim_results, optimal, sharpe_dict, forecasts):
    """
    Generate and save every plot.

    Parameters
    ----------
    stocks       : dict  {ticker: DataFrame with all features}
    corr_matrix  : DataFrame  correlation matrix
    sim_results  : DataFrame  Monte Carlo simulation results
    optimal      : Series     optimal portfolio row from sim_results
    sharpe_dict  : dict  {ticker: sharpe_ratio}
    forecasts    : dict  {ticker: np.array of forecast values}
    """
    print("\nGenerating plots...")

    # Per-stock plots
    for ticker, df in stocks.items():
        plot_price_and_ma(df, ticker)
        plot_rsi(df, ticker)
        plot_returns_distribution(df, ticker)
        if ticker in forecasts:
            plot_forecast(df, ticker, forecasts[ticker])

    # Multi-stock comparison plots
    plot_cumulative_returns(stocks)
    plot_volatility(stocks)
    plot_risk_vs_return(stocks)
    plot_sharpe_comparison(sharpe_dict)

    # Correlation & portfolio plots
    plot_correlation_heatmap(corr_matrix)
    plot_efficient_frontier(sim_results, optimal)

    print(f"\nAll plots saved to '{PLOT_DIR}/'")