# Predictive Financial Analytics Dashboard

A financial analytics system built with Python, Pandas, and NumPy that analyzes historical stock market data, extracts advanced insights, and performs predictive analysis — without using any ML libraries.

## Features

- Historical stock data loading and cleaning
- Feature engineering: MA, EMA, RSI, Bollinger Bands, Momentum
- Time-series analysis: rolling stats, correlation matrix
- Predictive modeling: linear regression and MA forecasting (pure NumPy)
- Portfolio optimization via Monte Carlo simulation
- 10 auto-generated visualizations

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/financial_dashboard.git
cd financial_dashboard
```

### 2. Create virtual environment

```bash
python -m venv virtualEnviroment
virtualEnviroment\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download stock data

```bash
python download_data.py
```

### 5. Run the dashboard

```bash
python main.py
```

## Output

All results are saved to the `output/` folder:

- `output/plots/` — 10 generated charts
- `output/correlation_matrix.csv`
- `output/monte_carlo_results.csv`
- `output/{TICKER}_features.csv`

## Tech Stack

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- yfinance (data download only)
