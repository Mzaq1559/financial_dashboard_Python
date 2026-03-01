import numpy as np
import pandas as pd

def portfolio_stats(weights, mean_returns, cov_matrix):
    """Calculate expected return and risk for a portfolio."""
    weights = np.array(weights)
    expected_return = np.dot(weights, mean_returns) * 252
    variance = weights @ cov_matrix @ weights.T * 252
    std_dev = np.sqrt(variance)
    sharpe = expected_return / std_dev  # assuming risk-free = 0
    return expected_return, std_dev, sharpe

def monte_carlo_simulation(close_prices, n_portfolios=10000, risk_free=0.02):
    """Simulate random portfolios and find the efficient frontier."""
    returns = close_prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov().values
    n_assets = len(close_prices.columns)
    
    results = np.zeros((3, n_portfolios))
    weights_record = []
    
    for i in range(n_portfolios):
        w = np.random.random(n_assets)
        w /= w.sum()
        weights_record.append(w)
        
        ret, std, sharpe = portfolio_stats(w, mean_returns.values, cov_matrix)
        results[0, i] = ret
        results[1, i] = std
        results[2, i] = sharpe
    
    results_df = pd.DataFrame(results.T, columns=['Return', 'Risk', 'Sharpe'])
    results_df['Weights'] = weights_record
    
    return results_df

def optimal_portfolio(simulation_results):
    """Find the portfolio with the highest Sharpe Ratio."""
    idx = simulation_results['Sharpe'].idxmax()
    return simulation_results.loc[idx]

def minimum_variance_portfolio(simulation_results):
    idx = simulation_results['Risk'].idxmin()
    return simulation_results.loc[idx]