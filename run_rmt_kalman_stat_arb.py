"""
Quantitative pipeline execute:
1. Synthetic generation of an N-asset correlated universe with cointegrated sub-structures.
2. Marchenko-Pastur spectral correlation matrix filtering.
3. Online Kalman estimation of dynamic beta and residual innovation.
4. Continuous-time OU parameter estimation.
5. Backtesting with microstructure frictions and performance diagnostics.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from spectral_denoising import MarchenkoPasturFilter
from stat_arb_backtester import PairsBacktester, BacktestConfig

def generate_synthetic_market_data(num_assets: int = 30, num_days: int = 1000, seed: int = 42):
    np.random.seed(seed)
    dt = 1.0 / 252.0

    # generate common latent macro factors
    t = num_days
    factor_market = np.exp(np.cumsum(np.random.normal(0.0003, 0.012, size=t)))
    factor_sector = np.exp(np.cumsum(np.random.normal(0.0001, 0.009, size=t)))

    prices = {}
    for i in range(num_assets):
        beta_m = np.random.uniform(0.6, 1.4)
        beta_s = np.random.uniform(0.3, 1.1)
        idio = np.exp(np.cumsum(np.random.normal(0.0, 0.012, size=num_days)))
        prices[f"TICKER_{i:02d}"] = 40.0 * (factor_market ** beta_m) * (factor_sector ** beta_s) * idio

    # cointegrated pairs inject
    pair_anchors = [(0, 1 ), (2, 3), (4, 5), (6, 7), (8, 9)]
    for idx_y, idx_x in pair_anchors:
        theta = np.random.uniform(12.0, 24.0) # fast half-life: ~7 to 15 days
        sigma_ou = 0.18
        spread = np.zeros(num_days)

        for t in range(1, num_days):
            spread[t] = spread[t - 1] - theta * spread[t - 1] * dt + sigma_ou * np.random.normal(0, np.sqrt(dt))

            # log space pair link
            px_val = prices[f"TICKER_{idx_x:02d}"]
            prices[f"TICKER_{idx_y:02d}"] = np.exp(0.3 + 1.15 * np.log(px_val) + spread)

    return pd.DataFrame(prices), pair_anchors

def run_pipeline() -> None:
    print("[1/4] Generating Multi-Asset Synthetic Market (N=30, T=1000bars)...")
    df_prices, true_pairs = generate_synthetic_market_data(num_assets=30, num_days=1000, seed=123)

    print("[2/4] Spectral Correlation Matrix Filtering via RMT...")
    log_returns = np.log(df_prices).diff().dropna()
    c_raw = log_returns.corr().to_numpy()
    q_ratio = float(len(log_returns)) / float(df_prices.shape[1])

    mp_filter = MarchenkoPasturFilter(q_factor=q_ratio)
    cleaned_corr = mp_filter.denoise(c_raw)

    print("[3/4] Screening Cointegrated Pairs via De-noised Metric Distance...")
    # used distance matrix D = sqrt(2*(1 - c_clean))
    dist_matrix = np.sqrt(np.maximum(2.0 * (1.0 - cleaned_corr), 0.0))
    np.fill_diagonal(dist_matrix, np.inf)

    # identify pairs with highest correlation in de-noised space
    screened_pairs = []
    tickers = list(df_prices.columns)
    visited = set()

    for i in range(len(tickers)):
        if i in visited:
            continue
        best_j = int(np.argmin(dist_matrix[i]))
        if best_j not in visited and dist_matrix[i, best_j] < 0.65: # find high correlation
            # validate stationarity via ADF test on OLS log-price residuals
            p_y = np.log(df_prices[tickers[i]].to_numpy())
            p_x = np.log(df_prices[tickers[best_j]].to_numpy())
            beta_est = np.cov(p_y, p_x)[0, 1] / np.var(p_x)
            resids = p_y - beta_est * p_x
            adf_stat, adf_pvalue, *_ = adfuller(resids)

            if adf_pvalue < 0.05: # stationary verified
                screened_pairs.append((tickers[i], tickers[best_j]))
                visited.add(i)
                visited.add(best_j)

    print(f"     Verified Cointegrated Basket: {screened_pairs}")

    print("[4/4] Running Multi-Pair Execution Engine with Realistic Friction...")
    cfg = BacktestConfig(
        entry_z=1.80,
        exit_z=0.20,
        stop_loss_z=3.50,
        slippage_bps=3.0,
        borrow_rate_annual=0.015,
        capital_allocation=1_000_000.0,
    )
    engine = PairsBacktester(config=cfg)
    wealth_df, metrics = engine.run(df_prices, screened_pairs)
       
    print("\n" + "=" * 60)
    print("                 PERFORMANCE METRICS REPORT                 ")
    print("=" * 60)
    print(f" Total Return:              {metrics['Total Return'] * 100.0:>10.2f} %")
    print(f" CAGR:                      {metrics['CAGR'] * 100.0:>10.2f} %")
    print(f" Annualized Volatility:     {metrics['Annualized Volatility'] * 100.0:>10.2f} %")
    print(f" Sharpe Ratio (Rf=2.0%):    {metrics['Sharpe Ratio']:>10.2f}")
    print(f" Maximum Drawdown:          {metrics['Max Drawdown'] * 100.0:>10.2f} %")
    print(f" Total Completed Trades:    {metrics['Completed Trades']:>10d}")
    print(f" Realized Win Rate:         {metrics['Win Rate'] * 100.0:>10.2f} %")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()