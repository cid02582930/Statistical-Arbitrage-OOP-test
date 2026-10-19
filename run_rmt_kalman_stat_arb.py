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
from spectral_denoising import MarchenkoPasturFilter
from kalman_filter import OnlineKalmanSpreadTracker
from ou_calibration import OrnsteinUhlenbeckEstimator
from stat_arb_backtester import PairsBacktester, BacktestConfig


def generate_synthetic_market_data(
    num_assets: int = 50, num_days: int = 1000, seed: int = 42
) -> tuple[pd.DataFrame, tuple[str, str]]:
    np.random.seed(seed)
    # generate common latent macro factors
    t = num_days
    factor_market = np.cumsum(np.random.normal(0.0002, 0.01, size=t))
    factor_sector = np.cumsum(np.random.normal(0.0, 0.008, size=t))

    prices = {}
    for i in range(num_assets):
        idio = np.cumsum(np.random.normal(0.0, 0.015, size=t))
        # loading on systemic factors
        p = np.exp(1.0 + 0.6 * factor_market + 0.4 * factor_sector + idio)
        prices[f"ASSET_{i:02d}"] = p

    # inject non-stationary cointegrated pair with time-varying beta and OU spread
    time_steps = np.linspace(0, 4.0, t)
    drifting_beta = 1.2 + 0.3 * np.sin(2.0 * np.pi * time_steps / 3.0)
    drifting_alpha = 0.5 + 0.1 * time_steps

    # generate synthetic OU spread process: dS = 12 * (0 - S) dt + 0.2 dW
    spread = np.zeros(t)
    theta_true = 12.0
    sigma_true = 0.25
    dt = 1.0 / 252.0

    for step in range(1, t):
        dW = np.random.normal(0, np.sqrt(dt))
        spread[step] = (
            spread[step - 1]
            - theta_true * spread[step - 1] * dt
            + sigma_true * dW
        )

    # asset A depends dynamically on Asset B + synthetic spread
    p_b = prices["ASSET_00"]
    p_a = drifting_alpha + drifting_beta * p_b + spread
    prices["ASSET_PA"] = p_a
    prices["ASSET_PB"] = p_b

    df_prices = pd.DataFrame(prices)
    return df_prices, ("ASSET_PA", "ASSET_PB")


def main() -> None:
    print("[1/5] Synthesizing Multi-Asset Universe & Structural Pairs...")
    num_assets = 40
    lookback = 1000
    df_prices, target_pair = generate_synthetic_market_data(
        num_assets=num_assets, num_days=lookback, seed=101
    )
    asset_y, asset_x = target_pair

    # calculate returns for spectral de-noising verification
    returns = df_prices.pct_change().dropna()
    t_samples, n_dims = returns.shape
    q_ratio = float(t_samples) / float(n_dims)

    print(f"[2/5] Running Marchenko-Pastur Spectral Filter (Q = {q_ratio:.2f})...")
    raw_corr = returns.corr().to_numpy()
    mp_filter = MarchenkoPasturFilter(q_factor=q_ratio)
    cleaned_corr = mp_filter.denoise_correlation_matrix(raw_corr)

    # spectral diagnostic verify
    orig_evals = np.linalg.eigvalsh(raw_corr)
    cleaned_evals = np.linalg.eigvalsh(cleaned_corr)
    bounds = mp_filter.compute_theoretical_bounds(sigma_sq=1.0)
    print(f"      MP Threshold (lambda_max): {bounds.lambda_max:.4f}")
    print(f"      Raw Max Eigenvalue: {orig_evals.max():.4f}")
    print(f"      Preserved Signal Eigenvalues: {int(np.sum(orig_evals > bounds.lambda_max))}")
    print(f"      Denoised Condition Number: {np.linalg.cond(cleaned_corr):.4f}")

    print(f"[3/5] Initializing Online Kalman State-Space Model on ({asset_y}, {asset_x})...")
    py_series = df_prices[asset_y].to_numpy()
    px_series = df_prices[asset_x].to_numpy()

    kalman = OnlineKalmanSpreadTracker(
        delta_noise=1e-5, obs_variance=1e-3, initial_beta=1.0
    )
    ou_estimator = OrnsteinUhlenbeckEstimator(dt=1.0 / 252.0)

    spread_history = np.zeros(lookback)
    beta_history = np.zeros(lookback)
    alpha_history = np.zeros(lookback)
    half_life_history = np.zeros(lookback)
    z_score_history = np.zeros(lookback)

    # 40-day rolling calibration window for running z-scores & OU parameters
    rolling_ou_window = 40

    for t in range(lookback):
        spread_t, alpha_t, beta_t = kalman.step(py_series[t], px_series[t])
        spread_history[t] = spread_t
        beta_history[t] = beta_t
        alpha_history[t] = alpha_t

        if t >= rolling_ou_window:
            local_window = spread_history[t - rolling_ou_window : t]
            ou_params = ou_estimator.fit(local_window)
            if ou_params.is_valid:
                half_life_history[t] = ou_params.half_life
            else:
                half_life_history[t] = np.inf

            local_mean = np.mean(local_window)
            local_std = np.std(local_window)
            z_score_history[t] = (
                (spread_t - local_mean) / local_std if local_std > 1e-6 else 0.0
            )
        else:
            half_life_history[t] = np.inf
            z_score_history[t] = 0.0

    print("[4/5] Executing Event-Driven Backtester with Microstructure Friction...")
    config = BacktestConfig(
        entry_z=1.75,
        exit_z=0.25,
        stop_loss_z=3.5,
        slippage_bps=3.0,
        borrow_rate_annual=0.015,
        capital_allocation=1_000_000.0,
    )

    backtester = PairsBacktester(config)
    results_df, metrics = backtester.run(
        prices_y=py_series,
        prices_x=px_series,
        spread_series=spread_history,
        betas=beta_history,
        ou_half_lives=half_life_history,
        z_scores=z_score_history,
    )

    print("\n" + "=" * 60)
    print("                 PERFORMANCE METRICS REPORT                 ")
    print("=" * 60)
    print(f" Total return:              {metrics.total_return * 100.0:>10.2f} %")
    print(f" CAGR:                      {metrics.cagr * 100.0:>10.2f} %")
    print(f" Annualized volatility:     {metrics.annualized_volatility * 100.0:>10.2f} %")
    print(f" Sharpe ratio (Rf=2.0%):    {metrics.sharpe_ratio:>10.2f}")
    print(f" Maximum drawdown:          {metrics.max_drawdown * 100.0:>10.2f} %")
    print(f" Total completed trades:    {metrics.total_trades:>10d}")
    print(f" Realized win rate:         {metrics.win_rate * 100.0:>10.2f} %")
    print("=" * 60)


if __name__ == "__main__":
    main()