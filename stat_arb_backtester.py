"""
Module: stat_arb_backtester
Execution simulator incorporating borrow fees, broker execution delay,
proportional transaction costs, and dynamic stop-losses.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_loss_z: float = 4.0
    half_life_min: float = 2.0 / 252.0  # 2 days in years
    half_life_max: float = 30.0 / 252.0  # 30 days in years
    slippage_bps: float = 3.0  # 3 bps per side execution drag
    borrow_rate_annual: float = 0.01  # 100 bps short borrow fee
    capital_allocation: float = 1_000_000.0


@dataclass
class PerformanceMetrics:
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int


class PairsBacktester:
    def __init__(self, config: BacktestConfig) -> None:
        self.cfg = config

    def run(
        self,
        prices_y: np.ndarray,
        prices_x: np.ndarray,
        spread_series: np.ndarray,
        betas: np.ndarray,
        ou_half_lives: np.ndarray,
        z_scores: np.ndarray,
    ) -> tuple[pd.DataFrame, PerformanceMetrics]:
        n = len(prices_y)
        # Position states: +1: Long Y / Short X; -1: Short Y / Long X; 0: Flat
        position = np.zeros(n, dtype=np.int8)
        units_y = np.zeros(n, dtype=np.float64)
        units_x = np.zeros(n, dtype=np.float64)

        cash = np.zeros(n, dtype=np.float64)
        portfolio_val = np.zeros(n, dtype=np.float64)
        cash[0] = self.cfg.capital_allocation
        portfolio_val[0] = self.cfg.capital_allocation

        dt_year = 1.0 / 252.0
        daily_borrow_factor = self.cfg.borrow_rate_annual * dt_year
        cost_multiplier = self.cfg.slippage_bps * 1e-4

        trade_count = 0
        winning_trades = 0
        in_trade = False
        trade_entry_val = 0.0

        for t in range(1, n):
            current_pos = position[t - 1]
            z = z_scores[t - 1]
            tau = ou_half_lives[t - 1]
            beta = betas[t - 1]

            desired_pos = current_pos

            # Trade execution logic governed by OU half-life and z-score bounds
            is_valid_ou = self.cfg.half_life_min <= tau <= self.cfg.half_life_max

            if current_pos == 0:
                if is_valid_ou:
                    if z <= -self.cfg.entry_z:
                        desired_pos = 1  # Long Spread: Long Y, Short X
                    elif z >= self.cfg.entry_z:
                        desired_pos = -1  # Short Spread: Short Y, Long X
            elif current_pos == 1:
                if z >= -self.cfg.exit_z or z <= -self.cfg.stop_loss_z or not is_valid_ou:
                    desired_pos = 0
            elif current_pos == -1:
                if z <= self.cfg.exit_z or z >= self.cfg.stop_loss_z or not is_valid_ou:
                    desired_pos = 0

            position[t] = desired_pos

            # Determine trading friction & portfolio adjustments
            if desired_pos != current_pos:
                trade_count += 1
                py = prices_y[t]
                px = prices_x[t]

                # Close existing positions if altering exposure
                if current_pos != 0:
                    exit_val = (
                        units_y[t - 1] * py
                        + units_x[t - 1] * px
                        - cost_multiplier * (np.abs(units_y[t - 1]) * py + np.abs(units_x[t - 1]) * px)
                    )
                    trade_pnl = exit_val - trade_entry_val
                    if trade_pnl > 0:
                        winning_trades += 1
                    in_trade = False

                if desired_pos != 0:
                    # Allocate capital uniformly: 50% leg Y, 50% leg X
                    notional = portfolio_val[t - 1] * 0.5
                    u_y = (desired_pos * notional) / py
                    u_x = (-desired_pos * notional * beta) / px

                    # Transaction fee drag on entry
                    fee = cost_multiplier * (np.abs(u_y) * py + np.abs(u_x) * px)
                    cash[t] = cash[t - 1] - (u_y * py + u_x * px) - fee

                    units_y[t] = u_y
                    units_x[t] = u_x
                    trade_entry_val = u_y * py + u_x * px
                    in_trade = True
                else:
                    cash[t] = (
                        cash[t - 1]
                        + units_y[t - 1] * py
                        + units_x[t - 1] * px
                        - cost_multiplier
                        * (np.abs(units_y[t - 1]) * py + np.abs(units_x[t - 1]) * px)
                    )
                    units_y[t] = 0.0
                    units_x[t] = 0.0
            else:
                # Retain open holdings
                units_y[t] = units_y[t - 1]
                units_x[t] = units_x[t - 1]
                cash[t] = cash[t - 1]

                # Deduct borrow fee on short equity legs
                short_liability = 0.0
                if units_y[t] < 0:
                    short_liability += np.abs(units_y[t]) * prices_y[t]
                if units_x[t] < 0:
                    short_liability += np.abs(units_x[t]) * prices_x[t]

                cash[t] -= short_liability * daily_borrow_factor

            portfolio_val[t] = cash[t] + units_y[t] * prices_y[t] + units_x[t] * prices_x[t]

        # Calculate performance statistics
        returns = np.diff(portfolio_val) / portfolio_val[:-1]
        total_ret = (portfolio_val[-1] - portfolio_val[0]) / portfolio_val[0]
        n_years = n / 252.0
        cagr = (portfolio_val[-1] / portfolio_val[0]) ** (1.0 / n_years) - 1.0
        vol = float(np.std(returns) * np.sqrt(252.0))
        sharpe = float((cagr - 0.02) / vol) if vol > 1e-6 else 0.0

        cum_max = np.maximum.accumulate(portfolio_val)
        drawdowns = (portfolio_val - cum_max) / cum_max
        max_dd = float(np.min(drawdowns))
        win_rate = float(winning_trades / (trade_count // 2)) if trade_count >= 2 else 0.0

        metrics = PerformanceMetrics(
            total_return=total_ret,
            cagr=cagr,
            annualized_volatility=vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=trade_count // 2,
        )

        df_out = pd.DataFrame(
            {
                "PortfolioValue": portfolio_val,
                "Position": position,
                "Spread": spread_series,
                "ZScore": z_scores,
                "UnitsY": units_y,
                "UnitsX": units_x,
                "Beta": betas,
                "HalfLife": ou_half_lives,
            }
        )

        return df_out, metrics
    