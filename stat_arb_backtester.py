"""
Module: stat_arb_backtester
Execution simulator incorporating borrow fees, broker execution delay,
proportional transaction costs, and dynamic stop-losses.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import scipy.optimize as opt
from ou_calibration import OrnsteinUhlenbeckEstimator
from kalman_filter import OnlineKalmanSpreadTracker


@dataclass
class BacktestConfig:
    entry_z: float = 1.80
    exit_z: float = 0.20
    stop_loss_z: float = 3.50
    slippage_bps: float = 3.0 
    borrow_rate_annual: float = 0.015
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
        prices_df: pd.DataFrame,
        pair_list: list[tuple[str, str]],
    ) -> tuple[pd.DataFrame, dict]:
        t_len = len(prices_df)
        num_pairs = len(pair_list)
        allocated_capital_per_pair = self.cfg.capital_allocation / num_pairs

        portfolio_wealth = np.zeros(t_len)
        portfolio_wealth[0] = self.cfg.capital_allocation
        cash = self.cfg.capital_allocation

        # state storage per pair
        pair_data = []
        ou_engine = OrnsteinUhlenbeckEstimator()

        for p_idx, (y_sym, x_sym) in enumerate(pair_list):
            py = prices_df[y_sym].to_numpy()
            px = prices_df[x_sym].to_numpy()
            log_py = np.log(py)
            log_px = np.log(px)

            kalman = OnlineKalmanSpreadTracker(process_var=1e-8, obs_var=1e-2)
            spreads = np.zeros(t_len)
            betas = np.zeros(t_len)
            z_scores = np.zeros(t_len)
            half_lives = np.zeros(t_len)

            rolling_w = 60
            for t in range(t_len):
                s, a, b = kalman.update(log_py[t], log_px[t])
                spreads[t] = s
                betas[t] = b

                if t >= rolling_w:
                    window = spreads[t - rolling_w : t]
                    mean_s = np.mean(window)
                    std_s = np.std(window)
                    z_scores[t] = (s - mean_s) / std_s if std_s > 1e-6 else 0.0

                    # calibrate OU parameters every 5 days to reduce overhead
                    if t % 5 == 0:
                        fit_res = ou_engine.fit(window)
                        half_lives[t] = fit_res.half_life if fit_res.is_valid else np.inf
                    else:
                        half_lives[t] = half_lives[t - 1]

                else:
                    z_scores[t] = 0.0
                    half_lives[t] = np.inf

            pair_data.append({
                "y_sym": y_sym, "x_sym": x_sym,
                "py": py, "px": px,
                "spread": spreads, "beta": betas,
                "z": z_scores, "hl": half_lives,
                "pos": 0, "units_y": 0.0, "units_x":0.0,
                "entry_bar": 0, "max_holding_bars": 0,
            })

        # portfolio sim loop
        daily_borrow = self.cfg.borrow_rate_annual / 252.0
        cost_rate = self.cfg.slippage_bps * 1e-4
        total_trades = 0
        winning_trades = 0

        for t in range(1, t_len):
            current_portfolio_positions_val = 0.0

            for p_info in pair_data:
                z = p_info["z"][t - 1]
                pos = p_info["pos"]
                hl = p_info["hl"][t - 1]
                py_t = p_info["py"][t]
                px_t = p_info["px"][t]
                beta_t = p_info["beta"][t - 1]

                holding_time = t - p_info["entry_bar"]
                time_stop_triggered = (pos != 0) and (holding_time >= p_info["max_holding_bars"])

                target_pos = pos

                if pos ==0:
                    # entry check
                    if (hl < np.inf) and (np.abs(z) >= self.cfg.entry_z):
                        target_pos = 1 if (z <= -self.cfg.entry_z) else -1
                        # max holding period set to 2.5x mean-reversion hl
                        p_info["max_holding_bars"] = int(np.clip(2.5 * hl * 252.0, 5, 45))
                        p_info["entry_bar"] = t
                elif pos == 1:
                    # exit condition
                    if (z>= -self.cfg.exit_z) or (z <= -self.cfg.stop_loss_z) or time_stop_triggered:
                        target_pos = 0
                elif pos == -1:
                    if (z <= self.cfg.exit_z) or (z >= self.cfg.stop_loss_z) or time_stop_triggered:
                        target_pos = 0

                if target_pos != pos:
                    total_trades += 1
                    # close exit exposure
                    if pos != 0:
                        gross_exit = p_info["units_y"] * py_t + p_info["units_x"] * px_t
                        friction = cost_rate * (np.abs(p_info["units_y"]) * py_t + np.abs(p_info["units_x"]) * px_t)
                        realized_pnl = gross_exit - friction
                        cash += gross_exit -friction
                        if realized_pnl > 0:
                            winning_trades += 1
                        p_info["units_y"] = 0.0
                        p_info["units_x"] = 0.0

                    # open new exposure
                    if target_pos != 0:
                        half_alloc = allocated_capital_per_pair * 0.5
                        uy = (target_pos * half_alloc) / py_t
                        ux = (-target_pos * half_alloc * beta_t) / px_t
                        friction = cost_rate * (np.abs(uy) * py_t + np.abs(ux) * px_t)

                        cash -= (uy * py_t + ux * px_t) + friction
                        p_info["units_y"] = uy
                        p_info["units_x"] = ux
                        p_info["pos"] = target_pos
                    else:
                        p_info["pos"] = 0

                # short borrow costs
                short_val = 0.0
                if p_info["units_y"] < 0:
                    short_val += np.abs(p_info["units_y"]) * py_t
                if p_info["units_x"] < 0:
                    short_val += np.abs(p_info["units_x"]) * px_t
                cash -= short_val * daily_borrow

                current_portfolio_positions_val += p_info["units_y"] * py_t + p_info["units_x"] * px_t

            portfolio_wealth[t] = cash + current_portfolio_positions_val

        # quantitative metric
        returns = np.diff(portfolio_wealth) / portfolio_wealth[:-1]
        total_ret = (portfolio_wealth[-1] - portfolio_wealth[0]) / portfolio_wealth[0]
        n_years = t_len / 252.0
        cagr = (portfolio_wealth[-1] / portfolio_wealth[0]) ** (1.0 / n_years) - 1.0
        ann_vol = float(np.std(returns) * np.sqrt(252.0))
        sharpe = float((cagr - 0.02) / ann_vol) if ann_vol > 1e-4 else 0.0

        acum_peak = np.maximum.accumulate(portfolio_wealth)
        drawdowns = (portfolio_wealth - acum_peak) / acum_peak
        max_dd = float(np.min(drawdowns))
        win_rate = float(winning_trades / (total_trades / 2.0)) if total_trades >= 2 else 0.0

        metrics = {
            "Total Return": total_ret,
            "CAGR": cagr,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Completed Trades": int(total_trades // 2),
            "Win Rate": win_rate,
        }
        return pd.DataFrame({"Wealth": portfolio_wealth}), metrics