"""
Module: ou_calibration
Calibrates 1D Ornstein-Uhlenbeck stochastic parameters via exact analytical MLE.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OUParameters:
    theta: float
    mu: float
    sigma: float
    half_life: float
    asymptotic_variance: float
    is_valid: bool


class OrnsteinUhlenbeckEstimator:
    def __init__(self, dt: float = 1.0 / 252.0) -> None:
        self.dt = dt

    def fit(self, spread_series: np.ndarray) -> OUParameters:
        """
        Calculates exact analytical MLE for dS_t = theta * (mu - S_t) dt + sigma * dW_t
        via conditional AR(1) linear discretization.
        """
        n = len(spread_series)
        if n < 10:
            return OUParameters(0.0, 0.0, 0.0, np.inf, 0.0, False)

        x_prev = spread_series[:-1]
        x_curr = spread_series[1:]
        m = n - 1

        sum_x = np.sum(x_prev)
        sum_y = np.sum(x_curr)
        sum_xx = np.sum(x_prev * x_prev)
        sum_yy = np.sum(x_curr * x_curr)
        sum_xy = np.sum(x_prev * x_curr)

        denom = m * sum_xx - sum_x**2
        if np.abs(denom) < 1e-14:
            return OUParameters(0.0, 0.0, 0.0, np.inf, 0.0, False)

        a = (m * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - a * sum_x) / m

        # Check for non-stationarity / unit root or explosive behavior
        if a <= 0.0 or a >= 1.0:
            return OUParameters(0.0, 0.0, 0.0, np.inf, 0.0, False)

        residuals = x_curr - a * x_prev - b
        sigma_eps_sq = np.sum(residuals**2) / m

        # Exact continuous parameter inversions
        theta = -np.log(a) / self.dt
        mu = b / (1.0 - a)
        sigma_sq = sigma_eps_sq * (-2.0 * np.log(a)) / (self.dt * (1.0 - a**2))
        sigma = np.sqrt(np.maximum(sigma_sq, 1e-14))

        half_life = np.log(2.0) / theta
        asymptotic_variance = sigma**2 / (2.0 * theta)

        return OUParameters(
            theta=theta,
            mu=mu,
            sigma=sigma,
            half_life=half_life,
            asymptotic_variance=asymptotic_variance,
            is_valid=True,
        )
    