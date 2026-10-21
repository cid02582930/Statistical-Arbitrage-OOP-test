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
    half_life: float
    is_valid: bool

class OrnsteinUhlenbeckEstimator:
    def __init__(self, dt: float = 1.0 / 252.0) -> None:
        self.dt = dt

    def fit(self, s: np.ndarray) -> OUParameters:
        n = len(s)
        if n < 30:
            return OUParameters(0.0, 0.0, np.inf, False)

        x = s[:-1]
        y = s[1:]
        z = n - 1

        denom = z * np.sum(x * x) - np.sum(x)**2
        if np.abs(denom) < 1e-12:
            return OUParameters(0.0, 0.0, np.inf, False)

        a = (z * np.sum(x * y) - np.sum(x) * np.sum(y)) / denom
        b = (np.sum(y) - a * np.sum(x)) / z

        # stationary condition for mean-reversion continuous
        if not(0.01 < a < 0.99):
            return OUParameters(0.0, 0.0, np.inf, False)

        theta = -np.log(a) / self.dt
        mu = b / (1.0 - a)
        half_life = np.log(2.0) / theta

        # half life bound
        is_valid = (2.0 / 252.0) <= half_life <= (35.0 / 252.0)
        return OUParameters(theta=theta, mu=mu, half_life=half_life, is_valid=is_valid)