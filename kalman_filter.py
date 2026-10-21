"""
Module: kalman_filter
Implements a state-space recursive online filter tracking dynamic cointegrating hedge ratios.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class KalmanState:
    alpha: float
    beta: float
    cov: np.ndarray

class OnlineKalmanSpreadTracker:
    def __init__(self, process_var: float = 1e-8, obs_var: float = 1e-2) -> None:
        # theta = [alpha, beta]^T
        self.state = np.array([0.0, 1.0], dtype=np.float64)
        self.cov = np.eye(2, dtype=np.float64) * 0.1
        # set W low decouple macro beta drift from mean-reverting spread
        self.W = np.eye(2, dtype=np.float64) * process_var
        self.V = obs_var

    def update(self, log_y: float, log_x: float) -> tuple[float, float, float]:
        state_pred = self.state.copy()
        cov_pred = self.cov + self.W

        H = np.array([1.0, log_x], dtype=np.float64)
        y_hat = float(np.dot(H, state_pred))
        innovation = log_y - y_hat
        F = float(H @ cov_pred @ H.T + self.V)

        K = (cov_pred @ H) / F
        self.state = state_pred + K * innovation
        self.cov = cov_pred - np.outer(K, H @ cov_pred)

        alpha, beta = self.state[0], self.state[1]

        spread = log_y - (alpha + beta * log_x)
        return spread, alpha, beta