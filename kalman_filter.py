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
    def __init__(
        self,
        delta_noise: float = 1e-4,
        obs_variance: float = 1e-3,
        initial_beta: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        delta_noise : float
            Process variance tuning parameter scaling W.
        obs_variance : float
            Observation variance V.
        """
        self.obs_variance = obs_variance
        # State transition: identity random walk
        # theta = [alpha, beta]^T
        self.state_mean = np.array([0.0, initial_beta], dtype=np.float64)
        # Prior state covariance
        self.state_cov = np.eye(2, dtype=np.float64) * 1.0
        # Process noise covariance W
        self.process_noise = (delta_noise / (1.0 - delta_noise)) * np.eye(2, dtype=np.float64)

    def step(self, y_val: float, x_val: float) -> tuple[float, float, float]:
        """
        Processes a single online bar observation (y_t, x_t).
        
        Returns
        -------
        spread_t : float
            The estimated residual innovation error (y_t - y_hat_t).
        alpha_t : float
            Updated intercept.
        beta_t : float
            Updated hedge ratio.
        """
        # 1. State Prediction (Time Update)
        theta_pred = self.state_mean.copy()
        cov_pred = self.state_cov + self.process_noise

        # 2. Measurement Vector H_t = [1.0, x_val]
        h_matrix = np.array([1.0, x_val], dtype=np.float64)

        # 3. Measurement Prediction & Innovation
        y_hat = float(np.dot(h_matrix, theta_pred))
        innovation = float(y_val - y_hat)

        # 4. Innovation Covariance F_t = H P H^T + V
        innovation_cov = float(h_matrix @ cov_pred @ h_matrix.T + self.obs_variance)

        # 5. Kalman Gain K_t = P H^T / F_t
        kalman_gain = (cov_pred @ h_matrix) / innovation_cov

        # 6. Posterior State Update
        self.state_mean = theta_pred + kalman_gain * innovation
        self.state_cov = cov_pred - np.outer(kalman_gain, h_matrix @ cov_pred)

        current_alpha = float(self.state_mean[0])
        current_beta = float(self.state_mean[1])

        return innovation, current_alpha, current_beta
    