"""
Module: spectral_denoising
Implements Marčenko-Pastur analytical fitting, spectral boundary extraction,
and constant trace shrinkage for empirical correlation matrices.
"""

from dataclasses import dataclass
import numpy as np
import scipy.optimize as opt


@dataclass(frozen=True)
class SpectralBounds:
    lambda_min: float
    lambda_max: float
    noise_variance: float


class MarchenkoPasturFilter:
    def __init__(self, q_factor: float) -> None:
        """
        Parameters
        ----------
        q_factor : float
            Ratio of sample observations to variables (T / N). Must be >= 1.0.
        """
        if q_factor < 1.0:
            raise ValueError(f"Q factor must be >= 1.0, provided: {q_factor: .2f}")
        self.q = q_factor

    def compute_theoretical_bounds(self, sigma_sq: float = 1.0) -> tuple[float, float]:
        inv_q_sqrt = np.sqrt(1.0 / self.q)
        lambda_min = sigma_sq * (1.0 - inv_q_sqrt) ** 2
        lambda_max = sigma_sq * (1.0 + inv_q_sqrt) ** 2
        return lambda_min, lambda_max

    def denoise(self, empirical_corr: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(empirical_corr)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[idx], 1e-10)
        eigenvectors = eigenvectors[:, idx]

        # noise variance fit via bounded optimazition on noise bulk
        def loss(s_cand: float) -> float:
            l_min, l_max = self.compute_theoretical_bounds(s_cand)
            bulk = eigenvalues[(eigenvalues >= l_min) & (eigenvalues <= l_max)]
            if len(bulk) == 0:
                return 1e4
            theoretical_var = (s_cand**2) * ((1.0 + 1.0 / self.q) - (1.0 - 1.0 / self.q)**2 / 2.0)
            return float((np.var(bulk) - theoretical_var)**2)

        res = opt.minimize_scalar(loss, bounds=(0.01, 2.0), method="bounded")
        sigma_sq = float(res.x)
        _, lambda_max = self.compute_theoretical_bounds(sigma_sq)

        signal_mask = eigenvalues > lambda_max
        cleaned_eigenvalues = eigenvalues.copy()
        if np.sum(~signal_mask) > 0:
            # constant trace preservation across noise subspace
            cleaned_eigenvalues[~signal_mask] = np.mean(eigenvalues[~signal_mask])

        cleaned_corr = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T
        inv_sqrt_diag = 1.0 / np.sqrt(np.diag(cleaned_corr))
        cleaned_corr = cleaned_corr * np.outer(inv_sqrt_diag, inv_sqrt_diag)
        np.fill_diagonal(cleaned_corr, 1.0)
        return cleaned_corr